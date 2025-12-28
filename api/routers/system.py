"""System/utility endpoints
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from infra.services import DockerManager
from app.settings import (
	APP_VERSION,
	EXEC_ALLOWED_LIBRARIES,
	EXEC_CPU_LIMIT_PERCENT,
	EXEC_MEMORY_LIMIT_MB,
	EXEC_NETWORK_ACCESS,
	EXEC_TIMEOUT_SECONDS,
	ENABLE_WS_TERMINAL,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "service": "PyTutor AI Backend", "version": APP_VERSION}

@router.get("/api/config")
async def get_config():
	return {
		"cpu_limit_percent": EXEC_CPU_LIMIT_PERCENT,
		"memory_limit_mb": EXEC_MEMORY_LIMIT_MB,
		"timeout_seconds": EXEC_TIMEOUT_SECONDS,
		"network_access": EXEC_NETWORK_ACCESS,
		"allowed_libraries": EXEC_ALLOWED_LIBRARIES,
		"enable_ws_terminal": ENABLE_WS_TERMINAL,
	}


@router.websocket("/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    # Dùng để mở terminal trong docker
    if not ENABLE_WS_TERMINAL:
        await websocket.accept()
        await websocket.send_text("WebSocket terminal is disabled on this environment.")
        await websocket.close()
        return

    await websocket.accept()
    container = None
    sock = None
    code_file = None

    try:
        init_msg = await websocket.receive_text()
        try:
            obj = json.loads(init_msg)
            if obj.get("type") != "start" or "code" not in obj:
                await websocket.send_text("ERROR: expected start message with code")
                await websocket.close()
                return
            code = obj["code"]
        except Exception:
            await websocket.send_text("ERROR: invalid start message")
            await websocket.close()
            return

        container, sock, code_file = _docker_manager.create_interactive_container(code)
        if not sock:
            await websocket.send_text("ERROR: failed to attach to container")
            await websocket.close()
            return

        sock_reader = getattr(sock, "_sock", sock)
        # Dùng running loop để tránh warning/deprecation.
        loop = asyncio.get_running_loop()

        async def read_from_container():
            """Đọc output từ container và gửi về WebSocket."""
            try:
                while True:
                    data = await loop.run_in_executor(None, sock_reader.recv, 4096)
                    if not data:
                        break
                    text = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else str(data)
                    await websocket.send_text(text)
            except Exception:
                pass
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

        async def read_from_websocket():
            """Đọc input từ WebSocket và gửi vào container."""
            try:
                while True:
                    msg = await websocket.receive_text()
                    if not msg:
                        continue

                    # Parse message (có thể là JSON {type: input, data: ...} hoặc raw text)
                    input_data = msg
                    try:
                        parsed = json.loads(msg)
                        if isinstance(parsed, dict) and parsed.get("type") == "input":
                            input_data = parsed.get("data", "")
                    except (json.JSONDecodeError, TypeError):
                        pass

                    if input_data:
                        try:
                            sock_writer = getattr(sock, "_sock", sock)
                            await loop.run_in_executor(None, sock_writer.sendall, input_data.encode("utf-8"))
                        except Exception:
                            pass
            except (WebSocketDisconnect, Exception):
                return

        reader_task = asyncio.create_task(read_from_container())
        ws_task = asyncio.create_task(read_from_websocket())

        done, pending = await asyncio.wait([reader_task, ws_task], return_when=asyncio.FIRST_COMPLETED)

        for t in pending:
            t.cancel()

    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup resources
        if sock:
            try:
                sock.close()
            except Exception:
                pass

        if container:
            try:
                container.kill()
            except Exception:
                pass
            try:
                container.remove(force=True)
            except Exception:
                pass

        if code_file:
            try:
                if os.path.exists(code_file):
                    os.remove(code_file)
            except Exception:
                pass


__all__ = ["router"]
