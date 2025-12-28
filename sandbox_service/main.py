import os
import sys
import pty
import select
import subprocess
import shlex
import struct
import fcntl
import termios
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import multiprocessing
import io
import contextlib
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (Giữ nguyên phần execute code cũ - CodeRequest, ExecutionResult, execute_code_worker, run_code) ...
# Để ngắn gọn, tôi chỉ hiển thị phần THÊM MỚI dưới đây. 
# Trong thực tế, bạn hãy GIỮ LẠI các hàm run_code cũ nhé!
# ----------------------------------------------------------------

class CodeRequest(BaseModel):
    code: str
    stdin: str = ""

class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    success: bool
    error: str = ""

def execute_code_worker(code, stdin_input, result_queue):
    # (Giữ nguyên logic cũ của bạn ở đây)
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    # Chuẩn hóa input: thay thế literal \n thành xuống dòng thật nếu cần
    if stdin_input:
        stdin_input = stdin_input.replace("\\n", "\n")
    stdin_capture = io.StringIO(stdin_input)
    success = False
    error_msg = None
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            sys.stdin = stdin_capture
            global_scope = {
                "__builtins__": __builtins__,
                "print": print, "input": input, "range": range, "len": len,
            }
            exec(code, global_scope)
            success = True
    except Exception:
        error_msg = traceback.format_exc()
        success = False
        stderr_capture.write(error_msg)
    finally:
        result_queue.put({
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "success": success,
            "error": error_msg
        })

@app.post("/run", response_model=ExecutionResult)
async def run_code(request: CodeRequest):
    # (Giữ nguyên logic cũ của bạn ở đây)
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code_worker, args=(request.code, request.stdin, queue))
    process.start()
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join()
        return ExecutionResult(stdout="", stderr="Time Limit Exceeded", success=False, error="Timeout")
    if not queue.empty():
        result = queue.get()
        return ExecutionResult(stdout=result["stdout"], stderr=result["stderr"], success=result["success"], error=str(result["error"] or ""))
    return ExecutionResult(stdout="", stderr="Crash", success=False, error="Crash")


# --- PHẦN MỚI: WEBSOCKET TERMINAL ---

async def _forward_output(fd, websocket: WebSocket):
    """Đọc từ pty master fd và gửi qua websocket"""
    loop = asyncio.get_event_loop()
    max_read_bytes = 1024
    
    while True:
        try:
            # Dùng run_in_executor để đọc file blocking trong async
            data = await loop.run_in_executor(None, lambda: os.read(fd, max_read_bytes))
            if not data:
                break
            await websocket.send_text(data.decode('utf-8', errors='replace'))
        except OSError:
            break
        except Exception as e:
            logger.error(f"Error reading from pty: {e}")
            break

@app.websocket("/terminal")
async def terminal_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Tạo pseudo-terminal (pty)
    # Master là phía server đọc/ghi, Slave là phía process con dùng làm stdio
    master_fd, slave_fd = pty.openpty()
    
    # Chạy process con (shell hoặc python interactive)
    # Sử dụng 'python -i' để vào chế độ interactive, hoặc '/bin/sh'
    # Ở đây ta dùng shell để mạnh mẽ hơn
    p = subprocess.Popen(
        ["/bin/sh"],
        preexec_fn=os.setsid,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        universal_newlines=True
    )
    
    # Đóng slave_fd ở process cha (vì process con đã giữ rồi)
    os.close(slave_fd)
    
    # Task đọc output từ process -> gửi cho client
    output_task = asyncio.create_task(_forward_output(master_fd, websocket))
    
    try:
        while True:
            # Nhận input từ client -> ghi vào process
            data = await websocket.receive_text()
            
            # Xử lý resize terminal (nếu client gửi lệnh đặc biệt)
            # Ví dụ protocol json: {"type": "resize", "cols": 80, "rows": 24}
            # Ở demo này nhận raw text
            
            os.write(master_fd, data.encode('utf-8'))
            
    except WebSocketDisconnect:
        logger.info("Websocket disconnected")
    except Exception as e:
        logger.error(f"Websocket error: {e}")
    finally:
        # Cleanup
        output_task.cancel()
        os.close(master_fd)
        if p.poll() is None:
            p.terminate()
            p.wait()
