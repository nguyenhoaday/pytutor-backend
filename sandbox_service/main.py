from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import io
import contextlib
import traceback
import signal
import multiprocessing

# --- Cấu hình ---
TIMEOUT_SECONDS = 30  # Giới hạn thời gian chạy 5s

app = FastAPI(title="Python Sandbox Service")

class CodeRequest(BaseModel):
    code: str
    stdin: str = ""

class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    success: bool
    error: str = None

# --- Hàm chạy code (chạy trong process riêng để kill được khi timeout) ---
def execute_code_worker(code, stdin_input, result_queue):
    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Mock stdin
    stdin_capture = io.StringIO(stdin_input)
    
    success = False
    error_msg = None

    try:
        # Redirect standard streams
        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            
            # Gán stdin
            sys.stdin = stdin_capture
            
            # Tạo môi trường chạy code cô lập
            # Chỉ cho phép các hàm an toàn cơ bản (có thể mở rộng sau)
            global_scope = {
                "__builtins__": __builtins__,
                "print": print,
                "input": input,
                "range": range,
                "len": len,
                # Thêm các thư viện phổ biến vào namespace nếu cần
                # "math": math,
            }
            
            # CHẠY CODE!
            exec(code, global_scope)
            success = True

    except Exception:
        # Bắt lỗi runtime (SyntaxError, NameError, v.v.)
        error_msg = traceback.format_exc()
        success = False
        # In lỗi ra stderr ảo để client nhận được
        stderr_capture.write(error_msg)
        
    finally:
        # Trả kết quả về qua Queue
        result_queue.put({
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "success": success,
            "error": error_msg
        })

@app.post("/run", response_model=ExecutionResult)
async def run_code(request: CodeRequest):
    """
    API endpoint để chạy code Python.
    Sử dụng Multiprocessing để có thể kill process nếu chạy quá lâu (timeout).
    """
    queue = multiprocessing.Queue()
    
    # Tạo process con để chạy code
    process = multiprocessing.Process(
        target=execute_code_worker,
        args=(request.code, request.stdin, queue)
    )
    
    process.start()
    
    # Đợi process chạy, có timeout
    process.join(TIMEOUT_SECONDS)
    
    if process.is_alive():
        # Nếu sau 5s vẫn sống -> Kill ngay
        process.terminate()
        process.join()
        return ExecutionResult(
            stdout="",
            stderr="Time Limit Exceeded (Timeout)",
            success=False,
            error="Timeout"
        )
    
    # Lấy kết quả từ queue
    if not queue.empty():
        result = queue.get()
        return ExecutionResult(
            stdout=result["stdout"],
            stderr=result["stderr"],
            success=result["success"],
            error=result["error"]
        )
    else:
        # Trường hợp process chết mà không kịp gửi kq
        return ExecutionResult(
            stdout="",
            stderr="Execution failed unexpectedly (Process crashed)",
            success=False,
            error="Crash"
        )

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "sandbox"}
