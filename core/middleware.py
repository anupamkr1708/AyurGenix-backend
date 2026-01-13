import time
import uuid
from fastapi import Request

async def request_logger(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    response = await call_next(request)

    process_time = round(time.time() - start_time, 4)

    print(
        f"🧾 [{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} | {process_time}s"
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    return response
