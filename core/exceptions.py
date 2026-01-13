from fastapi import Request
from fastapi.responses import JSONResponse

async def global_exception_handler(request: Request, exc: Exception):
    print("🔥 UNHANDLED SERVER ERROR")
    print(str(exc))

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Something went wrong. Please try again later."
        }
    )
