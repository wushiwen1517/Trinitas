# trinitas_server.py
import uuid
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 已关闭启动时 integrity_guard，避免用备份覆盖当前代码。需要时可在本机手动执行 ensure_integrity()。
# try:
#     from core.integrity_guard import ensure_integrity as _ensure_integrity
#     _ensure_integrity()
# except Exception:
#     pass

from core.orchestrator import Orchestrator
from core.config import API_KEY

app = FastAPI()
orch = Orchestrator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def verify_key(request: Request) -> bool:
    return request.headers.get("X-TRINITAS-KEY") == API_KEY

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/verify")
async def verify_api(request: Request):
    if not verify_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return {"status": "ok"}

@app.post("/api/chat")
async def chat_api(request: Request):
    if not verify_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    user_message = (data.get("message") or "").strip()
    chat_id = data.get("chat_id") or str(uuid.uuid4())
    mode = data.get("mode", "auto")

    async def stream():
        async for chunk in orch.handle(
            chat_id=chat_id, message=user_message,
            image_bytes=None, mode=str(mode),
        ):
            yield chunk
    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")

@app.post("/api/vision")
async def vision_api(
    request: Request,
    file: UploadFile = File(...),
    chat_id: str = Form(default=""),
    message: str = Form(default=""),
    mode: str = Form(default="auto"),
):
    if not verify_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        image_bytes = await file.read()
    except Exception:
        return JSONResponse({"error": "File read failed"}, status_code=400)
    if not image_bytes:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    resolved_chat_id = chat_id.strip() or str(uuid.uuid4())

    async def stream():
        async for chunk in orch.handle(
            chat_id=resolved_chat_id, message=message or "",
            image_bytes=image_bytes, mode=mode.strip() or "auto",
        ):
            yield chunk
    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")

app.mount("/", StaticFiles(directory="web", html=True), name="web")
