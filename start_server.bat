@echo off
cd /d D:\Trinitas
D:\Trinitas\.venv\Scripts\python.exe -c "from core.integrity_guard import ensure_integrity; ensure_integrity()" >nul 2>&1
D:\Trinitas\.venv\Scripts\python.exe -m uvicorn trinitas_server:app --host 0.0.0.0 --port 8181
