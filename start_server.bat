@echo off
cd /d D:\Trinitas
REM 已关闭启动时自检恢复，避免用备份覆盖当前代码。需要恢复时请手动运行 integrity_guard 或从备份复制。
D:\Trinitas\.venv\Scripts\python.exe -m uvicorn trinitas_server:app --host 0.0.0.0 --port 8181
