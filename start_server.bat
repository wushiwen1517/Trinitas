@echo off
cd /d D:\Trinitas
D:\Trinitas\.venv\Scripts\python.exe -m uvicorn trinitas_server:app --host 0.0.0.0 --port 8181