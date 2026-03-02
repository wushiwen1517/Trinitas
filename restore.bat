@echo off
chcp 65001 >nul
echo ========================================
echo   Trinitas 代码恢复工具
echo ========================================
echo.
echo 这将用备份覆盖当前代码！
echo 备份源: D:\Trinitas_backup
echo.
set /p confirm=确认恢复？(Y/N): 
if /i not "%confirm%"=="Y" (
    echo 已取消
    pause
    exit /b
)

set BACKUP_DIR=D:\Trinitas_backup
set DST=D:\Trinitas

echo 正在恢复核心代码...
copy /Y "%BACKUP_DIR%\core\*.py" "%DST%\core\" >nul
copy /Y "%BACKUP_DIR%\trinitas_server.py" "%DST%\" >nul
copy /Y "%BACKUP_DIR%\web\index.html" "%DST%\web\" >nul

echo.
echo [OK] 恢复完成！请重启服务器。
echo.
pause
