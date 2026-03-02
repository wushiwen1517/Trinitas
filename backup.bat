@echo off
chcp 65001 >nul
echo ========================================
echo   Trinitas 代码备份工具
echo ========================================

set BACKUP_DIR=D:\Trinitas_backup
set SRC=D:\Trinitas

if not exist "%BACKUP_DIR%\core" mkdir "%BACKUP_DIR%\core"
if not exist "%BACKUP_DIR%\web" mkdir "%BACKUP_DIR%\web"

echo 正在备份核心代码...
copy /Y "%SRC%\core\*.py" "%BACKUP_DIR%\core\" >nul
copy /Y "%SRC%\trinitas_server.py" "%BACKUP_DIR%\" >nul
copy /Y "%SRC%\web\index.html" "%BACKUP_DIR%\web\" >nul
copy /Y "%SRC%\start_server.bat" "%BACKUP_DIR%\" >nul
copy /Y "%SRC%\requirements.txt" "%BACKUP_DIR%\" >nul
copy /Y "%SRC%\Modelfile.*" "%BACKUP_DIR%\" >nul

echo 正在 git commit...
cd /d "%SRC%"
git add -A >nul 2>&1
git commit -m "Auto backup %date% %time%" >nul 2>&1

echo.
echo [OK] 备份完成！
echo   - 物理备份: %BACKUP_DIR%
echo   - Git commit: 已保存
echo.
pause
