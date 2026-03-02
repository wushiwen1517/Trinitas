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
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [WARN] 当前目录不是 Git 仓库，已跳过 commit/push
    goto :done
)

git add -A >nul 2>&1
git diff --cached --quiet >nul 2>&1
if not errorlevel 1 (
    echo [OK] 无改动，跳过 commit
) else (
    git -c user.name="Trinitas Agent" -c user.email="trinitas-agent@local" commit -m "Auto backup %date% %time%" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] git commit 失败（可能无改动/权限/仓库异常）
    ) else (
        echo [OK] git commit 完成
    )
)

echo 正在 git push...
set HTTPS_PROXY=
set HTTP_PROXY=
set https_proxy=
set http_proxy=
git push >nul 2>&1
if errorlevel 1 (
    echo [WARN] git push 失败（可能网络/代理/凭据问题）
) else (
    echo [OK] git push 完成
)

:done
echo.
echo [OK] 备份完成！
echo   - 物理备份: %BACKUP_DIR%
echo   - Git: 已尝试提交并推送
echo.
pause
