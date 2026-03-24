@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo [INFO] 启动规则模式（不调用 LLM）
echo [INFO] 将读取 data\inbox 下的输入文件
echo [INFO] 默认标签库: label_libraries\star_rail_feedback_labels.v3.0.0.json

python community_feedback_pipeline.py --label-library "label_libraries\star_rail_feedback_labels.v3.0.0.json"

if errorlevel 1 (
  echo.
  echo [ERROR] 运行失败，请检查 Python 环境、依赖或输入文件格式。
  pause
  exit /b 1
)

echo.
echo [DONE] 运行完成。
pause
