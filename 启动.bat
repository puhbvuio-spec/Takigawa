@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo ============================================================
echo  Community Feedback Pipeline - LLM 模式
echo  当前仅支持通义千问 Coding Plan（API Key 以 sk-sp- 开头）
echo ============================================================
echo.
set /p DASHSCOPE_API_KEY=请输入 Coding Plan API KEY (sk-sp-...):

if "%DASHSCOPE_API_KEY%"=="" (
  echo.
  echo [ERROR] 未提供 API KEY，已终止运行。
  pause
  exit /b 1
)

echo %DASHSCOPE_API_KEY% | findstr /b "sk-sp-" >nul
if errorlevel 1 (
  echo.
  echo [ERROR] API KEY 格式不正确，当前仅支持 Coding Plan 密钥（以 sk-sp- 开头）。
  echo         请前往 https://bailian.console.aliyun.com/ 获取 Coding Plan 密钥。
  pause
  exit /b 1
)

set CF_LLM_ENABLED=1

echo.
echo [INFO] API 地址: https://coding.dashscope.aliyuncs.com/v1/chat/completions
echo [INFO] 默认模型: qwen3.5-plus
echo [INFO] 正在调用 community_feedback_pipeline.py ...
echo.
python community_feedback_pipeline.py --llm --label-library "label_libraries\star_rail_feedback_labels.v3.0.0.json" --api-key "%DASHSCOPE_API_KEY%"

if errorlevel 1 (
  echo.
  echo [ERROR] 运行失败，请检查以下事项：
  echo         1. API KEY 是否有效（是否过期或欠费）
  echo         2. 网络是否能访问 coding.dashscope.aliyuncs.com
  echo         3. Coding Plan 套餐是否包含 qwen3.5-plus 模型权限
  echo         4. data\inbox 中是否存在有效的输入文件
  pause
  exit /b 1
)

echo 模式运行完成。
pause
