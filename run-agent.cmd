@echo off
REM Scheduled entry point. Appends every run to agent.log.
REM Register every 6 hours (from this folder):
REM   schtasks /create /tn "FlopAgent%~n0" /tr "%CD%\run-agent.cmd" /sc HOURLY /mo 6 /f
if "%PY%"=="" set PY=python
set DIR=%~dp0
echo. >> "%DIR%agent.log"
echo ===== %DATE% %TIME% ===== >> "%DIR%agent.log"
"%PY%" -u "%DIR%agent.py" --publish --budget 240 >> "%DIR%agent.log" 2>&1
echo exit=%ERRORLEVEL% >> "%DIR%agent.log"
