@echo off
REM Phase 2 collector, invoked by the Windows scheduled task.
REM Runs one sampling pass of the congestion logger using the project venv, and
REM appends both the log rows (inside the module) and a run trace here so a
REM failed run is visible. Edit --polls to trade runtime for stability.
cd /d E:\traffic-final
echo ==== run %DATE% %TIME% ====>> congestion_logger_run.log
".venv\Scripts\python.exe" -m routing.congestion_logger --polls 5 >> congestion_logger_run.log 2>&1
