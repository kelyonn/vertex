@echo off
REM Project Vertex v4 — JARVIS Hologram Workbench (Windows)

cd /d "%~dp0"
echo >> STARTING PROJECT VERTEX v4.0
echo >> ----------------------------

REM Detect Python
set PYTHON_CMD=python
where python3.11 >nul 2>&1 && set PYTHON_CMD=python3.11

REM Create virtual environment if needed
if not exist venv (
    echo >> Creating virtual environment...
    %PYTHON_CMD% -m venv venv
)

set VENV_PYTHON=%~dp0venv\Scripts\python.exe

REM Install / sync dependencies
echo >> Syncing dependencies...
%VENV_PYTHON% -m pip install -q --upgrade pip
%VENV_PYTHON% -m pip install -q -r requirements.txt

REM Download MediaPipe hand landmarker model if missing
if not exist src\hand_landmarker.task (
    echo >> Downloading MediaPipe hand landmarker model...
    curl -L -o src\hand_landmarker.task ^
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
)

if not exist models mkdir models

echo >> Launching application...
%VENV_PYTHON% src\main.py %*
