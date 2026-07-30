@echo off
setlocal

REM The py launcher is used rather than "python" or "pip" because on Windows
REM those are frequently absent from PATH even when Python is installed.

py --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.11+ from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" on the first installer screen.
    goto :error
)

echo Installing Python dependencies...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Checking FFmpeg...
py -c "from ffmpeg_utils import ensure_ffmpeg_available; print('FFmpeg ready:', ensure_ffmpeg_available())"
if errorlevel 1 (
    echo FFmpeg was not found. Install it with: winget install Gyan.FFmpeg
    echo Then close and reopen this window.
    goto :error
)

echo.
echo Setup complete. Edit YOUTUBE_URL in main.py, then run: py main.py
goto :done

:error
echo Setup failed.
exit /b 1

:done
endlocal
