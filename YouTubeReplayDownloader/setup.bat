@echo off
setlocal

echo Installing Python dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Checking FFmpeg...
python -c "from ffmpeg_utils import ensure_ffmpeg_available; print('FFmpeg ready:', ensure_ffmpeg_available())"
if errorlevel 1 goto :error

echo.
echo Setup complete. Edit main.py, then run: python main.py
goto :done

:error
echo Setup failed.
exit /b 1

:done
endlocal
