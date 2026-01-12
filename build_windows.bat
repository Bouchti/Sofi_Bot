@echo off
setlocal

REM Build standalone Windows binary with PyInstaller.
pyinstaller --noconfirm --clean --onefile --windowed ^
  --collect-submodules discum ^
  --hidden-import discum.start ^
  --hidden-import discum.start.superproperties ^
  --hidden-import discum.start.login ^
  --hidden-import certifi ^
  sofi_bot_gui_V2.py

endlocal
