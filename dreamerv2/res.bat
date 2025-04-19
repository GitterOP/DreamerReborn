@echo off
set PROJECT_PATH=C:\Users\cuent\DV2_again_bro\DreamerReborn\dreamerv2

:inicio
docker run -it --rm --gpus all ^
  -v %USERPROFILE%/logdir:/logdir ^
  -v %PROJECT_PATH%:/app ^
  dreamerv2 python3 /app/dreamerv2/train.py ^
  --logdir /logdir/atari_pong/dreamerv2/1 ^
  --configs atari ^
  --task atari_ms_pacman

echo.
set /p RES=Presiona [Enter] para reiniciar o escribe [n] para salir: 
if /i "%RES%"=="n" exit
goto inicio