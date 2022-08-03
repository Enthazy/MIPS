@echo off
set epoch=999
set name=F0P200W1000T400
set picLoadPath=E:/Workplace/research/MIPS/v3/test/fig/%name%/
set videoSavePath=./video/

python gen_gif.py 0 %epoch% %name%.avi %picLoadPath% %videoSavePath%
pause