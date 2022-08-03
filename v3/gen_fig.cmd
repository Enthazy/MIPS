@echo off
set a=test/results/F6P115W100T100
set savepath=./test/fig/F6P115W100T100/
echo targetpath is %a%
echo savepath is %savepath%
PAUSE

for %%A in ( %a%/* ) do (
    python gen_fig.py %a%/%%A %savepath%
)
pause