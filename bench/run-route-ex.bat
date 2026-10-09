@echo off
rem Full run of the route1 and ex1 GLiNER arms, on the machine with the GPU.
rem Run from the repository root (or a copy with bench\, requests\route1\ and cases\ex1\):
rem   set PYTHON=<venv>\Scripts\python.exe
rem   bench\run-route-ex.bat
rem Needs pip install "gliner2[local]" (2.0.0) and torch with CUDA. HF_HOME may point to a model cache.
rem HOST_LABEL is what the host field of every row says (default: local).
setlocal
cd /d "%~dp0.."
if "%PYTHON%"=="" set PYTHON=python
if "%HOST_LABEL%"=="" set HOST_LABEL=local
set PYTHONIOENCODING=utf-8

%PYTHON% bench\run-gliner.py --pv route1 --model fastino/GLiNER2.5-multi-Decide --arm gliner2.5-multi-decide --mode prompt --host %HOST_LABEL% || goto :fail
%PYTHON% bench\run-gliner.py --pv route1 --model fastino/GLiNER2.5-multi-Decide --arm gliner2.5-multi-decide-intext --mode intext --host %HOST_LABEL% || goto :fail
%PYTHON% bench\run-gliner.py --pv route1 --model fastino/GLiNER2.5-Decide --arm gliner2.5-decide --mode prompt --host %HOST_LABEL% || goto :fail
%PYTHON% bench\run-gliner.py --pv route1 --model fastino/GLiNER2.5-Decide --arm gliner2.5-decide-intext --mode intext --host %HOST_LABEL% || goto :fail
%PYTHON% bench\run-gliner.py --pv route1 --model fastino/GLiNER2.5-Decide-1B --arm gliner2.5-decide-1b --mode prompt --host %HOST_LABEL% || goto :fail
%PYTHON% bench\run-gliner.py --pv route1 --model fastino/GLiNER2.5-Decide-1B --arm gliner2.5-decide-1b-intext --mode intext --host %HOST_LABEL% || goto :fail
%PYTHON% bench\run-gliner-extract.py --model fastino/gliner2.5-multi-v1 --arm gliner2.5-multi --host %HOST_LABEL% || goto :fail
%PYTHON% bench\run-gliner-extract.py --model fastino/gliner2-large-v1 --arm gliner2-large --lang en --host %HOST_LABEL% || goto :fail
echo GLiNER side done. Copy results\route1\ and results\ex1\ to the scoring machine.
exit /b 0
:fail
echo A GLiNER run failed, see above. 1>&2
exit /b 1
