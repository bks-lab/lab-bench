@echo off
rem C9 preparation on the workstation, CPU only, before any C9 job (plan/c9.md, notes of 2026-10-12).
rem   set PY312=<python.exe of the system Python 3.12>  then  prep-pc.bat <folder holding docs.jsonl from bench/c9/freeze.py>
rem - D:\bench-data\c9\docs.jsonl: the frozen documents (every job checks the hash in cases/c9/manifest.json)
rem - D:\bench-cap\.venv-presidio: a new venv from the system Python with Presidio and spaCy de_core_news_lg (CPU arm)
rem - D:\bench-cap\gliner1: the gliner package (v1) without dependencies, put in front of the bench venv together
rem   with tf4 (transformers 4.57.6; gliner 0.2.29 needs transformers below 5.17)
rem - model snapshots at the pinned revisions into HF_HOME=D:\hf (the jobs run offline)
rem gliner (v1) reads the config of its backbone microsoft/mdeberta-v3-base by name, without a revision, so the
rem cache needs refs/main as well (main was a0484667b223 on 2026-10-12; the second call writes it).
rem The pip configuration of the workstation names an index that does not resolve: PIP_CONFIG_FILE=nul.
setlocal
if "%PY312%"=="" (echo set PY312 first & exit /b 2)
set PIP_CONFIG_FILE=nul
set HF_HOME=D:\hf
set BENCH=D:\bench-cap\.venv\Scripts\python.exe
if not "%~1"=="" (if not exist D:\bench-data\c9 mkdir D:\bench-data\c9) & if not "%~1"=="" copy /Y "%~1\docs.jsonl" D:\bench-data\c9\ >nul

if not exist D:\bench-cap\.venv-presidio\Scripts\python.exe "%PY312%" -m venv D:\bench-cap\.venv-presidio || exit /b 1
D:\bench-cap\.venv-presidio\Scripts\python.exe -m pip install --quiet --disable-pip-version-check presidio-analyzer==2.2.364 spacy==3.8.16 || exit /b 1
D:\bench-cap\.venv-presidio\Scripts\python.exe -m pip install --quiet --disable-pip-version-check https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl || exit /b 1

%BENCH% -m pip install --quiet --disable-pip-version-check --no-deps --target D:\bench-cap\gliner1 gliner==0.2.29 sentencepiece==0.2.1 || exit /b 1

%BENCH% -c "from huggingface_hub import snapshot_download as s; print(s('urchade/gliner_multi_pii-v1', revision='1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d')); print(s('E3-JSI/gliner-multi-pii-domains-v1', revision='66b7951b1e020535d2f470d4b35bc90149a1ed0d', allow_patterns=['*.json','*.model','model.safetensors','README.md'])); print(s('microsoft/mdeberta-v3-base', revision='a0484667b22365f84929a935b5e50a51f71f159d', allow_patterns=['*.json','*.model','*.txt'])); print(s('microsoft/mdeberta-v3-base', allow_patterns=['*.json','*.model','*.txt']))" || exit /b 1
echo prep done
