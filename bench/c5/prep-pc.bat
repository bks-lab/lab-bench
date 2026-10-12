@echo off
rem C5 preparation on the workstation, before any C5 job (plan/c5.md, notes of 2026-10-12).
rem   set PY312=<python.exe of the system Python 3.12>  then  prep-pc.bat <folder holding deu_Latn.jsonl and eng_Latn.jsonl>
rem - D:\bench-data\c5: the two gated FLORES+ devtest files (hash-checked by every job against cases/c5/manifest.json)
rem - D:\bench-cap\.venv-c5: a new venv with torch (CUDA 12.8), transformers 4.x and sentencepiece through
rem   unbabel-comet 2.2.7 (which pins transformers below 5 and numpy below 2), and sacrebleu; it runs the two
rem   dedicated MT arms, the sanity gate and the COMET pass. The bench venv stays untouched.
rem - model snapshots at the pinned revisions into HF_HOME=D:\hf; the EuroLLM GGUF through Ollama
rem The pip configuration of the workstation names an index that does not resolve: PIP_CONFIG_FILE=nul.
setlocal
if "%PY312%"=="" (echo set PY312 first & exit /b 2)
if "%~1"=="" (echo give the folder with the two FLORES+ files & exit /b 2)
set PIP_CONFIG_FILE=nul
set HF_HOME=D:\hf
set V=D:\bench-cap\.venv-c5\Scripts\python.exe

if not exist D:\bench-data\c5 mkdir D:\bench-data\c5
copy /Y "%~1\deu_Latn.jsonl" D:\bench-data\c5\ >nul || exit /b 1
copy /Y "%~1\eng_Latn.jsonl" D:\bench-data\c5\ >nul || exit /b 1

if not exist %V% "%PY312%" -m venv D:\bench-cap\.venv-c5 || exit /b 1
%V% -m pip install --quiet --disable-pip-version-check torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
echo torch==2.11.0+cu128> D:\bench-cap\.venv-c5\constraints.txt
%V% -m pip install --quiet --disable-pip-version-check -c D:\bench-cap\.venv-c5\constraints.txt unbabel-comet==2.2.7 sacrebleu==2.6.0 sentencepiece==0.2.1 || exit /b 1
%V% -c "import torch, transformers, comet, sacrebleu; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, sacrebleu.__version__)" || exit /b 1

%V% -c "from huggingface_hub import snapshot_download as s; print(s('Helsinki-NLP/opus-mt-de-en', revision='1a922f3b32a8e809e17a47d4b32142d8105924e5', allow_patterns=['*.json','*.spm','pytorch_model.bin'])); print(s('Helsinki-NLP/opus-mt-en-de', revision='6183067f769a302e3861815543b9f312c71b0ca4', allow_patterns=['*.json','*.spm','pytorch_model.bin'])); print(s('google/madlad400-3b-mt', revision='fa184c675da0b5c9e1c8694fccd4e12e2d422094', allow_patterns=['*.json','spiece.model','model.safetensors'])); print(s('Unbabel/wmt22-comet-da', revision='2760a223ac957f30acfb18c8aa649b01cf1d75f2')); print(s('xlm-roberta-large', allow_patterns=['*.json','*.model']))" || exit /b 1

ollama pull hf.co/bartowski/EuroLLM-9B-Instruct-GGUF:Q8_0 || exit /b 1
echo prep done
