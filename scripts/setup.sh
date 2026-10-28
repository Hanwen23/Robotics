#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  candidate="${PYTHON_BIN:-}"
  if [[ -z "$candidate" ]] && command -v python3.12 >/dev/null 2>&1; then
    candidate="$(command -v python3.12)"
  fi
  if [[ -z "$candidate" ]] && [[ -x "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3" ]]; then
    candidate="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
  fi
  if [[ -z "$candidate" ]]; then
    echo 'Python 3.12 is required. Set PYTHON_BIN=/absolute/path/to/python3.12.' >&2
    exit 1
  fi
  "$candidate" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
  "$candidate" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
mkdir -p .cache/pip
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  .venv/bin/python -m pip --isolated install --cache-dir "$PWD/.cache/pip" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.lock.txt
.venv/bin/python -m pip --isolated check
echo 'Environment ready. Download: .venv/bin/python scripts/download_data.py'
