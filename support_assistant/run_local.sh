#!/usr/bin/env bash
set -euo pipefail
python -m support_assistant.ingest
exec uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
