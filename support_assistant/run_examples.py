from __future__ import annotations

import json

from .graph import ask

for q in ["How much is standard delivery?", "What is the capital of India?"]:
    print(json.dumps({"query": q, "response": ask(q).model_dump()}, ensure_ascii=False))
