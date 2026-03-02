# core/failure_guard.py

import httpx
import json

async def safe_stream_model(client, url, payload):
    try:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                yield f"\n❌ 模型调用失败 HTTP {r.status_code}\n"
                return

            async for line in r.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except:
                        continue

    except Exception as e:
        yield f"\n❌ 模型异常：{str(e)}\n"