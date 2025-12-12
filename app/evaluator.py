import os
import openai
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY không được cấu hình")
openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "Bạn là một chuyên gia kiểm định giáo dục. Dựa trên 'Yêu cầu' và các 'Minh chứng' được liệt kê, "
    "viết một báo cáo tiếng Việt (tóm tắt nhận định, điểm chính, 3 khuyến nghị hành động). "
    "Trả về đúng JSON (không có text khác) theo schema: "
    '{"report":"...","evidences":[{"id":"...","score":0.0,"snippet":"..."}]}'
)

def build_user_prompt(query: str, evidences: List[Dict[str, Any]]) -> str:
    ev_texts = ""
    for i, ev in enumerate(evidences, start=1):
        ev_texts += f"{i}. id: {ev['id']}, score: {ev['score']:.4f}\nsnippet: {ev['snippet']}\n\n"
    return f"Yêu cầu / Tiêu chí:\n{query}\n\nMinh chứng tìm được (top {len(evidences)}):\n{ev_texts}\n\nHãy tạo báo cáo và liệt kê các minh chứng đã dùng."

def call_chat(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.0,
        max_tokens=900
    )
    return resp.choices[0].message.content

def generate_report(query: str, evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(query, evidences)
    raw = call_chat(SYSTEM_PROMPT, user_prompt)
    # try to extract JSON
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        json_text = raw[start:end]
        result = json.loads(json_text)
    except Exception as e:
        # fallback: embed raw into response for debugging
        raise RuntimeError(f"Không parse được output từ model: {e}. Raw output: {raw[:1000]}")
    # attach audit info and returned evidence list (with normalized fields)
    result["_audit"] = {"model": "openai:gpt-4o-mini", "raw_output": raw[:2000]}
    # ensure evidences exist
    result["evidences"] = result.get("evidences", evidences)
    return result
