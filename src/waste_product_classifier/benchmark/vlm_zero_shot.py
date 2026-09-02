import base64
import json
import os
import re
import time

import ollama

from waste_product_classifier.config import DEFAULT_OLLAMA_HOST

OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
client = ollama.Client(OLLAMA_HOST)

PROMPT = """You are a waste-sorting assistant. Look at the image and classify the item as
exactly one of: "recyclable" or "organic".
 
Respond with ONLY a JSON object in this exact format, no extra text:
{"label": "recyclable" | "organic", "confidence": <float 0-1>, "reason": "<one short sentence>"}
"""

def classify_with_vlm(image_path: str, model_name: str = "qwen2.5vl") -> dict:
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    start = time.time()
    response = client.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [base64.b64encode(img_bytes).decode("utf-8")]
            }
        ]
    )
    latency = time.time() - start

    raw_text = response["message"]["content"]
    parsed = parse_json_response(raw_text)

    return {
        "label": parsed.get("label"),
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason"),
        "latency_s": latency,
        "raw_response": raw_text
    }

def parse_json_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"label": None, "confidence": None, "reason": "unparsable"}
    try: 
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": None, "confidence": None, "reason": "unparsable"}

# --- Optional: cloud API fallback (swap in if you'd rather not run a VLM container at all) ---
# def classify_with_vlm_cloud(image_path, api_key):
#     import google.generativeai as genai
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     img = genai.upload_file(image_path)
#     start = time.time()
#     response = model.generate_content([PROMPT, img])
#     latency = time.time() - start
#     parsed = _parse_json_response(response.text)
#     parsed["latency_s"] = latency
#     return parsed