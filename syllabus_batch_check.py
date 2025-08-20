import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ===== モデル準備 =====
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True
)

# ===== JSONファイル読み込み =====
with open("syllabus_2025_full.json", "r", encoding="utf-8") as f:
    syllabus_data = json.load(f)

# ===== 質問設定 =====
question = "この授業の内容はタイトルにふさわしいですか？"

# ===== プロンプト + 実行 =====
def make_prompt(course):
    title = course.get("title", "（タイトルなし）")
    overview = course.get("overview_jp", "")
    goals = course.get("goals", "")
    schedule = course.get("schedule", "")
    
    prompt = f"""以下は大学の授業シラバス情報です：

【授業名】：{title}
【授業概要】：{overview}
【到達目標】：{goals}
【授業計画】：{schedule}

この内容に基づき、以下の質問に1文だけ日本語で簡潔に答えてください。

質問：{question}
回答："""
    return prompt

# ===== モデル呼び出し関数 =====
def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "回答：" in response:
        answer_full = response.split("回答：", 1)[1].split("質問：")[0].strip()
    else:
        answer_full = response.strip()

    sentence_end = re.split(r"[。！？\?]", answer_full)
    answer = sentence_end[0].strip() + "。" if sentence_end[0] else answer_full.strip()

    return answer

# ===== 一括処理 & 結果保存 =====
results = []

for i, course in enumerate(syllabus_data):
    try:
        prompt = make_prompt(course)
        answer = query_model(prompt)
        title = course.get("title", "（タイトルなし）")
        results.append({
            "index": i + 1,
            "title": title,
            "question": question,
            "answer": answer
        })
        print(f"[{i+1}] {title} → {answer}")
    except Exception as e:
        print(f"[{i+1}] エラー: {e}")
        results.append({
            "index": i + 1,
            "title": course.get("title", "（タイトルなし）"),
            "question": question,
            "answer": f"エラー: {e}"
        })

# ===== 結果をJSONで保存 =====
with open("syllabus_check_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ 完了：syllabus_check_result.json に保存しました。")
