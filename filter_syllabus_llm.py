#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter Hosei syllabus JSON by topic using an LLM (OpenAI or Ollama).
- Method 1 (default): Embedding-based semantic search (fast, scalable)
- Method 2: LLM classification (zero-shot; strong semantics, slower)
Fallback: simple keyword filter when no LLM configured.

Usage examples:
  # Embedding search with OpenAI
  python filter_syllabus_llm.py --json syllabus_2025_full.json --query "LLM,生成AI,大規模言語モデル" --provider openai --method embed --top-k 50

  # Embedding search with Ollama (local)
  python filter_syllabus_llm.py --json syllabus_2025_full.json --query "ネットワークセキュリティ" --provider ollama --embed-model nomic-embed-text --method embed --threshold 0.28

  # LLM classification (OpenAI)
  python filter_syllabus_llm.py --json syllabus_2025_full.json --query "情報理論と符号化" --provider openai --method llm

  # Keyword fallback (no LLM required)
  python filter_syllabus_llm.py --json syllabus_2025_full.json --query "C++ OR オブジェクト指向 -入門" --method keyword

Outputs:
  - /mnt/data/filtered_courses.csv
  - /mnt/data/filtered_courses.json
"""
import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

# Optional deps
try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

# HTTP for Ollama fallback
try:
    import requests
except Exception:
    requests = None

# OpenAI SDK (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class Course:
    idx: int
    title: str
    teacher: str
    url: str
    overview_outline: str
    overview_goal: str
    overview_en: str
    teaching_plan_text: str
    raw: Dict[str, Any]

    def to_text(self) -> str:
        parts = [
            f"Title: {self.title}" if self.title else "",
            f"Teacher: {self.teacher}" if self.teacher else "",
            f"Overview: {self.overview_outline}" if self.overview_outline else "",
            f"Goal: {self.overview_goal}" if self.overview_goal else "",
            f"Overview_EN: {self.overview_en}" if self.overview_en else "",
            f"Plan: {self.teaching_plan_text}" if self.teaching_plan_text else "",
            f"URL: {self.url}" if self.url else "",
        ]
        return "\n".join([p for p in parts if p]).strip()


def load_courses(json_path: str) -> List[Course]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of course objects")
    courses: List[Course] = []
    for i, it in enumerate(data):
        if not isinstance(it, dict):
            continue
        title = it.get("title", "") or ""
        teacher = it.get("teacher", "") or ""
        url = it.get("url", "") or ""
        overview_en = it.get("overview_en", "") or ""
        outline = ""
        goal = ""
        if isinstance(it.get("overview_structured"), dict):
            outline = it["overview_structured"].get("outline", "") or ""
            goal = it["overview_structured"].get("goal", "") or ""
        plan_items = it.get("teaching_plan") or []
        plan_texts = []
        if isinstance(plan_items, list):
            for p in plan_items:
                if isinstance(p, dict):
                    c = p.get("content") or ""
                    if c:
                        plan_texts.append(c)
        plan_text = " / ".join(plan_texts[:30])  # cap length
        courses.append(Course(
            idx=i,
            title=title,
            teacher=teacher,
            url=url,
            overview_outline=outline,
            overview_goal=goal,
            overview_en=overview_en,
            teaching_plan_text=plan_text,
            raw=it
        ))
    return courses


# --------- Keyword fallback ---------
def parse_boolean_query(q: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Very light boolean support: tokens separated by spaces; OR/AND/-negation supported.
    Returns (must_terms, any_terms, not_terms)
    """
    tokens = re.findall(r'"[^"]+"|\S+', q)
    must_terms, any_terms, not_terms = [], [], []
    mode_any = False
    for tok in tokens:
        if tok.upper() == "OR":
            mode_any = True
            continue
        if tok.upper() == "AND":
            mode_any = False
            continue
        neg = tok.startswith("-")
        term = tok[1:] if neg else tok
        term = term.strip('"')
        if not term:
            continue
        if neg:
            not_terms.append(term)
        else:
            if mode_any:
                any_terms.append(term)
            else:
                must_terms.append(term)
    return must_terms, any_terms, not_terms


def keyword_filter(courses: List[Course], query: str) -> List[Course]:
    must_terms, any_terms, not_terms = parse_boolean_query(query)
    def ok(text: str) -> bool:
        t = text.lower()
        if any(nt.lower() in t for nt in not_terms):
            return False
        if must_terms and not all(mt.lower() in t for mt in must_terms):
            return False
        if any_terms and not any(at.lower() in t for at in any_terms):
            return False
        return True

    out = []
    for c in courses:
        text = c.to_text()
        if ok(text):
            out.append(c)
    return out


# --------- Embedding-based semantic search ---------
def cosine(a: List[float], b: List[float]) -> float:
    if np is None:
        # simple manual cosine
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(x*x for x in b))
        return (dot / (norm_a * norm_b)) if norm_a and norm_b else 0.0
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va.dot(vb) / denom) if denom else 0.0


def openai_embed(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    if OpenAI is None:
        raise RuntimeError("openai python package not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def ollama_embed(texts: List[str], model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> List[List[float]]:
    if requests is None:
        raise RuntimeError("requests package not installed")
    embs = []
    for t in texts:
        r = requests.post(f"{host}/api/embeddings", json={"model": model, "prompt": t})
        r.raise_for_status()
        embs.append(r.json()["embedding"])
    return embs


def embedding_search(courses: List[Course], query: str, provider: str, embed_model: str,
                     top_k: int = 50, threshold: float = 0.25) -> List[Tuple[Course, float]]:
    texts = [c.to_text() for c in courses]
    # Embed
    if provider == "openai":
        q_emb = openai_embed([query], model=embed_model)[0]
        c_embs = openai_embed(texts, model=embed_model)
    elif provider == "ollama":
        q_emb = ollama_embed([query], model=embed_model)[0]
        c_embs = ollama_embed(texts, model=embed_model)
    elif provider == "hf":
        q_emb = hf_embed([query], model_name=embed_model)[0]
        c_embs = hf_embed(texts, model_name=embed_model)
    else:
        raise ValueError("provider must be 'openai' or 'ollama' or 'hf'")
    # Score
    scored = []
    for c, e in zip(courses, c_embs):
        s = cosine(q_emb, e)
        if s >= threshold:
            scored.append((c, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# --------- LLM zero-shot classification ---------
# ---- HF providers (Transformers / Sentence-Transformers) ----
HF_CHAT_PIPELINE = None
HF_EMBED_MODEL = None

def hf_prepare_chat(model_name: str = "elyza/ELYZA-japanese-Llama-2-7b-instruct"):
    global HF_CHAT_PIPELINE
    if HF_CHAT_PIPELINE is not None:
        return HF_CHAT_PIPELINE
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
    except Exception as e:
        raise RuntimeError("transformers is not installed. pip install transformers accelerate sentencepiece") from e
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    HF_CHAT_PIPELINE = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=200,
        do_sample=False
    )
    return HF_CHAT_PIPELINE

def hf_chat(prompt: str, model: str = "elyza/ELYZA-japanese-Llama-2-7b-instruct") -> str:
    pipe = hf_prepare_chat(model)
    # Keep instruction in the prompt; model is instruction-tuned
    full_prompt = f"{LLM_INSTRUCTION}\n\n{prompt}\n"
    out = pipe(full_prompt, return_full_text=False)
    text = out[0]["generated_text"]
    return text

def hf_embed(texts: List[str], model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> List[List[float]]:
    global HF_EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is not installed. pip install sentence-transformers") from e
    if HF_EMBED_MODEL is None or getattr(HF_EMBED_MODEL, 'model_name', '') != model_name:
        HF_EMBED_MODEL = SentenceTransformer(model_name)
        HF_EMBED_MODEL.model_name = model_name
    emb = HF_EMBED_MODEL.encode(texts, normalize_embeddings=False, convert_to_numpy=False)
    return [e.tolist() for e in emb]

LLM_INSTRUCTION = """あなたは大学のシラバス検索アシスタントです。以下の「検索意図」に最も関連する授業を厳密に選定します。
- 出力はJSON（UTF-8）で、各要素は {"idx": <整数>, "relevance": <0.0~1.0>, "reason": "<50字以内>"} としてください。
- "relevance" は0.5以上なら「関連あり」と見なします。厳しめに判定してください。
- 過剰一致や一般語（例: コンピュータ、データ）だけでの一致はスコアを下げてください。
- 同義語や見落としに注意し、シラバスの要旨・到達目標・授業計画も考慮してください。
- 返答はJSON配列のみ。前置きや説明文は絶対に出力しないでください。
"""

def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]

def openai_chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    if OpenAI is None:
        raise RuntimeError("openai python package not installed")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"} if False else None,  # we will ask for raw JSON array
    )
    return resp.choices[0].message.content

def ollama_chat(prompt: str, model: str = "llama3.1:8b", host: str = "http://localhost:11434") -> str:
    if requests is None:
        raise RuntimeError("requests package not installed")
    r = requests.post(f"{host}/api/chat", json={
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "stream": False
    })
    r.raise_for_status()
    return r.json()["message"]["content"]

def llm_classify(courses: List[Course], query: str, provider: str, chat_model: str, batch_size: int = 24) -> List[Tuple[Course, float, str]]:
    results: List[Tuple[Course, float, str]] = []
    for batch in chunk_list(courses, batch_size):
        # Build prompt with compact items to reduce tokens
        lines = [f"検索意図: {query}", "候補一覧:"]
        for c in batch:
            # Keep concise text, cap each field
            t = c.to_text()
            t = (t[:500] + "…") if len(t) > 500 else t
            lines.append(json.dumps({"idx": c.idx, "text": t}, ensure_ascii=False))
        lines.append('上記の候補から関連性のある科目のみをJSON配列で返してください（例: [{"idx":0,"relevance":0.78,"reason":"..."}]）。')
        prompt = "\n".join(lines)

        if provider == "openai":
            raw = openai_chat(prompt, model=chat_model)
        elif provider == "ollama":
            raw = ollama_chat(prompt, model=chat_model)
        elif provider == "hf":
            raw = hf_chat(prompt, model=chat_model)
        else:
            raise ValueError("provider must be 'openai' or 'ollama' or 'hf'")
        # Some models may wrap in code fences; strip them
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            # remove possible language hint like json\n
            raw = re.sub(r"^json\s*", "", raw, flags=re.IGNORECASE)
        try:
            arr = json.loads(raw)
            if isinstance(arr, dict) and "items" in arr:
                arr = arr["items"]
        except Exception as e:
            # best-effort: try to find JSON array
            m = re.search(r"\[.*\]", raw, flags=re.S)
            if not m:
                print("WARN: failed to parse LLM output; skipping batch")
                continue
            arr = json.loads(m.group(0))
        for r in arr:
            try:
                idx = int(r["idx"])
                score = float(r.get("relevance", 0.0))
                reason = str(r.get("reason", ""))
                # find course by idx in this batch
                c = next((x for x in batch if x.idx == idx), None)
                if c:
                    results.append((c, score, reason))
            except Exception:
                continue
    # Deduplicate by idx, keep max score
    dedup: Dict[int, Tuple[Course, float, str]] = {}
    for c, s, reason in results:
        if c.idx not in dedup or s > dedup[c.idx][1]:
            dedup[c.idx] = (c, s, reason)
    out = list(dedup.values())
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def to_records_llm(scored: List[Tuple[Course, float, str]]) -> List[Dict[str, Any]]:
    recs = []
    for c, s, reason in scored:
        recs.append({
            "score": round(float(s), 4),
            "title": c.title,
            "teacher": c.teacher,
            "url": c.url,
            "reason": reason,
            "overview_outline": c.overview_outline,
            "overview_goal": c.overview_goal,
            "overview_en": c.overview_en,
            "teaching_plan": c.teaching_plan_text,
            "idx": c.idx
        })
    return recs


def to_records_embed(scored: List[Tuple[Course, float]]) -> List[Dict[str, Any]]:
    recs = []
    for c, s in scored:
        recs.append({
            "score": round(float(s), 4),
            "title": c.title,
            "teacher": c.teacher,
            "url": c.url,
            "overview_outline": c.overview_outline,
            "overview_goal": c.overview_goal,
            "overview_en": c.overview_en,
            "teaching_plan": c.teaching_plan_text,
            "idx": c.idx
        })
    return recs


def to_records_basic(items: List[Course]) -> List[Dict[str, Any]]:
    recs = []
    for c in items:
        recs.append({
            "title": c.title,
            "teacher": c.teacher,
            "url": c.url,
            "overview_outline": c.overview_outline,
            "overview_goal": c.overview_goal,
            "overview_en": c.overview_en,
            "teaching_plan": c.teaching_plan_text,
            "idx": c.idx
        })
    return recs



def save_outputs(records: List[Dict[str, Any]], out_prefix: str = "filtered_courses"):
    """
    out_prefix に拡張子なしのパス（例: 'filtered_courses' や 'out/security'）を渡すと
    out/security.json と out/security.csv を生成します。ディレクトリは自動作成。
    既存呼び出しの互換性を保つため、デフォルトはカレント直下の 'filtered_courses' に変更。
    """
    import os, json
    try:
        import pandas as pd  # 既存環境に合わせて動的import
    except Exception:
        pd = None

    # ディレクトリ自動作成（例: out/security → out を作る）
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    json_path = f"{out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON saved: {json_path}")

    if pd is not None:
        df = pd.DataFrame(records)
        csv_path = f"{out_prefix}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ CSV saved: {csv_path}")
    else:
        print("⚠ pandas が無いため CSV はスキップしました（pip install pandas で有効化）")






def provider_auto_detect(preferred: str = "hf") -> str:
    """Choose a working provider automatically: hf -> ollama -> openai -> keyword-only fallback marker."""
    order = [preferred, "ollama", "openai"]
    for prov in order:
        if prov == "hf":
            try:
                # Try lazy import to see if transformers exists
                import transformers  # noqa: F401
                return "hf"
            except Exception:
                pass
        elif prov == "ollama":
            try:
                import requests
                r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
                if r.status_code == 200:
                    return "ollama"
            except Exception:
                pass
        elif prov == "openai":
            import os
            if os.environ.get("OPENAI_API_KEY"):
                return "openai"
    return "keyword"  # final fallback


def interactive_loop(args, courses):
    """
    Simple interactive loop so non-technical users can run multiple searches.
    - Press Enter on "検索キーワード" to exit.
    - Keeps last choices; just hit Enter to reuse.
    """
    last_query = args.query or ""
    last_method = args.method
    last_provider = args.provider
    last_embed_model = args.embed_model
    last_chat_model = args.chat_model
    last_topk = args.top_k
    last_threshold = args.threshold
    last_batch = min(args.batch_size, 12)

    print("\n====== 反復検索モード / Interactive Search ======")
    print("空Enterで終了。各設定は直前の値が初期値として表示されます。\n")

    
    while True:
        try:
            q = input(f"🔎 検索キーワード [{last_query}]: ").strip()
            if not q:
                if not last_query:
                    print("👋 終了します。")
                    break
                q = last_query

            # Default simple mode: no complex prompts
            if not getattr(args, "advanced", False):
                # auto-detect working provider
                prov = provider_auto_detect(preferred="hf")
                if prov == "keyword":
                    method = "keyword"
                    provider = "keyword"
                else:
                    provider = prov
                    method = args.method  # default llm
                embed_model = last_embed_model
                chat_model = last_chat_model
                top_k = last_topk
                threshold = last_threshold
                batch_size = last_batch
            else:
                # Advanced prompts for power users
                method = input(f"⚙️ 方式 embed/llm/keyword [{last_method}]: ").strip().lower() or last_method
                provider = last_provider
                embed_model = last_embed_model
                chat_model = last_chat_model
                top_k = last_topk
                threshold = last_threshold
                batch_size = last_batch

                if method in ("embed", "llm"):
                    provider = input(f"🤖 プロバイダ openai/ollama/hf [{last_provider}]: ").strip().lower() or last_provider

                if method == "embed":
                    if provider == "openai":
                        default_embed = "text-embedding-3-small"
                    elif provider == "ollama":
                        default_embed = "nomic-embed-text"
                    else:
                        default_embed = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    embed_model = input(f"🧩 埋め込みモデル [{last_embed_model or default_embed}]: ").strip() or (last_embed_model or default_embed)
                    top_k_in = input(f"📈 上位件数 top-k [{last_topk}]: ").strip()
                    if top_k_in:
                        try:
                            top_k = int(top_k_in)
                        except:
                            print("  ↪ 無効な数値です。前回値を使います。")
                    thr_in = input(f"📏 しきい値 threshold (0~1) [{last_threshold}]: ").strip()
                    if thr_in:
                        try:
                            threshold = float(thr_in)
                        except:
                            print("  ↪ 無効な数値です。前回値を使います。")

                if method == "llm":
                    if provider == "openai":
                        default_chat = "gpt-4o-mini"
                    elif provider == "ollama":
                        default_chat = "llama3.1:8b"
                    else:
                        default_chat = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
                    chat_model = input(f"🗣 生成モデル [{last_chat_model or default_chat}]: ").strip() or (last_chat_model or default_chat)
                    bs_in = input(f"🧺 バッチサイズ [{last_batch}]: ").strip()
                    if bs_in:
                        try:
                            batch_size = int(bs_in)
                        except:
                            print("  ↪ 無効な数値です。前回値を使います。")

            # Execute
            if method == "keyword":
                items = keyword_filter(courses, q)
                print(f"🔎 Keyword matches: {len(items)}")
                recs = to_records_basic(items)
                save_outputs(recs)
            elif method == "embed":
                scored = embedding_search(
                    courses, q, provider=provider,
                    embed_model=embed_model, top_k=top_k, threshold=threshold
                )
                print(f"🔎 Semantic matches (>= {threshold}): {len(scored)}")
                recs = to_records_embed(scored)
                save_outputs(recs)
            elif method == "llm":
                scored = llm_classify(
                    courses, q, provider=provider,
                    chat_model=chat_model, batch_size=batch_size
                )
                print(f"🔎 LLM-classified relevant courses: {len(scored)}")
                recs = to_records_llm(scored)
                save_outputs(recs)
            else:
                print("❌ 未対応の方式です（embed/llm/keyword のいずれか）。")

            last_query = q
            last_method = method
            last_provider = provider
            last_embed_model = embed_model
            last_chat_model = chat_model
            last_topk = top_k
            last_threshold = threshold
            last_batch = batch_size

            print("✅ 完了。結果ファイル: /mnt/data/filtered_courses.json, /mnt/data/filtered_courses.csv")
            print("（高度な設定を出したい場合は --advanced を付けて起動）")
            print("--------------------------------------------------")
        except KeyboardInterrupt:
            print("👋 中断されました。終了します。")
            break
        except Exception as e:
            print(f"⚠ エラー: {e}")
            print("設定を見直してもう一度お試しください。")
            print("--------------------------------------------------")
            continue
        except KeyboardInterrupt:
            print("\n👋 中断されました。終了します。")
            break
        except Exception as e:
            print(f"⚠ エラー: {e}")
            print("設定を見直してもう一度お試しください。")
            print("--------------------------------------------------\n")
            continue


def main():
    ap = argparse.ArgumentParser(description="Filter syllabus by topic using LLM")
    ap.add_argument("--json", required=True, help="Path to syllabus JSON (e.g., syllabus_2025_full.json)")
    ap.add_argument("--query", help="Topic / natural language request")
    ap.add_argument("--method", choices=["embed", "llm", "keyword"], default="llm",
                    help="embed: semantic search (default), llm: zero-shot classification, keyword: simple boolean filter")
    ap.add_argument("--provider", choices=["openai", "ollama", "hf"], default="hf",
                    help="LLM provider for embed/llm methods")
    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model (HF/OpenAI/Ollama)")
    ap.add_argument("--chat-model", default="elyza/ELYZA-japanese-Llama-2-7b-instruct", help="Chat model (ELYZA for HF, or OpenAI/Ollama names)")
    ap.add_argument("--top-k", type=int, default=50, help="Top K (embed)")
    ap.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold (embed)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for llm method")
    ap.add_argument("--interactive", action="store_true", help="Enter interactive repeated-search mode")
    ap.add_argument("--advanced", action="store_true", help="Show advanced prompts (provider/models) in interactive mode")
    args = ap.parse_args()

    courses = load_courses(args.json)
    if args.interactive:
        interactive_loop(args, courses)
        return

    # If no query provided, enter interactive loop directly
    if not args.query:
        interactive_loop(args, courses)
        return

    if args.method == "keyword":
        items = keyword_filter(courses, args.query)
        print(f"🔎 Keyword matches: {len(items)}")
        recs = to_records_basic(items)
        save_outputs(recs)
        return

    if args.method == "embed":
        scored = embedding_search(
            courses, args.query, provider=args.provider,
            embed_model=args.embed_model, top_k=args.top_k, threshold=args.threshold
        )
        print(f"🔎 Semantic matches (>= {args.threshold}): {len(scored)}")
        recs = to_records_embed(scored)
        save_outputs(recs)
        return

    if args.method == "llm":
        scored = llm_classify(
            courses, args.query, provider=args.provider,
            chat_model=args.chat_model, batch_size=args.batch_size
        )
        print(f"🔎 LLM-classified relevant courses: {len(scored)}")
        recs = to_records_llm(scored)
        save_outputs(recs)
        return

    raise ValueError("Unknown method")


if __name__ == "__main__":
    main()
