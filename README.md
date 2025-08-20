# syllabus-checker
syllabus_batch_checkはjsonファイルから授業内容とタイトルの整合性を確認します。
syllabus_check_resultはsyllabus_batch_checkの出力内容です。

# 📘 シラバス検索・関連科目抽出システム
filter_syllabus_llmは実行後入力した検索キーワードに関連する授業を一覧出力します。

## 🔎 本プロジェクトについて
本プロジェクトは、**法政大学応用情報工学科のシラバス検索・関連科目抽出** を目的としたシステムです。  
入力されたキーワードに基づき、授業内容を検索・抽出します。

---

## LLM vs Embedding

本システムは当初、**LLM（大規模言語モデル）による検索** と **Embedding（ベクトル類似度検索）** の両方式を検討しました。

その結果、以下の理由から **Embedding方式** を採用しています。

### ✅ Embedding方式
- シンプルかつ高速に動作  
- 再現性が高く、同じ検索条件なら常に同じ結果  
- 「検索精度」「類似度閾値」「モデル比較」といった研究的な検証が可能  
- シラバス検索のような **情報検索タスクに最適**

### ❌ LLM方式
- 曖昧な質問に答える能力はあるが、再現性が低い  
- 出力フォーマットが揺らぎがちで安定性に欠ける  
- 計算コスト・GPU負荷がEmbeddingより高い  
- 今回の科目検索用途には **オーバースペック**

---

## 📌 結論
- 実装は **Embedding検索を標準採用**  
- LLMは必要に応じて拡張機能として検討可能（例: 要約、シラバスQ&A など）


### ⚙️ 実行方法
コマンドライン引数 `--method` により切り替え可能です。
```bash
# Embedding検索（標準）
python filter_syllabus_llm.py --json syllabus_2025_full.json --method embed

# LLM検索（実験的）
python filter_syllabus_llm.py --json syllabus_2025_full.json --method llm
