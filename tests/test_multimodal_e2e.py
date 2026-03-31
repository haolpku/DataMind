#!/usr/bin/env python3
"""
Phase 2 多模态 RAG 端到端测试

使用 vlm_describe 模式，真实调用大模型 API:
  1. 读取 mm_demo profile 的 JSONL + 图片
  2. VLM 生成图片描述 (调用 GPT-4o)
  3. 文本 embedding + 构建 Chroma 索引
  4. 用自然语言查询，验证能否检索到图片中的信息
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["DATA_PROFILE"] = "mm_demo"
os.environ["IMAGE_EMBEDDING_MODE"] = "vlm_describe"

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "mm_demo")


def clean_index():
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
        print(f"[CLEAN] 已删除旧索引: {STORAGE_DIR}")


def main():
    clean_index()

    from config import Settings
    cfg = Settings(
        _env_file=os.path.join(os.path.dirname(__file__), "..", ".env"),
        data_profile="mm_demo",
        image_embedding_mode="vlm_describe",
    )

    import config as config_mod
    config_mod.settings = cfg

    from llama_index.core import Settings as LlamaSettings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    llm = OpenAI(
        api_base=cfg.llm_api_base,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
    )
    embed_model = OpenAIEmbedding(
        api_base=cfg.embedding_api_base,
        api_key=cfg.embedding_api_key,
        model_name=cfg.embedding_model,
    )
    LlamaSettings.llm = llm
    LlamaSettings.embed_model = embed_model

    print(f"\n{'='*60}")
    print(f"LLM:       {cfg.llm_model} @ {cfg.llm_api_base}")
    print(f"Embedding: {cfg.embedding_model}")
    print(f"Mode:      {cfg.image_embedding_mode}")
    print(f"Profile:   {cfg.data_profile}")
    print(f"Data dir:  {cfg.data_dir}")
    print(f"{'='*60}\n")

    # Step 1: 加载数据 (VLM 会为无描述的图片调用 API)
    print("[STEP 1] 加载预分块数据 + VLM 图片描述...")
    from modules.rag.indexer import load_pre_chunked, build_index
    result = load_pre_chunked()
    text_nodes = result["text_nodes"]

    print(f"\n共产出 {len(text_nodes)} 个 TextNode:")
    for i, n in enumerate(text_nodes):
        preview = n.text[:120].replace("\n", " ")
        has_img = "image_path" in n.metadata
        print(f"  [{i}] {'📷 ' if has_img else '📝 '}{preview}...")

    # Step 2: 构建索引
    print(f"\n[STEP 2] 构建 Chroma 向量索引...")
    index = build_index(text_nodes=text_nodes)
    print(f"索引构建完成\n")

    # Step 3: 查询测试
    print("[STEP 3] 查询测试 — 验证大模型能否检索到图片中的信息\n")

    query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")

    questions = [
        "系统架构有哪几层？每层包含什么组件？",
        "哪种 RAG 检索策略的召回率最高？具体是多少？",
        "知识图谱中 LlamaIndex 和哪些技术有关系？",
    ]

    expected_keywords = [
        ["Data Layer", "Service Layer", "API Gateway", "PostgreSQL", "Redis"],
        ["Hybrid", "91"],
        ["Python", "RAG", "Agent"],
    ]

    all_pass = True
    for q, keywords in zip(questions, expected_keywords):
        print(f"Q: {q}")
        response = query_engine.query(q)
        answer = str(response).strip()
        print(f"A: {answer}\n")

        found = [kw for kw in keywords if kw.lower() in answer.lower()]
        missing = [kw for kw in keywords if kw.lower() not in answer.lower()]

        if missing:
            print(f"  ⚠ 未找到关键词: {missing}")
            all_pass = False
        else:
            print(f"  ✓ 关键词全部命中: {found}")
        print()

    # Step 4: 检查 VLM 生成的描述
    print("[STEP 4] VLM 生成的图片描述:\n")
    for n in text_nodes:
        desc = n.metadata.get("image_description", "")
        if desc:
            img = n.metadata.get("image_path", "?")
            print(f"  📷 {img}:")
            print(f"     {desc[:200]}")
            print()

    print("=" * 60)
    if all_pass:
        print("端到端测试通过！大模型成功从图片中提取信息并回答问题。")
    else:
        print("部分关键词未命中，但流程跑通。请查看上方回答判断质量。")

    clean_index()


if __name__ == "__main__":
    main()
