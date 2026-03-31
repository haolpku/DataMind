#!/usr/bin/env python3
"""
Phase 2 多模态 RAG 验证脚本

测试三种 image_embedding_mode 下 load_pre_chunked() 的分流逻辑：
  1. disabled  — 忽略 image_path，仅产出纯文本 TextNode
  2. vlm_describe — 图片描述拼入 text，全部产出 TextNode
  3. clip — 文本 → TextNode，图片 → ImageNode

使用 mm_demo profile 的示例数据，不调用任何外部 API。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["DATA_PROFILE"] = "mm_demo"
os.environ["IMAGE_EMBEDDING_MODE"] = "disabled"

from config import Settings


def test_disabled_mode():
    """disabled 模式: 只产出有 text 的 TextNode，忽略 image_path"""
    cfg = Settings(
        _env_file=None,
        data_profile="mm_demo",
        image_embedding_mode="disabled",
    )

    import importlib
    import config as config_mod
    config_mod.settings = cfg

    from modules.rag.indexer import load_pre_chunked
    result = load_pre_chunked()

    text_nodes = result["text_nodes"]
    image_nodes = result["image_nodes"]

    print("=" * 60)
    print("TEST: disabled mode")
    print(f"  text_nodes:  {len(text_nodes)}")
    print(f"  image_nodes: {len(image_nodes)}")

    assert len(image_nodes) == 0, "disabled 模式不应产出 ImageNode"

    for n in text_nodes:
        assert n.text.strip(), f"TextNode 的 text 不应为空: {n.metadata}"

    # 原始数据 5 行: 3 行有 text, 1 行纯 image (text=""), 1 行 text_image
    # disabled 模式: text="" 的纯 image chunk 被跳过, 其余 4 行有 text
    assert len(text_nodes) == 4, f"disabled 模式应产出 4 个 TextNode，实际 {len(text_nodes)}"

    print("  PASSED ✓")
    return True


def test_vlm_describe_mode():
    """vlm_describe 模式: 有 image_description 的拼入 text，没有的跳过 VLM 调用（图片存在但我们不真调 API）"""
    cfg = Settings(
        _env_file=None,
        data_profile="mm_demo",
        image_embedding_mode="vlm_describe",
    )

    import config as config_mod
    config_mod.settings = cfg

    import importlib
    import modules.rag.indexer as indexer_mod
    importlib.reload(indexer_mod)

    # mock VLM 调用，避免真实 API
    original_fn = indexer_mod._describe_image_with_vlm
    indexer_mod._describe_image_with_vlm = lambda path, model=None: f"[MOCK VLM] 描述 {os.path.basename(path)}"

    try:
        result = indexer_mod.load_pre_chunked()
    finally:
        indexer_mod._describe_image_with_vlm = original_fn

    text_nodes = result["text_nodes"]
    image_nodes = result["image_nodes"]

    print("=" * 60)
    print("TEST: vlm_describe mode")
    print(f"  text_nodes:  {len(text_nodes)}")
    print(f"  image_nodes: {len(image_nodes)}")

    assert len(image_nodes) == 0, "vlm_describe 模式不应产出 ImageNode"
    assert len(text_nodes) == 5, f"vlm_describe 模式应产出 5 个 TextNode，实际 {len(text_nodes)}"

    for n in text_nodes:
        img_path = n.metadata.get("image_path")
        if img_path:
            assert "[图片描述]" in n.text, f"带 image_path 的 chunk 应包含 [图片描述]: {n.text[:80]}"
            desc = n.metadata.get("image_description", "")
            assert desc, f"vlm_describe 模式应在 metadata 中记录 image_description"

    print("  PASSED ✓")
    return True


def test_clip_mode():
    """clip 模式: 纯文本 → TextNode，带 image_path → ImageNode"""
    cfg = Settings(
        _env_file=None,
        data_profile="mm_demo",
        image_embedding_mode="clip",
    )

    import config as config_mod
    config_mod.settings = cfg

    import importlib
    import modules.rag.indexer as indexer_mod
    importlib.reload(indexer_mod)

    result = indexer_mod.load_pre_chunked()

    text_nodes = result["text_nodes"]
    image_nodes = result["image_nodes"]

    print("=" * 60)
    print("TEST: clip mode")
    print(f"  text_nodes:  {len(text_nodes)}")
    print(f"  image_nodes: {len(image_nodes)}")

    # 5 行数据:
    #   1. text-only → 1 TextNode
    #   2. text_image (有 text + image) → 1 TextNode + 1 ImageNode
    #   3. image-only (text="") → 0 TextNode + 1 ImageNode
    #   4. text-only → 1 TextNode
    #   5. text_image (有 text + image) → 1 TextNode + 1 ImageNode
    # 总计: 4 TextNode, 3 ImageNode
    assert len(text_nodes) == 4, f"clip 模式应产出 4 个 TextNode，实际 {len(text_nodes)}"
    assert len(image_nodes) == 3, f"clip 模式应产出 3 个 ImageNode，实际 {len(image_nodes)}"

    for img_n in image_nodes:
        assert os.path.isfile(img_n.image_path), f"ImageNode 的 image_path 应为有效文件: {img_n.image_path}"

    print("  PASSED ✓")
    return True


def test_config_fields():
    """验证配置字段正确读取"""
    cfg = Settings(
        _env_file=None,
        image_embedding_mode="clip",
        clip_model="openai/clip-vit-large-patch14",
        vlm_model="gpt-4o",
        use_multimodal_llm=True,
        image_similarity_top_k=5,
    )

    print("=" * 60)
    print("TEST: config fields")
    assert cfg.image_embedding_mode == "clip"
    assert cfg.clip_model == "openai/clip-vit-large-patch14"
    assert cfg.vlm_model == "gpt-4o"
    assert cfg.use_multimodal_llm is True
    assert cfg.image_similarity_top_k == 5
    print(f"  image_embedding_mode = {cfg.image_embedding_mode}")
    print(f"  clip_model           = {cfg.clip_model}")
    print(f"  vlm_model            = {cfg.vlm_model}")
    print(f"  use_multimodal_llm   = {cfg.use_multimodal_llm}")
    print(f"  image_similarity_top_k = {cfg.image_similarity_top_k}")
    print("  PASSED ✓")
    return True


def test_schema():
    """验证 RAGChunk schema 正确解析多模态字段"""
    from schema.types import RAGChunk, Modality

    print("=" * 60)
    print("TEST: RAGChunk schema")

    chunk1 = RAGChunk.model_validate({
        "text": "hello", "modality": "text"
    })
    assert chunk1.image_path is None
    assert chunk1.image_description is None
    assert chunk1.modality == Modality.TEXT

    chunk2 = RAGChunk.model_validate({
        "text": "图文混合",
        "image_path": "images/test.png",
        "image_description": "一张测试图片",
        "modality": "text_image",
    })
    assert chunk2.image_path == "images/test.png"
    assert chunk2.image_description == "一张测试图片"
    assert chunk2.modality == Modality.TEXT_IMAGE

    chunk3 = RAGChunk.model_validate({
        "text": "",
        "image_path": "images/only.png",
        "modality": "image",
        "extra_field": "should_be_ignored",
    })
    assert chunk3.modality == Modality.IMAGE
    assert chunk3.text == ""

    print("  all schema validations passed")
    print("  PASSED ✓")
    return True


if __name__ == "__main__":
    results = []
    results.append(("config_fields", test_config_fields()))
    results.append(("schema", test_schema()))
    results.append(("disabled", test_disabled_mode()))
    results.append(("vlm_describe", test_vlm_describe_mode()))
    results.append(("clip", test_clip_mode()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    all_pass = True
    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:20s} {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n所有测试通过！Phase 2 多模态分流逻辑验证成功。")
    else:
        print("\n存在失败的测试，请检查。")
        sys.exit(1)
