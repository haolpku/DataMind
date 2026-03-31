"""
文档索引模块: 加载文档 -> 分块 -> 构建 Chroma 向量索引

支持两种输入:
  方式 A: 原始文档 (profile 目录) -> SentenceSplitter 自动分块 -> Embedding -> 存 Chroma
  方式 B: 预分块 JSONL (profile/chunks/*.jsonl) -> 跳过分块 -> Embedding -> 存 Chroma

多模态支持 (image_embedding_mode):
  disabled     — 忽略 image_path，纯文本行为
  vlm_describe — 图片通过 VLM 转为文本描述，拼入 text 后走文本 embedding
  clip         — 图片用 CLIP embedding，构建 MultiModalVectorStoreIndex
"""

from __future__ import annotations

import json
import os
import glob
from typing import TYPE_CHECKING

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, ImageNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import settings
from schema.types import RAGChunk

if TYPE_CHECKING:
    from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex as MMIndexType


RESERVED_SUBDIRS = {"chunks", "triplets", "tables", "images"}

IndexType = VectorStoreIndex  # runtime union, see build_index return


def load_documents(data_dir: str = None):
    """从 profile 目录加载所有文档 (支持 PDF, TXT, MD, DOCX 等)
    自动排除 chunks/, triplets/, tables/, images/ 等保留子目录"""
    if data_dir is None:
        data_dir = settings.data_dir
    if not os.path.exists(data_dir):
        print(f"[WARNING] data 目录不存在: {data_dir}")
        return []

    files = []
    for item in os.listdir(data_dir):
        if item in RESERVED_SUBDIRS or item.startswith("."):
            continue
        full_path = os.path.join(data_dir, item)
        if os.path.isfile(full_path):
            files.append(full_path)
        elif os.path.isdir(full_path):
            for root, dirs, filenames in os.walk(full_path):
                for f in filenames:
                    if not f.startswith("."):
                        files.append(os.path.join(root, f))

    if not files:
        print(f"[WARNING] data 目录为空: {data_dir}")
        return []

    reader = SimpleDirectoryReader(input_files=files)
    documents = reader.load_data()
    print(f"[INFO] 加载了 {len(documents)} 个文档")
    return documents


# ---------------------------------------------------------------------------
# VLM 描述生成 (vlm_describe 模式，chunk 无预填 image_description 时调用)
# ---------------------------------------------------------------------------

def _describe_image_with_vlm(image_path: str, vlm_model: str | None = None) -> str:
    """调用 VLM API 为图片生成文字描述（使用 OpenAI SDK，兼容性更好）。"""
    import base64
    from openai import OpenAI as _OpenAI

    model = vlm_model or settings.vlm_model or settings.llm_model
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")

    client = _OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_api_base)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "请用中文详细描述这张图片的内容，包括所有可见的文字、数据和结构。"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# load_pre_chunked: 根据 image_embedding_mode 分流
# ---------------------------------------------------------------------------

def _resolve_image_path(raw_path: str) -> str:
    """将 chunk 中的 image_path 解析为绝对路径（相对于 profile 目录）。"""
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(settings.data_dir, raw_path)


def load_pre_chunked(
    chunks_dir: str = None,
    image_embed_model=None,
) -> dict:
    """从 profile/chunks/ 目录加载预分块的 JSONL 文件。

    Returns:
        dict with keys:
            text_nodes  — list[TextNode]
            image_nodes — list[ImageNode]  (仅 clip 模式)
    """
    if chunks_dir is None:
        chunks_dir = os.path.join(settings.data_dir, "chunks")
    if not os.path.exists(chunks_dir):
        return {"text_nodes": [], "image_nodes": []}

    jsonl_files = glob.glob(os.path.join(chunks_dir, "*.jsonl"))
    if not jsonl_files:
        return {"text_nodes": [], "image_nodes": []}

    mode = settings.image_embedding_mode
    text_nodes: list[TextNode] = []
    image_nodes: list[ImageNode] = []

    for filepath in jsonl_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARNING] {filepath} 第 {line_num} 行 JSON 解析失败，跳过")
                    continue

                try:
                    chunk = RAGChunk.model_validate(raw)
                except Exception as e:
                    print(f"[WARNING] {filepath} 第 {line_num} 行 schema 校验失败: {e}，跳过")
                    continue

                metadata = chunk.metadata.copy()
                metadata["source_file"] = os.path.basename(filepath)
                if chunk.image_path:
                    metadata["image_path"] = chunk.image_path
                if chunk.modality.value != "text":
                    metadata["modality"] = chunk.modality.value

                has_image = bool(chunk.image_path)

                if mode == "disabled" or not has_image:
                    if not chunk.text:
                        continue
                    text_nodes.append(TextNode(text=chunk.text, metadata=metadata))

                elif mode == "vlm_describe":
                    desc = chunk.image_description
                    if not desc:
                        abs_path = _resolve_image_path(chunk.image_path)
                        if os.path.isfile(abs_path):
                            print(f"[INFO] VLM 描述: {chunk.image_path}")
                            desc = _describe_image_with_vlm(abs_path)
                        else:
                            print(f"[WARNING] 图片不存在: {abs_path}，跳过 VLM 描述")
                    combined = chunk.text
                    if desc:
                        combined = f"{chunk.text}\n[图片描述] {desc}" if chunk.text else f"[图片描述] {desc}"
                    if not combined.strip():
                        continue
                    metadata["image_description"] = desc or ""
                    text_nodes.append(TextNode(text=combined, metadata=metadata))

                elif mode == "clip":
                    if chunk.text:
                        text_nodes.append(TextNode(text=chunk.text, metadata=metadata))
                    abs_path = _resolve_image_path(chunk.image_path)
                    if os.path.isfile(abs_path):
                        img_node = ImageNode(
                            image_path=abs_path,
                            text=chunk.text or "",
                            metadata=metadata,
                        )
                        image_nodes.append(img_node)
                    else:
                        print(f"[WARNING] 图片不存在: {abs_path}，跳过 ImageNode 创建")

    total = len(text_nodes) + len(image_nodes)
    if total:
        msg = f"[INFO] 从预分块文件加载了 {len(text_nodes)} 个文本 chunks"
        if image_nodes:
            msg += f" + {len(image_nodes)} 个图片 nodes"
        msg += f" (来自 {len(jsonl_files)} 个文件)"
        print(msg)
    return {"text_nodes": text_nodes, "image_nodes": image_nodes}


# ---------------------------------------------------------------------------
# build_index: 根据 image_embedding_mode 选择索引类型
# ---------------------------------------------------------------------------

def build_index(
    documents=None,
    nodes=None,
    persist_dir: str = None,
    text_nodes: list[TextNode] | None = None,
    image_nodes: list[ImageNode] | None = None,
    image_embed_model=None,
):
    """构建或加载向量索引。

    CLIP 模式下构建 MultiModalVectorStoreIndex；其余模式使用 VectorStoreIndex。
    """
    if persist_dir is None:
        persist_dir = settings.storage_dir

    mode = settings.image_embedding_mode
    use_clip = mode == "clip" and image_nodes

    if use_clip:
        return _build_multimodal_index(
            text_nodes=text_nodes or [],
            image_nodes=image_nodes,
            persist_dir=persist_dir,
            image_embed_model=image_embed_model,
        )

    all_nodes = nodes or text_nodes
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("rag_allinone")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if all_nodes:
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        print(f"[INFO] 从预分块数据构建索引完成 ({len(all_nodes)} chunks)，已持久化到: {persist_dir}")
    elif documents:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True,
        )
        print(f"[INFO] 从文档构建索引完成，已持久化到: {persist_dir}")
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)
        print(f"[INFO] 从已有索引加载: {persist_dir}")

    return index


def _build_multimodal_index(
    text_nodes: list[TextNode],
    image_nodes: list[ImageNode],
    persist_dir: str,
    image_embed_model=None,
):
    """CLIP 模式: 构建 MultiModalVectorStoreIndex（双 collection）。"""
    from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    text_collection = chroma_client.get_or_create_collection("rag_text")
    image_collection = chroma_client.get_or_create_collection("rag_image")

    text_store = ChromaVectorStore(chroma_collection=text_collection)
    image_store = ChromaVectorStore(chroma_collection=image_collection)

    storage_context = StorageContext.from_defaults(
        vector_store=text_store,
        image_store=image_store,
    )

    all_nodes = list(text_nodes) + list(image_nodes)

    kwargs = dict(
        nodes=all_nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    if image_embed_model is not None:
        kwargs["image_embed_model"] = image_embed_model

    index = MultiModalVectorStoreIndex(**kwargs)
    print(
        f"[INFO] 多模态索引构建完成: {len(text_nodes)} text + {len(image_nodes)} image, "
        f"已持久化到: {persist_dir}"
    )
    return index


# ---------------------------------------------------------------------------
# get_or_create_index: 入口，兼容新旧两种索引
# ---------------------------------------------------------------------------

def get_or_create_index(image_embed_model=None):
    """如果索引已存在则加载，否则自动检测数据源并构建。

    优先级:
      1. 已有索引 -> 直接加载
      2. profiles/{profile}/chunks/*.jsonl 预分块数据 -> 分流处理
      3. data/ 目录下的原始文档 -> SentenceSplitter 分块 + Embedding
    """
    mode = settings.image_embedding_mode
    persist_dir = settings.storage_dir

    if mode == "clip":
        return _get_or_create_clip_index(persist_dir, image_embed_model)

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection("rag_allinone")

    if collection.count() > 0:
        print(f"[INFO] 检测到已有索引 ({collection.count()} 条向量)，直接加载")
        return build_index()

    result = load_pre_chunked()
    if result["text_nodes"]:
        print("[INFO] 检测到预分块数据，使用方式 B 构建索引 (跳过分块)")
        return build_index(text_nodes=result["text_nodes"])

    print("[INFO] 未检测到预分块数据，使用方式 A 从文档构建索引...")
    docs = load_documents()
    if not docs:
        return None
    return build_index(documents=docs)


def _get_or_create_clip_index(persist_dir: str, image_embed_model=None):
    """CLIP 模式: 检查双 collection 是否已有数据。"""
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    text_col = chroma_client.get_or_create_collection("rag_text")
    img_col = chroma_client.get_or_create_collection("rag_image")

    if text_col.count() > 0 or img_col.count() > 0:
        print(f"[INFO] 检测到已有多模态索引 (text={text_col.count()}, image={img_col.count()})，直接加载")
        return _load_existing_clip_index(persist_dir, image_embed_model)

    result = load_pre_chunked(image_embed_model=image_embed_model)
    if result["text_nodes"] or result["image_nodes"]:
        print("[INFO] 检测到预分块数据，构建多模态索引...")
        return build_index(
            text_nodes=result["text_nodes"],
            image_nodes=result["image_nodes"],
            image_embed_model=image_embed_model,
        )

    print("[INFO] 未检测到预分块数据，使用方式 A 从文档构建索引 (无图片)...")
    docs = load_documents()
    if not docs:
        return None
    return build_index(documents=docs)


def _load_existing_clip_index(persist_dir: str, image_embed_model=None):
    """从已有 Chroma collection 加载 MultiModalVectorStoreIndex。"""
    from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    text_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("rag_text"))
    image_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("rag_image"))

    storage_context = StorageContext.from_defaults(
        vector_store=text_store,
        image_store=image_store,
    )

    kwargs = dict(nodes=[], storage_context=storage_context)
    if image_embed_model is not None:
        kwargs["image_embed_model"] = image_embed_model

    return MultiModalVectorStoreIndex(**kwargs)
