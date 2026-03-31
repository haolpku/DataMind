"""
DataMind 数据契约 — 定义预处理仓库与 DataMind 之间的标准数据格式。

RAGChunk 的多模态字段 (image_path, image_description, modality) 已被
DataMind 原生支持，通过 IMAGE_EMBEDDING_MODE 配置项启用。

GraphTriple 的 subject_properties / object_properties 为多模态预留，
当前仅存储为 EntityNode.properties，后续版本会进一步消费。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TEXT_IMAGE = "text_image"


class RAGChunk(BaseModel):
    """RAG 向量检索的最小单元。

    必填字段:
        text — chunk 文本内容（纯文本模式）。当 modality 为 image 时可为空。

    多模态字段:
        image_path        — 图片文件相对路径，供 CLIP embedding / VLM 使用。
        image_description — VLM 生成的图片文字描述（vlm_describe 模式使用，
                            预处理仓库可以预先填好，也可由 DataMind 入库时生成）。
        modality          — 标注该 chunk 的模态类型。
    """

    text: str = ""
    metadata: dict = Field(default_factory=dict)

    image_path: str | None = None
    image_description: str | None = None
    modality: Modality = Modality.TEXT

    model_config = {"extra": "ignore"}


class GraphTriple(BaseModel):
    """GraphRAG 知识图谱三元组。

    必填字段:
        subject, relation, object — 实体-关系-实体。

    可选字段:
        subject_type / object_type — 实体类型标签。

    多模态预留:
        subject_properties / object_properties — 实体级附加属性，
        可携带 image、description 等字段供多模态 GraphRAG 使用。
    """

    subject: str
    relation: str
    object: str

    subject_type: str = "entity"
    object_type: str = "entity"

    # 多模态预留: {"image": "img/entityA.png", "description": "..."}
    subject_properties: dict = Field(default_factory=dict)
    object_properties: dict = Field(default_factory=dict)

    confidence: float = 1.0
    source: str = ""

    model_config = {"extra": "ignore"}


class DatabaseImport(BaseModel):
    """Database 文件化导入的元信息。

    描述一个 .sql 文件的导入方式。当前 DataMind 直接扫描
    profile 目录下的 tables/*.sql 文件执行，此 schema 用于
    未来支持更复杂的导入（如 CSV + DDL 组合）。
    """

    file_path: str
    format: Literal["sql", "csv"] = "sql"
    table_name: str | None = None

    model_config = {"extra": "ignore"}
