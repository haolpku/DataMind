"""
统一配置: LLM + Embedding + 路径 + Retriever + Memory + Data Profile

所有配置项优先从环境变量读取，fallback 到默认值。
支持 .env 文件自动加载。

环境变量名与字段名一致（大写），例如:
  LLM_API_BASE=https://api.deepseek.com/v1
  LLM_MODEL=deepseek-chat
  RETRIEVER_MODE=multi_query
  DATA_PROFILE=2wiki_chunk512

常见 API 提供商示例:
  DeepSeek:  llm_api_base="https://api.deepseek.com/v1",  llm_model="deepseek-chat"
  智谱:      llm_api_base="https://open.bigmodel.cn/api/paas/v4", llm_model="glm-4-flash"
  月之暗面:   llm_api_base="https://api.moonshot.cn/v1",   llm_model="moonshot-v1-8k"
  硅基流动:   llm_api_base="https://api.siliconflow.cn/v1", llm_model="deepseek-ai/DeepSeek-V3"
  OpenAI:    llm_api_base="https://api.openai.com/v1",    llm_model="gpt-4o-mini"
"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- LLM ----
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-your-api-key-here"
    llm_model: str = "gpt-4o-mini"

    # ---- Embedding ----
    use_local_embedding: bool = False
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_api_key: str = "sk-your-api-key-here"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ---- Retriever ----
    retriever_mode: str = "simple"  # "simple" | "multi_query"
    multi_query_count: int = 3
    similarity_top_k: int = 3

    # ---- Memory ----
    memory_token_limit: int = 30000
    chat_history_token_ratio: float = 0.7

    # ---- Data Profile ----
    data_profile: str = "default"

    # ---- Paths ----
    @property
    def base_dir(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def data_dir(self) -> str:
        """Profile 级数据目录: data/profiles/{data_profile}/"""
        return os.path.join(self.base_dir, "data", "profiles", self.data_profile)

    @property
    def bench_dir(self) -> str:
        """问题集目录（跨 profile 共享）: data/bench/"""
        return os.path.join(self.base_dir, "data", "bench")

    @property
    def storage_dir(self) -> str:
        """索引持久化目录（按 profile 隔离）: storage/{data_profile}/"""
        return os.path.join(self.base_dir, "storage", self.data_profile)

    @property
    def skills_dir(self) -> str:
        """技能文档目录（跨 profile 共享）: data/skills/"""
        return os.path.join(self.base_dir, "data", "skills")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
