"""
统一配置: LLM + Embedding + 路径

========== 请在这里填入你的 API 信息 ==========
"""

import os

# ---- LLM 配置 (OpenAI 兼容 API) ----
# 替换为你的 API 地址和密钥
# 示例:
#   DeepSeek:  api_base="https://api.deepseek.com/v1",  model="deepseek-chat"
#   智谱:      api_base="https://open.bigmodel.cn/api/paas/v4", model="glm-4-flash"
#   月之暗面:   api_base="https://api.moonshot.cn/v1",   model="moonshot-v1-8k"
#   硅基流动:   api_base="https://api.siliconflow.cn/v1", model="deepseek-ai/DeepSeek-V3"
#   OpenAI:    api_base="https://api.openai.com/v1",    model="gpt-4o-mini"

LLM_API_BASE = "https://api.openai.com/v1"               # <-- 改成你的 API 地址
LLM_API_KEY = "sk-your-api-key-here"                      # <-- 改成你的 API Key
LLM_MODEL = "gpt-4o-mini"                                 # <-- 改成你的模型名称

# ---- Embedding 配置 ----
# 如果你的 API 提供商也支持 embedding，可以用同一个地址
# 如果不支持，设 USE_LOCAL_EMBEDDING = True 使用本地模型 (CPU 可跑, 免费)
USE_LOCAL_EMBEDDING = False

EMBEDDING_API_BASE = "https://api.openai.com/v1"          # <-- embedding API 地址 (可与 LLM 相同)
EMBEDDING_API_KEY = "sk-your-api-key-here"                 # <-- embedding API Key
EMBEDDING_MODEL = "text-embedding-3-small"                 # <-- embedding 模型名

# 本地 embedding 备选 (USE_LOCAL_EMBEDDING=True 时生效)
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"       # 中文小模型, ~100MB, CPU 可跑

# ---- 路径配置 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SKILLS_DIR = os.path.join(DATA_DIR, "skills")

# ---- Memory 配置 ----
MEMORY_TOKEN_LIMIT = 30000
CHAT_HISTORY_TOKEN_RATIO = 0.7
