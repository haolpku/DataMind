"""
DataMind Web Server

FastAPI 后端，提供:
- 流式对话 API (SSE)
- Skills / Memory / RAG / GraphRAG / Database 的查询和管理 API

启动: python server.py
访问: http://localhost:8000
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

import config
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from modules.rag.indexer import get_or_create_index, load_documents, load_pre_chunked, build_index
from modules.graphrag.graph_rag import get_or_create_graph_index, build_graph_index
from modules.database.database import init_demo_database, create_sql_query_engine, DB_PATH
from modules.skills.tools import get_all_skills
from modules.skills.knowledge import get_or_create_skill_index, build_skill_index, load_skill_documents
from modules.agent.agent import create_agent
from modules.memory.memory import create_memory

import chromadb
from sqlalchemy import create_engine, text, inspect


app_state = {}


def _recreate_agent():
    """Helper to recreate agent with current state"""
    return create_agent(
        vector_index=app_state.get("vector_index"),
        graph_index=app_state.get("graph_index"),
        sql_query_engine=app_state.get("sql_query_engine"),
        skill_index=app_state.get("skill_index"),
        extra_tools=app_state.get("skills", []),
        llm=app_state["llm"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] 初始化 DataMind ...")

    llm = OpenAI(api_base=config.LLM_API_BASE, api_key=config.LLM_API_KEY, model=config.LLM_MODEL)
    if config.USE_LOCAL_EMBEDDING:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=config.LOCAL_EMBEDDING_MODEL)
    else:
        embed_model = OpenAIEmbedding(
            api_base=config.EMBEDDING_API_BASE, api_key=config.EMBEDDING_API_KEY,
            model_name=config.EMBEDDING_MODEL,
        )
    Settings.llm = llm
    Settings.embed_model = embed_model

    app_state["llm"] = llm
    app_state["vector_index"] = get_or_create_index()

    try:
        app_state["graph_index"] = get_or_create_graph_index()
    except Exception as e:
        print(f"[WARNING] GraphRAG 初始化失败: {e}")
        app_state["graph_index"] = None

    try:
        app_state["db_engine"] = init_demo_database()
        app_state["sql_query_engine"] = create_sql_query_engine(app_state["db_engine"])
    except Exception as e:
        print(f"[WARNING] Database 初始化失败: {e}")
        app_state["db_engine"] = None
        app_state["sql_query_engine"] = None

    app_state["skills"] = get_all_skills()

    try:
        app_state["skill_index"] = get_or_create_skill_index()
    except Exception as e:
        print(f"[WARNING] 知识型 Skills 初始化失败: {e}")
        app_state["skill_index"] = None

    app_state["agent"] = _recreate_agent()
    app_state["memory"] = create_memory()

    print("[INFO] DataMind 初始化完成")
    yield
    print("[INFO] DataMind 关闭")


app = FastAPI(title="DataMind", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)


# ============================================================
# Chat API (SSE streaming)
# ============================================================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    agent = app_state["agent"]
    memory = app_state["memory"]

    async def event_stream():
        try:
            handler = agent.run(req.message, memory=memory)
            response = await handler
            response_text = str(response)

            for i in range(0, len(response_text), 4):
                chunk = response_text[i:i+4]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.02)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# Skills API (工具型)
# ============================================================
@app.get("/api/skills")
async def list_skills():
    skills = app_state.get("skills", [])
    result = []
    for tool in skills:
        meta = tool.metadata
        result.append({
            "name": meta.name,
            "description": meta.description,
            "parameters": str(meta.fn_schema.schema()) if meta.fn_schema else "",
        })
    return {"skills": result, "count": len(result)}


# ============================================================
# Skills API (知识型)
# ============================================================
@app.get("/api/skills/knowledge")
async def list_knowledge_skills():
    skills_dir = config.SKILLS_DIR
    files = []
    if os.path.exists(skills_dir):
        for f in os.listdir(skills_dir):
            if f.endswith(".md") and not f.startswith("."):
                filepath = os.path.join(skills_dir, f)
                size = os.path.getsize(filepath)
                with open(filepath, "r", encoding="utf-8") as fh:
                    content = fh.read()
                first_line = content.split("\n")[0].strip().lstrip("# ") if content else ""
                files.append({"name": f, "size": size, "title": first_line})

    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    try:
        collection = chroma_client.get_or_create_collection("skills_knowledge")
        vector_count = collection.count()
    except Exception:
        vector_count = 0

    return {"files": files, "count": len(files), "vector_count": vector_count}


@app.post("/api/skills/knowledge/upload")
async def upload_knowledge_skill(file: UploadFile = File(...)):
    if not file.filename.endswith(".md"):
        raise HTTPException(400, "只支持 .md 格式的技能文件")
    os.makedirs(config.SKILLS_DIR, exist_ok=True)
    filepath = os.path.join(config.SKILLS_DIR, file.filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return {"status": "ok", "filename": file.filename, "size": len(content)}


@app.delete("/api/skills/knowledge/{filename}")
async def delete_knowledge_skill(filename: str):
    filepath = os.path.join(config.SKILLS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, "文件不存在")
    os.remove(filepath)
    return {"status": "ok", "message": f"已删除 {filename}"}


@app.post("/api/skills/knowledge/rebuild")
async def rebuild_knowledge_skills():
    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    try:
        chroma_client.delete_collection("skills_knowledge")
    except Exception:
        pass

    docs = load_skill_documents()
    if not docs:
        app_state["skill_index"] = None
        app_state["agent"] = _recreate_agent()
        return {"status": "ok", "message": "无技能文件，已清空索引"}

    skill_index = build_skill_index(documents=docs)
    app_state["skill_index"] = skill_index
    app_state["agent"] = _recreate_agent()
    return {"status": "ok", "message": f"技能知识索引重建完成 ({len(docs)} 个文档)"}


# ============================================================
# Memory API
# ============================================================
@app.get("/api/memory")
async def get_memory():
    memory = app_state.get("memory")
    if not memory:
        return {"chat_history": [], "summary": ""}
    try:
        messages = await memory.aget()
        chat_history = []
        for msg in messages:
            chat_history.append({
                "role": str(msg.role.value) if hasattr(msg.role, 'value') else str(msg.role),
                "content": str(msg.content) if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False, default=str),
            })
        return {"chat_history": chat_history, "count": len(chat_history)}
    except Exception as e:
        return {"chat_history": [], "error": str(e)}


@app.delete("/api/memory")
async def clear_memory():
    app_state["memory"] = create_memory()
    return {"status": "ok", "message": "记忆已清空"}


# ============================================================
# RAG Knowledge Base API
# ============================================================
@app.get("/api/rag/documents")
async def list_documents():
    files = []
    if os.path.exists(config.DATA_DIR):
        exclude_dirs = {"chunks", "triplets", "skills"}
        for root, dirs, filenames in os.walk(config.DATA_DIR):
            rel_root = os.path.relpath(root, config.DATA_DIR)
            top_dir = rel_root.split(os.sep)[0] if rel_root != "." else ""
            if top_dir in exclude_dirs:
                continue
            for f in filenames:
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, config.DATA_DIR)
                size = os.path.getsize(filepath)
                files.append({"name": relpath, "size": size, "path": filepath})

    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    collection = chroma_client.get_or_create_collection("rag_allinone")
    vector_count = collection.count()

    return {"documents": files, "vector_count": vector_count}


@app.post("/api/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    filepath = os.path.join(config.DATA_DIR, file.filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return {"status": "ok", "filename": file.filename, "size": len(content)}


@app.delete("/api/rag/documents/{filename}")
async def delete_document(filename: str):
    filepath = os.path.join(config.DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, "文件不存在")
    os.remove(filepath)
    return {"status": "ok", "message": f"已删除 {filename}"}


@app.post("/api/rag/rebuild")
async def rebuild_rag_index():
    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    try:
        chroma_client.delete_collection("rag_allinone")
    except Exception:
        pass

    pre_chunked_nodes = load_pre_chunked()
    if pre_chunked_nodes:
        index = build_index(nodes=pre_chunked_nodes)
        mode = "预分块"
    else:
        docs = load_documents()
        if not docs:
            raise HTTPException(400, "data 目录为空，且 data/chunks/ 中无预分块数据")
        index = build_index(documents=docs)
        mode = "文档分块"

    app_state["vector_index"] = index
    app_state["agent"] = _recreate_agent()
    return {"status": "ok", "message": f"RAG 索引重建完成 (模式: {mode})"}


# ============================================================
# GraphRAG API
# ============================================================
@app.get("/api/graphrag/status")
async def graphrag_status():
    graph_index = app_state.get("graph_index")
    if graph_index is None:
        return {"status": "not_loaded", "entities": [], "relations": []}

    try:
        store = graph_index.property_graph_store
        triplets = store.get_triplets()
        entities = set()
        relations = []
        for subj, rel, obj in triplets:
            entities.add(str(subj))
            entities.add(str(obj))
            relations.append({
                "subject": str(subj),
                "relation": str(rel),
                "object": str(obj),
            })
        return {
            "status": "loaded",
            "entity_count": len(entities),
            "relation_count": len(relations),
            "entities": sorted(entities),
            "relations": relations[:100],
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "entities": [], "relations": []}


@app.post("/api/graphrag/rebuild")
async def rebuild_graph_index():
    try:
        import shutil
        from modules.graphrag.graph_rag import GRAPH_STORAGE_DIR
        if os.path.exists(GRAPH_STORAGE_DIR):
            shutil.rmtree(GRAPH_STORAGE_DIR)
        graph_index = build_graph_index()
        app_state["graph_index"] = graph_index
        app_state["agent"] = _recreate_agent()
        return {"status": "ok", "message": "GraphRAG 重建完成"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================
# Database API
# ============================================================
@app.get("/api/database/tables")
async def list_tables():
    engine = app_state.get("db_engine")
    if not engine:
        return {"tables": []}

    inspector = inspect(engine)
    tables = []
    for table_name in inspector.get_table_names():
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append({"name": col["name"], "type": str(col["type"])})

        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()

        tables.append({"name": table_name, "columns": columns, "row_count": count})

    return {"tables": tables, "db_path": DB_PATH}


@app.get("/api/database/query")
async def query_table(table: str, limit: int = 50):
    engine = app_state.get("db_engine")
    if not engine:
        raise HTTPException(500, "数据库未初始化")

    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        raise HTTPException(404, f"表 {table} 不存在")

    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT * FROM {table} LIMIT {limit}")).fetchall()
        columns = [col["name"] for col in inspector.get_columns(table)]
        data = [dict(zip(columns, row)) for row in rows]

    return {"table": table, "columns": columns, "data": data, "count": len(data)}


# ============================================================
# 前端页面
# ============================================================
@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
