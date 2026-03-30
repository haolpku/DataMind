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
from pydantic import BaseModel
from typing import Optional

from config import settings
from core.bootstrap import initialize, recreate_agent, AppState
from core.session import SessionManager

from modules.rag.indexer import load_documents, load_pre_chunked, build_index
from modules.graphrag.graph_rag import build_graph_index, GRAPH_STORAGE_DIR
from modules.skills.knowledge import build_skill_index, load_skill_documents
from modules.database.database import DB_PATH
from modules.memory.memory import create_memory

import chromadb
from sqlalchemy import text, inspect


_state: AppState = None  # type: ignore[assignment]
_session_mgr: SessionManager = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state, _session_mgr
    _state = initialize()
    _session_mgr = SessionManager()
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
    agent = _state.agent
    memory = _session_mgr.get_memory(req.session_id)

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
    skills = _state.skills or []
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
    skills_dir = settings.skills_dir
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

    chroma_client = chromadb.PersistentClient(path=settings.storage_dir)
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
    os.makedirs(settings.skills_dir, exist_ok=True)
    filepath = os.path.join(settings.skills_dir, file.filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return {"status": "ok", "filename": file.filename, "size": len(content)}


@app.delete("/api/skills/knowledge/{filename}")
async def delete_knowledge_skill(filename: str):
    filepath = os.path.join(settings.skills_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, "文件不存在")
    os.remove(filepath)
    return {"status": "ok", "message": f"已删除 {filename}"}


@app.post("/api/skills/knowledge/rebuild")
async def rebuild_knowledge_skills():
    chroma_client = chromadb.PersistentClient(path=settings.storage_dir)
    try:
        chroma_client.delete_collection("skills_knowledge")
    except Exception:
        pass

    docs = load_skill_documents()
    if not docs:
        _state.skill_index = None
        recreate_agent(_state)
        return {"status": "ok", "message": "无技能文件，已清空索引"}

    skill_index = build_skill_index(documents=docs)
    _state.skill_index = skill_index
    recreate_agent(_state)
    return {"status": "ok", "message": f"技能知识索引重建完成 ({len(docs)} 个文档)"}


# ============================================================
# Memory API
# ============================================================
@app.get("/api/memory")
async def get_memory(session_id: str = "default"):
    memory = _session_mgr.get_memory(session_id)
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
async def clear_memory(session_id: Optional[str] = None):
    if session_id:
        _session_mgr.clear(session_id)
        return {"status": "ok", "message": f"已清空 session {session_id} 的记忆"}
    _session_mgr.clear_all()
    return {"status": "ok", "message": "已清空所有 session 的记忆"}


# ============================================================
# RAG Knowledge Base API
# ============================================================
@app.get("/api/rag/documents")
async def list_documents():
    files = []
    data_dir = settings.data_dir
    if os.path.exists(data_dir):
        exclude_dirs = {"chunks", "triplets", "skills"}
        for root, dirs, filenames in os.walk(data_dir):
            rel_root = os.path.relpath(root, data_dir)
            top_dir = rel_root.split(os.sep)[0] if rel_root != "." else ""
            if top_dir in exclude_dirs:
                continue
            for f in filenames:
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, data_dir)
                size = os.path.getsize(filepath)
                files.append({"name": relpath, "size": size, "path": filepath})

    chroma_client = chromadb.PersistentClient(path=settings.storage_dir)
    collection = chroma_client.get_or_create_collection("rag_allinone")
    vector_count = collection.count()

    return {"documents": files, "vector_count": vector_count}


@app.post("/api/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs(settings.data_dir, exist_ok=True)
    filepath = os.path.join(settings.data_dir, file.filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return {"status": "ok", "filename": file.filename, "size": len(content)}


@app.delete("/api/rag/documents/{filename}")
async def delete_document(filename: str):
    filepath = os.path.join(settings.data_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, "文件不存在")
    os.remove(filepath)
    return {"status": "ok", "message": f"已删除 {filename}"}


@app.post("/api/rag/rebuild")
async def rebuild_rag_index():
    chroma_client = chromadb.PersistentClient(path=settings.storage_dir)
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

    _state.vector_index = index
    recreate_agent(_state)
    return {"status": "ok", "message": f"RAG 索引重建完成 (模式: {mode})"}


# ============================================================
# GraphRAG API
# ============================================================
@app.get("/api/graphrag/status")
async def graphrag_status():
    graph_index = _state.graph_index
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
        if os.path.exists(GRAPH_STORAGE_DIR):
            shutil.rmtree(GRAPH_STORAGE_DIR)
        graph_index = build_graph_index()
        _state.graph_index = graph_index
        recreate_agent(_state)
        return {"status": "ok", "message": "GraphRAG 重建完成"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================
# Database API
# ============================================================
@app.get("/api/database/tables")
async def list_tables():
    engine = _state.db_engine
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
    engine = _state.db_engine
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
