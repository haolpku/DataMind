"""
Database 模块: 自然语言查询数据库 (NL2SQL)

使用 NLSQLTableQueryEngine + SQLite
用户用自然语言提问，LLM 自动生成 SQL 并执行查询。

数据来源（按优先级）:
  1. profile 目录下 tables/*.sql 文件 -> 执行建表 + 插入数据
  2. 无 .sql 文件时 -> fallback 到内置 demo 员工数据库
"""

import glob
import os
from sqlalchemy import create_engine, inspect, text

from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine

from config import settings


def _db_path() -> str:
    return os.path.join(settings.storage_dir, "demo.db")


def _tables_dir() -> str:
    return os.path.join(settings.data_dir, "tables")


def _load_sql_files(engine, tables_dir: str) -> list[str]:
    """扫描并执行 tables_dir 下所有 .sql 文件，返回已创建的表名列表。"""
    sql_files = sorted(glob.glob(os.path.join(tables_dir, "*.sql")))
    if not sql_files:
        return []

    with engine.connect() as conn:
        for fpath in sql_files:
            with open(fpath, "r", encoding="utf-8") as f:
                sql_text = f.read().strip()
            if not sql_text:
                continue
            for statement in sql_text.split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            print(f"[Database] 已执行 SQL 文件: {os.path.basename(fpath)}")
        conn.commit()

    table_names = inspect(engine).get_table_names()
    print(f"[Database] 从 SQL 文件导入完成，共 {len(table_names)} 张表: {table_names}")
    return table_names


def _init_demo_tables(engine):
    """内置 demo 数据库（员工 + 项目），作为无 .sql 文件时的 fallback。"""
    from sqlalchemy import MetaData, Table, Column, String, Integer, Float

    metadata = MetaData()

    employees = Table(
        "employees", metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(50), nullable=False),
        Column("department", String(50), nullable=False),
        Column("position", String(50), nullable=False),
        Column("salary", Float, nullable=False),
        Column("city", String(50), nullable=False),
    )

    projects = Table(
        "projects", metadata,
        Column("id", Integer, primary_key=True),
        Column("project_name", String(100), nullable=False),
        Column("lead_employee_id", Integer, nullable=False),
        Column("budget", Float, nullable=False),
        Column("status", String(20), nullable=False),
    )

    metadata.create_all(engine)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM employees")).scalar()
        if count > 0:
            print(f"[Database] 示例数据库已存在 ({count} 条员工记录)")
            return ["employees", "projects"]

        conn.execute(employees.insert(), [
            {"id": 1, "name": "张三", "department": "工程部", "position": "高级工程师", "salary": 35000, "city": "北京"},
            {"id": 2, "name": "李四", "department": "工程部", "position": "工程师", "salary": 25000, "city": "上海"},
            {"id": 3, "name": "王五", "department": "产品部", "position": "产品经理", "salary": 30000, "city": "北京"},
            {"id": 4, "name": "赵六", "department": "设计部", "position": "UI设计师", "salary": 22000, "city": "杭州"},
            {"id": 5, "name": "孙七", "department": "工程部", "position": "架构师", "salary": 45000, "city": "北京"},
            {"id": 6, "name": "周八", "department": "市场部", "position": "市场总监", "salary": 38000, "city": "上海"},
            {"id": 7, "name": "吴九", "department": "工程部", "position": "工程师", "salary": 26000, "city": "深圳"},
            {"id": 8, "name": "郑十", "department": "产品部", "position": "产品总监", "salary": 40000, "city": "北京"},
        ])
        conn.execute(projects.insert(), [
            {"id": 1, "project_name": "RAG智能助手", "lead_employee_id": 5, "budget": 500000, "status": "进行中"},
            {"id": 2, "project_name": "数据分析平台", "lead_employee_id": 1, "budget": 300000, "status": "进行中"},
            {"id": 3, "project_name": "移动端App", "lead_employee_id": 3, "budget": 200000, "status": "已完成"},
            {"id": 4, "project_name": "品牌推广", "lead_employee_id": 6, "budget": 150000, "status": "规划中"},
        ])
        conn.commit()

    print(f"[Database] 示例数据库创建完成: {_db_path()}")
    print("[Database]   - employees 表: 8 条记录 (员工信息)")
    print("[Database]   - projects 表: 4 条记录 (项目信息)")
    return ["employees", "projects"]


def init_database():
    """初始化数据库，优先从 profile/tables/*.sql 导入，否则 fallback 到 demo 数据。

    Returns:
        (engine, table_names): SQLAlchemy engine 和可查询的表名列表
    """
    os.makedirs(settings.storage_dir, exist_ok=True)
    db_path = _db_path()
    engine = create_engine(f"sqlite:///{db_path}")

    tables_dir = _tables_dir()
    if os.path.isdir(tables_dir) and glob.glob(os.path.join(tables_dir, "*.sql")):
        existing = inspect(engine).get_table_names()
        if existing:
            print(f"[Database] 数据库已存在 ({len(existing)} 张表)，直接加载")
            return engine, existing
        table_names = _load_sql_files(engine, tables_dir)
        return engine, table_names

    table_names = _init_demo_tables(engine)
    return engine, table_names


# Backward compatibility
def init_demo_database():
    engine, _ = init_database()
    return engine


def create_sql_query_engine(engine=None, table_names: list[str] | None = None):
    """创建自然语言 SQL 查询引擎"""
    if engine is None:
        engine, table_names = init_database()

    if table_names is None:
        table_names = inspect(engine).get_table_names()

    sql_database = SQLDatabase(
        engine,
        include_tables=table_names,
    )

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=table_names,
    )

    return query_engine
