"""
Database 模块: 自然语言查询数据库 (NL2SQL)

使用 NLSQLTableQueryEngine + SQLite
用户用自然语言提问，LLM 自动生成 SQL 并执行查询。

本 demo 创建一个示例员工数据库用于演示。
"""

import os
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, text

from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine

from config import settings

DB_PATH = os.path.join(settings.storage_dir, "demo.db")


def init_demo_database():
    """创建并填充示例数据库"""
    os.makedirs(settings.storage_dir, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
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
            return engine

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

    print(f"[Database] 示例数据库创建完成: {DB_PATH}")
    print("[Database]   - employees 表: 8 条记录 (员工信息)")
    print("[Database]   - projects 表: 4 条记录 (项目信息)")
    return engine


def create_sql_query_engine(engine=None):
    """创建自然语言 SQL 查询引擎"""
    if engine is None:
        engine = init_demo_database()

    sql_database = SQLDatabase(
        engine,
        include_tables=["employees", "projects"],
    )

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["employees", "projects"],
    )

    return query_engine
