"""
DataMind 并发推理 Benchmark

直接调用 Python API（不经过 HTTP），支持并发控制和指标统计。

用法:
  python -m benchmark.run --questions data/bench/questions.jsonl --concurrency 5

  RETRIEVER_MODE=multi_query python -m benchmark.run --questions data/bench/questions.jsonl

  DATA_PROFILE=2wiki python -m benchmark.run --questions data/bench/2wiki.jsonl

问题集格式 (JSONL，每行一个 JSON):
  {"question": "RAG的核心原理是什么？"}
  {"question": "When did X happen?", "reference_answer": "1982"}
"""

import argparse
import asyncio
import json
import logging
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _suppress_noisy_loggers():
    for name in ("httpx", "openai", "httpcore", "llama_index"):
        logging.getLogger(name).setLevel(logging.WARNING)


from core.bootstrap import initialize
from core.session import SessionManager


_completed = 0
_total = 0
_errors = 0
_lock = asyncio.Lock()


def _progress_bar(completed: int, total: int, errors: int, width: int = 40):
    pct = completed / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    err_str = f"  err={errors}" if errors else ""
    print(f"\r  [{bar}] {completed}/{total} ({pct*100:.1f}%){err_str}", end="", flush=True)


async def _run_one(idx: int, item: dict, agent, session_mgr: SessionManager, semaphore: asyncio.Semaphore):
    global _completed, _errors
    async with semaphore:
        memory = session_mgr.get_memory(f"bench_{idx}")
        question = item["question"]
        start = time.perf_counter()
        try:
            response = await agent.run(question, memory=memory)
            answer = str(response)
            error = None
        except Exception as e:
            answer = ""
            error = str(e)
        elapsed = time.perf_counter() - start

        async with _lock:
            _completed += 1
            if error:
                _errors += 1
            _progress_bar(_completed, _total, _errors)

        result = {
            "index": idx,
            "question": question,
            "answer": answer,
            "error": error,
            "latency_s": round(elapsed, 3),
        }
        if "reference_answer" in item:
            result["reference_answer"] = item["reference_answer"]
        if "question_id" in item:
            result["question_id"] = item["question_id"]
        return result


async def run_benchmark(items: list[dict], agent, session_mgr: SessionManager, concurrency: int = 5):
    global _completed, _total, _errors
    _completed, _errors = 0, 0
    _total = len(items)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _run_one(i, item, agent, session_mgr, semaphore)
        for i, item in enumerate(items)
    ]

    print(f"\n  Running {_total} queries (concurrency={concurrency}) ...")
    _progress_bar(0, _total, 0)

    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_elapsed = time.perf_counter() - wall_start

    print()

    latencies = [r["latency_s"] for r in results]
    errors = [r for r in results if r["error"]]
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    print("\n" + "=" * 50)
    print("  Benchmark Results")
    print("=" * 50)
    print(f"  Total queries:  {n}")
    print(f"  Concurrency:    {concurrency}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Wall time:      {wall_elapsed:.3f}s")
    print("-" * 50)
    print(f"  Avg latency:    {sum(latencies) / n:.3f}s")
    print(f"  Min latency:    {sorted_lat[0]:.3f}s")
    print(f"  P50 latency:    {sorted_lat[n // 2]:.3f}s")
    print(f"  P90 latency:    {sorted_lat[int(n * 0.9)]:.3f}s")
    print(f"  P95 latency:    {sorted_lat[int(n * 0.95)]:.3f}s")
    print(f"  Max latency:    {sorted_lat[-1]:.3f}s")
    print(f"  Throughput:     {n / wall_elapsed:.2f} QPS")
    print("=" * 50)

    return results


def load_questions(filepath: str) -> list[dict]:
    """加载问题集，返回 dict 列表（保留 reference_answer 等可选字段）。"""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("question", "").strip():
                    items.append(item)
            except json.JSONDecodeError:
                items.append({"question": line})
    return items


def main():
    parser = argparse.ArgumentParser(description="DataMind Benchmark Runner")
    parser.add_argument("--questions", required=True, help="JSONL 文件路径")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数 (default: 5)")
    parser.add_argument("--output", default="benchmark_results.json", help="结果输出文件")
    args = parser.parse_args()

    items = load_questions(args.questions)
    if not items:
        print("[ERROR] 问题集为空，请检查文件格式")
        sys.exit(1)

    has_ref = sum(1 for it in items if "reference_answer" in it)
    print(f"[INFO] 加载了 {len(items)} 个问题 ({has_ref} 条含参考答案)，并发数: {args.concurrency}")

    from config import settings as cfg
    print(f"[INFO] Data profile: {cfg.data_profile}")

    _suppress_noisy_loggers()
    state = initialize()
    session_mgr = SessionManager()

    results = asyncio.run(run_benchmark(items, state.agent, session_mgr, args.concurrency))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 结果已保存到: {args.output}")
    if has_ref:
        print(f"[TIP]  运行评估: python -m benchmark.evaluate {args.output}")


if __name__ == "__main__":
    main()
