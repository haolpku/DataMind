"""
DataMind 并发推理 Benchmark

直接调用 Python API（不经过 HTTP），支持并发控制和指标统计。

用法:
  # 默认配置跑 benchmark
  python benchmark.py --questions data/bench_questions.jsonl --concurrency 5

  # 切换检索模式
  RETRIEVER_MODE=multi_query python benchmark.py --questions data/bench_questions.jsonl

  # 切换模型
  LLM_MODEL=deepseek-chat python benchmark.py --questions data/bench_questions.jsonl

问题集格式 (JSONL，每行一个 JSON):
  {"question": "RAG的核心原理是什么？"}
  {"question": "张三的工资是多少？"}
"""

import argparse
import asyncio
import json
import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _suppress_noisy_loggers():
    """Suppress per-request HTTP logs from httpx / openai / llama_index."""
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


async def run_one(idx: int, question: str, agent, session_mgr: SessionManager, semaphore: asyncio.Semaphore):
    global _completed, _errors
    async with semaphore:
        memory = session_mgr.get_memory(f"bench_{idx}")
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

        return {
            "index": idx,
            "question": question,
            "answer": answer,
            "error": error,
            "latency_s": round(elapsed, 3),
        }


async def run_benchmark(questions: list[str], agent, session_mgr: SessionManager, concurrency: int = 5):
    global _completed, _total, _errors
    _completed, _errors = 0, 0
    _total = len(questions)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        run_one(i, q, agent, session_mgr, semaphore)
        for i, q in enumerate(questions)
    ]

    print(f"\n  Running {_total} queries (concurrency={concurrency}) ...")
    _progress_bar(0, _total, 0)

    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_elapsed = time.perf_counter() - wall_start

    print()  # newline after progress bar

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


def load_questions(filepath: str) -> list[str]:
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                q = item.get("question", "").strip()
                if q:
                    questions.append(q)
            except json.JSONDecodeError:
                questions.append(line)
    return questions


def main():
    parser = argparse.ArgumentParser(description="DataMind Benchmark")
    parser.add_argument("--questions", required=True, help="JSONL 文件路径，每行 {\"question\": \"...\"}")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数 (default: 5)")
    parser.add_argument("--output", default="benchmark_results.json", help="结果输出文件 (default: benchmark_results.json)")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("[ERROR] 问题集为空，请检查文件格式")
        sys.exit(1)

    print(f"[INFO] 加载了 {len(questions)} 个问题，并发数: {args.concurrency}")

    _suppress_noisy_loggers()
    state = initialize()
    session_mgr = SessionManager()

    results = asyncio.run(run_benchmark(questions, state.agent, session_mgr, args.concurrency))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
