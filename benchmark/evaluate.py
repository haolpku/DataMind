"""
Benchmark 评估脚本 —— 对比生成答案与标准答案

支持两种匹配指标:
  - Exact Match (EM): 标准答案是否完整出现在生成答案中（大小写不敏感）
  - F1 Score: 基于 token 重叠的 F1（适用于答案较长 / 表述不一致的场景）

用法:
  python -m benchmark.evaluate benchmark_results.json
  python -m benchmark.evaluate benchmark_results.json --output eval_report.json
"""

import argparse
import json
import re
import sys
from collections import Counter


def _normalize(text: str) -> str:
    """统一大小写、去除多余空白和标点。"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def exact_match(prediction: str, reference: str) -> bool:
    """标准答案的文本（normalize 后）是否完整出现在生成答案中。"""
    return _normalize(reference) in _normalize(prediction)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(results: list[dict]) -> dict:
    """对结果列表做评估，返回汇总 + 逐条明细。"""
    evaluated = []
    em_correct = 0
    f1_sum = 0.0
    total = 0

    for r in results:
        ref = r.get("reference_answer")
        if ref is None:
            continue
        total += 1
        answer = r.get("answer", "")
        em = exact_match(answer, ref)
        f1 = token_f1(answer, ref)
        if em:
            em_correct += 1
        f1_sum += f1

        evaluated.append({
            "index": r.get("index"),
            "question_id": r.get("question_id"),
            "question": r["question"],
            "answer": answer,
            "reference_answer": ref,
            "exact_match": em,
            "f1": round(f1, 4),
        })

    summary = {
        "total": total,
        "exact_match_count": em_correct,
        "exact_match_rate": round(em_correct / total, 4) if total else 0,
        "avg_f1": round(f1_sum / total, 4) if total else 0,
    }
    return {"summary": summary, "details": evaluated}


def _print_report(report: dict):
    s = report["summary"]
    print("\n" + "=" * 55)
    print("  Evaluation Report")
    print("=" * 55)
    print(f"  Total evaluated:   {s['total']}")
    print(f"  Exact Match:       {s['exact_match_count']}/{s['total']} ({s['exact_match_rate']*100:.1f}%)")
    print(f"  Avg Token-F1:      {s['avg_f1']:.4f}")
    print("=" * 55)

    details = report["details"]
    wrong = [d for d in details if not d["exact_match"]]
    if wrong:
        print(f"\n  ✗ 未命中的问题 ({len(wrong)} 条):\n")
        for d in wrong[:20]:
            print(f"    Q: {d['question']}")
            print(f"    A: {d['answer'][:120]}{'...' if len(d['answer']) > 120 else ''}")
            print(f"    R: {d['reference_answer']}")
            print(f"    F1={d['f1']:.4f}")
            print()


def main():
    parser = argparse.ArgumentParser(description="DataMind Benchmark Evaluator")
    parser.add_argument("results", help="benchmark runner 输出的 JSON 文件路径")
    parser.add_argument("--output", default=None, help="评估报告输出路径 (JSON)")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    has_ref = sum(1 for r in results if "reference_answer" in r)
    if has_ref == 0:
        print("[ERROR] 结果文件中没有 reference_answer 字段，无法评估。")
        print("        请确认问题集 JSONL 包含 reference_answer，并使用新版 benchmark runner 重跑。")
        sys.exit(1)

    report = evaluate(results)
    _print_report(report)

    out_path = args.output or args.results.replace(".json", "_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 评估报告已保存到: {out_path}")


if __name__ == "__main__":
    main()
