"""
Full Evaluation: Run General ANoT Phase 2 on 100 restaurants (PARALLEL)
WITH AUPRC SCORING against ground truth!
"""

import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from explore.general_anot.phase2_v2 import FormulaSeedInterpreter
from explore.g1_gt_compute import compute_gt_for_k
from explore.score_auprc import calculate_ordinal_auprc, CLASS_ORDER, print_report, DEFAULT_TOLERANCES_V2
from dataclasses import asdict


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load restaurants from dataset_K200.jsonl."""
    entries = []
    with open(dataset_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


async def evaluate_restaurant(
    seed: Dict[str, Any],
    entry: Dict[str, Any],
    idx: int,
    total: int
) -> Dict[str, Any]:
    """Evaluate a single restaurant."""
    restaurant = entry.get("business", {})
    reviews = entry.get("reviews", [])
    name = restaurant.get("name", "Unknown")

    print(f"[{idx+1}/{total}] {name} ({len(reviews)} reviews)...", end=" ", flush=True)

    interpreter = FormulaSeedInterpreter(seed, verbose=False)

    try:
        result = await interpreter.execute(reviews, restaurant)

        # Get ground truth
        try:
            gt = compute_gt_for_k(name, k=200, version="v2")
            gt_verdict = gt.verdict
            gt_score = gt.final_risk_score
            gt_dict = asdict(gt)
        except (ValueError, KeyError):
            gt_verdict = None
            gt_score = None
            gt_dict = {}

        pred_verdict = result.get("VERDICT", "Unknown")
        pred_score = result.get("FINAL_RISK_SCORE", 0)

        match = "✓" if pred_verdict == gt_verdict else "✗"
        gt_str = f"{gt_score:.1f}" if gt_score else "N/A"
        print(f"{pred_verdict} (pred={pred_score:.1f}, gt={gt_str}) {match}")

        # Build parsed dict for primitive accuracy (uppercase keys)
        parsed = {k.upper(): v for k, v in result.items() if isinstance(v, (int, float, str))}

        return {
            "restaurant_id": restaurant.get("business_id"),
            "restaurant_name": name,
            "categories": restaurant.get("categories"),
            "n_reviews": len(reviews),
            "gt_verdict": gt_verdict,
            "gt_score": gt_score,
            "pred_verdict": pred_verdict,
            "pred_score": pred_score,
            "ground_truth": gt_dict,
            "parsed": parsed,
            **result
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "restaurant_id": restaurant.get("business_id"),
            "restaurant_name": name,
            "error": str(e)
        }


async def run_parallel_evaluation(
    seed: Dict[str, Any],
    entries: List[Dict[str, Any]],
    max_concurrent: int = 20
) -> List[Dict[str, Any]]:
    """Run evaluation on all restaurants in parallel."""
    total = len(entries)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_eval(entry, idx):
        async with semaphore:
            return await evaluate_restaurant(seed, entry, idx, total)

    tasks = [limited_eval(entry, i) for i, entry in enumerate(entries)]
    results = await asyncio.gather(*tasks)
    return list(results)


def compute_primitive_accuracy(parsed: Dict, ground_truth: Dict, tolerances: Dict[str, float]) -> float:
    """Compute primitive accuracy for a single restaurant."""
    n_correct = 0
    n_total = 0

    for field, tol in tolerances.items():
        if field == 'final_risk_score':
            continue  # Skip final score (captured by AUPRC)

        gt_val = ground_truth.get(field)
        pred_val = parsed.get(field.upper())

        if gt_val is None or pred_val is None:
            continue

        n_total += 1

        # Check if within tolerance
        if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            if abs(float(gt_val) - float(pred_val)) <= tol:
                n_correct += 1
        elif gt_val == pred_val:
            n_correct += 1

    return n_correct / n_total if n_total > 0 else 0.0


def compute_auprc(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute ordinal AUPRC and adjusted AUPRC from results."""
    y_true_ordinal = []
    y_scores = []
    primitive_accuracies = []

    for r in results:
        if "error" in r:
            continue

        gt_verdict = r.get("gt_verdict")
        pred_score = r.get("pred_score")
        gt_dict = r.get("ground_truth", {})
        parsed = r.get("parsed", {})

        if gt_verdict in CLASS_ORDER and pred_score is not None:
            y_true_ordinal.append(CLASS_ORDER[gt_verdict])
            y_scores.append(float(pred_score))

            # Compute primitive accuracy for this restaurant
            if gt_dict and parsed:
                prim_acc = compute_primitive_accuracy(parsed, gt_dict, DEFAULT_TOLERANCES_V2)
                primitive_accuracies.append(prim_acc)

    if len(y_true_ordinal) == 0:
        return {"error": "No valid predictions"}

    y_true_ordinal = np.array(y_true_ordinal)
    y_scores = np.array(y_scores)

    metrics = calculate_ordinal_auprc(y_true_ordinal, y_scores)

    # Add primitive accuracy metrics
    if primitive_accuracies:
        avg_prim_acc = np.mean(primitive_accuracies)
        metrics['avg_primitive_accuracy'] = float(avg_prim_acc)
        metrics['adjusted_auprc'] = metrics['ordinal_auprc'] * avg_prim_acc
        metrics['adjusted_nap'] = metrics['ordinal_nap'] * avg_prim_acc
        metrics['primitive_accuracy_std'] = float(np.std(primitive_accuracies))
        metrics['primitive_accuracy_min'] = float(min(primitive_accuracies))
        metrics['primitive_accuracy_max'] = float(max(primitive_accuracies))

    return metrics


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize evaluation results."""
    verdicts_pred = {}
    verdicts_gt = {}
    correct = 0
    total_valid = 0

    for r in results:
        if "error" in r:
            continue

        pred = r.get("pred_verdict", "Unknown")
        gt = r.get("gt_verdict")

        verdicts_pred[pred] = verdicts_pred.get(pred, 0) + 1
        if gt:
            verdicts_gt[gt] = verdicts_gt.get(gt, 0) + 1
            total_valid += 1
            if pred == gt:
                correct += 1

    return {
        "total_restaurants": len(results),
        "valid_with_gt": total_valid,
        "verdict_accuracy": correct / total_valid if total_valid > 0 else 0,
        "pred_distribution": verdicts_pred,
        "gt_distribution": verdicts_gt,
    }


async def main():
    """Main entry point."""
    print("=" * 70)
    print("GENERAL ANOT FULL EVALUATION WITH AUPRC")
    print("=" * 70)

    # Paths
    dataset_path = Path(__file__).parent.parent / "data" / "dataset_K200.jsonl"
    seed_path = Path(__file__).parent.parent / "results" / "phase1_v2" / "formula_seed.json"
    output_dir = Path(__file__).parent.parent / "results" / "general_anot_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load formula seed
    print(f"\nLoading formula seed from: {seed_path}")
    with open(seed_path) as f:
        seed = json.load(f)
    print(f"  Task: {seed.get('task_name', 'unknown')}")

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    entries = load_dataset(dataset_path)
    total_reviews = sum(len(e.get("reviews", [])) for e in entries)
    print(f"  Restaurants: {len(entries)}")
    print(f"  Total reviews: {total_reviews}")

    # Run parallel evaluation
    print("\n" + "=" * 70)
    print("RUNNING PARALLEL EVALUATION")
    print("=" * 70)

    start_time = datetime.now()
    results = await run_parallel_evaluation(seed, entries, max_concurrent=20)
    elapsed = (datetime.now() - start_time).total_seconds()

    # Compute AUPRC
    print("\n" + "=" * 70)
    print("COMPUTING AUPRC")
    print("=" * 70)

    metrics = compute_auprc(results)
    if "error" not in metrics:
        print_report(metrics)
    else:
        print(f"  ERROR: {metrics['error']}")

    # Summary
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)

    summary = summarize_results(results)
    print(f"  Total restaurants: {summary['total_restaurants']}")
    print(f"  Valid with GT: {summary['valid_with_gt']}")
    print(f"  Verdict accuracy: {summary['verdict_accuracy']:.1%}")
    print(f"  Pred distribution: {summary['pred_distribution']}")
    print(f"  GT distribution: {summary['gt_distribution']}")
    print(f"\n  Elapsed time: {elapsed:.1f}s ({elapsed/len(results):.2f}s per restaurant)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "seed_path": str(seed_path),
            "dataset_path": str(dataset_path),
            "summary": summary,
            "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in metrics.items()},
            "elapsed_seconds": elapsed,
            "results": results
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_file}")

    # Final score
    print("\n" + "=" * 70)
    print("🎯 FINAL SCORE")
    print("=" * 70)
    ordinal = metrics.get('ordinal_auprc', 0)
    prim_acc = metrics.get('avg_primitive_accuracy', 0)
    adjusted = metrics.get('adjusted_auprc', 0)
    print(f"  Ordinal AUPRC:      {ordinal:.4f}")
    print(f"  Primitive Accuracy: {prim_acc:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  ADJUSTED AUPRC:     {adjusted:.4f}  = {ordinal:.3f} × {prim_acc:.3f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    asyncio.run(main())
