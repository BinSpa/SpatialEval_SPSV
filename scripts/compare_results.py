"""
compare_results.py - Compare SpatialEval benchmark results across models.

Usage:
    python scripts/compare_results.py --output_folder outputs --eval_summary_dir eval_summary

Generates a markdown comparison table and CSV with per-task, per-mode accuracy.
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SpatialEval results")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--eval_summary_dir", type=str, default="eval_summary")
    parser.add_argument("--dataset_id", type=str, default="MilaWang/SpatialEval")
    parser.add_argument("--output_report", type=str, default="results_comparison.md")
    return parser.parse_args()


def extract_model_name(filename: str, suffix: str = "_w_reason.jsonl") -> str:
    """Extract model name from output filename."""
    prefix = "m-"
    if filename.startswith(prefix) and filename.endswith(suffix):
        return filename[len(prefix):-len(suffix)]
    return None


def compute_accuracy(jsonl_path: str, model_name: str = None):
    """Compute accuracy from a JSONL output file with per-task breakdown."""
    from evals.evaluation import (
        extract_answer_from_text_spatialmap,
        extract_answer_from_text_mazenav,
        extract_answer_from_text_spatialgrid,
        extract_answer_from_text_spatialreal,
    )

    task_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            question_id = int(data['id'].split('.')[-1])
            task = data['id'].split('.')[0]

            try:
                if task == 'spatialmap':
                    model_answer = extract_answer_from_text_spatialmap(
                        data['answer'], question_id, model_name)
                elif task == 'mazenav':
                    model_answer = extract_answer_from_text_mazenav(
                        data['answer'], question_id, model_name)
                elif task == 'spatialgrid':
                    model_answer = extract_answer_from_text_spatialgrid(
                        data['answer'], question_id, model_name)
                elif task == 'spatialreal':
                    model_answer = extract_answer_from_text_spatialreal(
                        data['answer'], question_id, model_name)
                else:
                    continue

                ref_ans = str(data['oracle_answer']).lower().strip()
                model_answer_str = str(model_answer).lower().strip() if model_answer else ""

                eval_result = int(ref_ans in model_answer_str)
                task_stats[task]["correct"] += eval_result
                task_stats[task]["total"] += 1
            except (ValueError, TypeError):
                continue

    return dict(task_stats)


def find_all_results(output_folder: str, dataset_id: str):
    """Find all result JSONL files."""
    dataset_dir = os.path.join(output_folder, dataset_id.replace("/", "__"))
    results = []

    if not os.path.isdir(dataset_dir):
        return results

    for mode in os.listdir(dataset_dir):
        mode_dir = os.path.join(dataset_dir, mode)
        if not os.path.isdir(mode_dir):
            continue
        for task in os.listdir(mode_dir):
            task_dir = os.path.join(mode_dir, task)
            if not os.path.isdir(task_dir):
                continue
            for filename in os.listdir(task_dir):
                if filename.endswith(".jsonl"):
                    filepath = os.path.join(task_dir, filename)
                    model_name = extract_model_name(filename)
                    if model_name:
                        results.append({
                            "filepath": filepath,
                            "filename": filename,
                            "mode": mode,
                            "task": task,
                            "model_name": model_name,
                        })

    return results


def generate_report(all_results: list, output_path: str):
    """Generate markdown comparison report."""
    # Group results by mode
    by_mode = defaultdict(list)
    for r in all_results:
        by_mode[r["mode"]].append(r)

    lines = []
    lines.append("# SpatialEval Benchmark Comparison Report\n")
    lines.append(f"Generated from {len(all_results)} evaluation results.\n")

    for mode in ["vqa", "vtqa", "tqa"]:
        if mode not in by_mode:
            continue

        lines.append(f"\n## Mode: {mode.upper()}\n")

        mode_results = by_mode[mode]
        # Collect all models and tasks
        models = sorted(set(r["model_name"] for r in mode_results))
        tasks = sorted(set(r["task"] for r in mode_results))

        # Build accuracy table
        lines.append(f"\n| Model | " + " | ".join(tasks) + " | Overall |")
        lines.append("|" + "|".join(["---"] * (len(tasks) + 2)) + "|")

        model_accuracies = {}
        for model in models:
            model_accs = {}
            for task in tasks:
                matching = [r for r in mode_results
                           if r["model_name"] == model and r["task"] == task]
                if matching and matching[0].get("accuracy"):
                    model_accs[task] = matching[0]["accuracy"]
                else:
                    model_accs[task] = None

            model_accuracies[model] = model_accs

        for model in models:
            accs = model_accuracies[model]
            total_correct = 0
            total_samples = 0
            cells = []
            for task in tasks:
                acc_data = accs.get(task)
                if acc_data:
                    total = acc_data.get("total", 0)
                    correct = acc_data.get("correct", 0)
                    pct = correct / total * 100 if total > 0 else 0
                    cells.append(f"{pct:.1f}%")
                    total_correct += correct
                    total_samples += total
                else:
                    cells.append("-")

            overall = total_correct / total_samples * 100 if total_samples > 0 else 0
            short_name = model.split("__")[-1] if "__" in model else model
            lines.append(f"| {short_name} | " + " | ".join(cells) + f" | {overall:.1f}% |")

    # Add analysis section
    lines.append("\n## Notes\n")
    lines.append("- Accuracy computed with substring matching (same as original evaluation.py)")
    lines.append("- `w_reason` flag uses chain-of-thought prompting")
    lines.append("- `-` indicates no results available for that combination")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    return report


def main():
    args = parse_args()

    results = find_all_results(args.output_folder, args.dataset_id)
    if not results:
        print(f"No results found in {args.output_folder}")
        return

    print(f"Found {len(results)} result files.")

    # Compute accuracies
    for r in results:
        try:
            task_stats = compute_accuracy(r["filepath"], r["model_name"])
            r["accuracy"] = task_stats
        except Exception as e:
            print(f"Error computing accuracy for {r['filepath']}: {e}")
            r["accuracy"] = None

    # Generate report
    generate_report(results, args.output_report)
    print(f"\nReport saved to {args.output_report}")


if __name__ == "__main__":
    main()
