"""
eval_robust.py - Compare evaluation strategies for SpatialEval.

Runs 4 matching strategies (substring, exact, word_boundary, option_letter)
across all models and tasks, producing a comparison table and disagreement log.

Usage:
    python evals/eval_robust.py
"""

import json
import os
import re
import sys
from typing import Optional

import pandas as pd

# Reuse existing extraction functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evals.evaluation import (
    extract_answer_from_text_spatialmap,
    extract_answer_from_text_mazenav,
    extract_answer_from_text_spatialgrid,
    extract_answer_from_text_spatialreal,
    extract_model_name,
)

DATA_DIR = "outputs/MilaWang__SpatialEval/vqa/all/"

EXTRACTORS = {
    "spatialmap": extract_answer_from_text_spatialmap,
    "mazenav": extract_answer_from_text_mazenav,
    "spatialgrid": extract_answer_from_text_spatialgrid,
    "spatialreal": extract_answer_from_text_spatialreal,
}

TASK_QID_LABELS = {
    ("spatialmap", 0): "Map: Direction",
    ("spatialmap", 1): "Map: Object",
    ("spatialmap", 2): "Map: Count",
    ("mazenav", 0): "Maze: Right Turns",
    ("mazenav", 1): "Maze: Total Turns",
    ("mazenav", 2): "Maze: Yes/No",
    ("spatialgrid", 0): "Grid: Count",
    ("spatialgrid", 1): "Grid: Animal ID",
    ("spatialgrid", 2): "Grid: Animal Pos",
    ("spatialreal", 0): "Real: Mixed",
}

MODEL_SHORT = {
    "BytedanceDouyinContent__SAIL-VL2-8B": "SAIL-VL2-8B",
    "OpenGVLab__InternVL3-8B-Instruct": "InternVL3-8B",
    "OpenGVLab__InternVL3_5-4B-HF": "InternVL3-4B",
    "Qwen__Qwen3-VL-4B-Instruct": "Qwen3-4B",
    "Qwen__Qwen3-VL-8B-Instruct": "Qwen3-8B",
    "allenai__Molmo2-4B": "Molmo2-4B",
    "allenai__Molmo2-8B": "Molmo2-8B",
    "google__gemma-3-4b-it": "Gemma3-4B",
    "lmms-lab__LLaVA-OneVision-1.5-4B-Instruct": "LLaVA-OV-4B",
    "lmms-lab__LLaVA-OneVision-1.5-8B-Instruct": "LLaVA-OV-8B",
    "openbmb__MiniCPM-V-4_5": "MiniCPM-V-4.5",
}


# ============================================================
# Option letter extraction
# ============================================================

def extract_option_letter(raw_answer: str) -> Optional[str]:
    """Extract option letter (A/B/C/D) from raw model answer."""
    if not raw_answer or not raw_answer.strip():
        return None
    raw = raw_answer.strip()
    patterns = [
        r'\b([A-D])[\\.]\s',          # "A. " or "B. "
        r'\b([A-D])\:\s',             # "A: "
        r'^([A-D])$',                  # bare letter
        r'(?i)the answer is ([A-D])',  # "The answer is A"
        r'\*\*Answer:\*\*\s*([A-D])',  # "**Answer:** A"
        r'\*\*Concise Answer:\*\*\s*.*?([A-D])[\\.]\s',  # "**Concise Answer:** ... A. "
    ]
    for p in patterns:
        m = re.search(p, raw)
        if m:
            return m.group(1).upper()
    return None


# ============================================================
# Matching strategies
# ============================================================

def match_substring(ref: str, extracted: str) -> bool:
    """Current baseline: substring containment."""
    if not ref or not extracted:
        return False
    return ref.lower().strip() in extracted.lower().strip()


def match_exact(ref: str, extracted: str) -> bool:
    """Strict equality after extraction."""
    if not ref or not extracted:
        return False
    return ref.lower().strip() == extracted.lower().strip()


def match_word_boundary(ref: str, extracted: str) -> bool:
    """Word-boundary regex match to prevent partial substrings."""
    if not ref or not extracted:
        return False
    ref_l = ref.lower().strip()
    ext_l = extracted.lower().strip()

    # For numeric answers, compare as integers to avoid "2" matching "25"
    try:
        return int(ref_l) == int(ext_l)
    except (ValueError, TypeError):
        pass

    # For non-numeric: word-boundary match
    return bool(re.search(rf'\b{re.escape(ref_l)}\b', ext_l))


def match_option_letter(raw_answer: str, oracle_option: str) -> bool:
    """Compare extracted option letter to oracle option."""
    extracted = extract_option_letter(raw_answer)
    if extracted is None or not oracle_option:
        return False
    return extracted.upper() == oracle_option.upper()


# ============================================================
# Main evaluation
# ============================================================

def evaluate_all_strategies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Run all 4 strategies on every sample across all models."""
    records = []
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".jsonl"))

    for filename in files:
        model_name = extract_model_name(filename)
        if not model_name:
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath) as fh:
            for line in fh:
                item = json.loads(line)
                sample_id = item["id"]
                parts = sample_id.split(".")
                task = parts[0]
                qid = int(parts[-1])
                raw_answer = item.get("answer", "")
                oracle_answer = item.get("oracle_answer", "")
                oracle_option = item.get("oracle_option", "")

                # Extract using existing per-task functions
                extractor = EXTRACTORS.get(task)
                if extractor is None:
                    continue
                try:
                    extracted = extractor(raw_answer, qid, model_name)
                except Exception:
                    extracted = None

                if extracted is not None:
                    extracted = str(extracted)

                oracle_str = str(oracle_answer).lower().strip() if oracle_answer else ""

                records.append({
                    "model": model_name,
                    "model_short": MODEL_SHORT.get(model_name, model_name),
                    "sample_id": sample_id,
                    "task": task,
                    "qid": qid,
                    "task_qid": TASK_QID_LABELS.get((task, qid), f"{task}:q{qid}"),
                    "oracle_answer": oracle_str,
                    "oracle_option": (oracle_option or "").upper(),
                    "extracted": (extracted or "").lower().strip(),
                    "raw_answer": raw_answer[:300] if raw_answer else "",
                    "substring": match_substring(oracle_str, extracted or ""),
                    "exact": match_exact(oracle_str, extracted or ""),
                    "word_boundary": match_word_boundary(oracle_answer or "", extracted or ""),
                    "option_letter": match_option_letter(raw_answer, oracle_option or ""),
                })

    return pd.DataFrame(records)


def compute_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy per (model, task) for each strategy."""
    strategies = ["substring", "exact", "word_boundary", "option_letter"]
    rows = []

    # Per task
    for (model, task), grp in df.groupby(["model_short", "task"]):
        row = {"model": model, "task": task, "level": "task"}
        for s in strategies:
            row[s] = grp[s].mean()
        rows.append(row)

    # Overall
    for model, grp in df.groupby("model_short"):
        row = {"model": model, "task": "overall", "level": "overall"}
        for s in strategies:
            row[s] = grp[s].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def flag_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Flag samples where substring and exact disagree."""
    dis = df[df["substring"] != df["exact"]].copy()
    dis["type"] = dis.apply(
        lambda r: "FP" if r["substring"] and not r["exact"] else "FN", axis=1
    )
    return dis.sort_values(["model_short", "task", "qid"])


def print_comparison(acc_df: pd.DataFrame):
    """Print formatted comparison table."""
    strategies = ["substring", "exact", "word_boundary", "option_letter"]

    # Print overall table
    overall = acc_df[acc_df["level"] == "overall"].sort_values("exact", ascending=False)
    print("\n" + "=" * 100)
    print("OVERALL ACCURACY COMPARISON")
    print("=" * 100)
    header = f"{'Model':<18} {'Substring':>10} {'Exact':>10} {'Word-Bnd':>10} {'Opt-Letr':>10} {'Delta':>8}"
    print(header)
    print("-" * 100)
    for _, row in overall.iterrows():
        delta = row["substring"] - row["exact"]
        print(
            f"{row['model']:<18} {row['substring']:>9.1%} {row['exact']:>9.1%} "
            f"{row['word_boundary']:>9.1%} {row['option_letter']:>9.1%} {delta:>+7.1%}"
        )

    # Print per-task table
    tasks = acc_df[acc_df["level"] == "task"]
    print("\n" + "=" * 100)
    print("PER-TASK ACCURACY (substring / exact)")
    print("=" * 100)
    models_ordered = overall["model"].tolist()
    task_order = ["spatialmap", "mazenav", "spatialgrid", "spatialreal"]
    header = f"{'Task':<16}" + "".join(f"{m:>14}" for m in models_ordered)
    print(header)
    print("-" * 100)
    for task in task_order:
        vals = []
        for m in models_ordered:
            match = tasks[(tasks["model"] == m) & (tasks["task"] == task)]
            if len(match) == 1:
                r = match.iloc[0]
                vals.append(f"{r['substring']:.0%}/{r['exact']:.0%}")
            else:
                vals.append("N/A")
        print(f"{task:<16}" + "".join(f"{v:>14}" for v in vals))


def main():
    print("Loading all model outputs...")
    df = evaluate_all_strategies()
    print(f"Loaded {len(df)} samples from {df['model'].nunique()} models")

    # Accuracy table
    acc = compute_accuracy_table(df)
    print_comparison(acc)

    # Disagreements
    dis = flag_disagreements(df)
    print(f"\nDisagreements (substring vs exact): {len(dis)} samples")
    fp = dis[dis["type"] == "FP"]
    fn = dis[dis["type"] == "FN"]
    print(f"  False Positives (substring says correct, exact says wrong): {len(fp)}")
    print(f"  False Negatives (substring says wrong, exact says correct): {len(fn)}")

    # Per-model FP counts
    if len(fp) > 0:
        print("\nFP breakdown by model:")
        for m, grp in fp.groupby("model_short"):
            print(f"  {m}: {len(grp)} FP")
            for task, tgrp in grp.groupby("task"):
                print(f"    {task}: {len(tgrp)}")

    # Export
    os.makedirs("eval_summary", exist_ok=True)
    df.to_csv("eval_summary/full_strategy_comparison.csv", index=False)
    dis.to_json("eval_summary/disagreements.jsonl", orient="records", lines=True)
    print(f"\nExported: eval_summary/full_strategy_comparison.csv ({len(df)} rows)")
    print(f"Exported: eval_summary/disagreements.jsonl ({len(dis)} rows)")


if __name__ == "__main__":
    main()
