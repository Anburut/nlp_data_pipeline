import yaml
import os
import polars as pl
from src.pipeline import NLPPipeline
from src.analyzer import DatasetAnalyzer


def generate_report(stats, final_df, ratio_df):
    print("\n" + "="*51)
    print("NMT RESEARCH CORPUS FIREWALL v4.0")
    print("   English <-> Afaan Oromo | Academic Edition")
    print("="*51)
    print(f"\nCORPUS OVERVIEW")
    print(f"  Total pairs     : {stats['total']}")

    print("\n" + "-"*51)
    print("PHASE 1 - CORPUS INTEGRITY")
    print(f"  Empty / null rows     : {stats['P1_EMPTY']}")
    print(f"  Exact duplicates      : {stats['P2_IDENTICAL']}")
    p1_total = stats['P1_EMPTY'] + stats['P2_IDENTICAL']
    print(f"  Total removed P1      : {p1_total} ({(p1_total / stats['total']) * 100:.2f}%)")

    print("\n" + "-"*51)
    print("PHASE 2 - LANGUAGE IDENTIFICATION")
    print(f"  EN -> OM (Valid)        : {stats['valid']} ({(stats['valid'] / stats['total']) * 100:.2f}%)")
    print(f"  EN -> EN (Untranslated) : {stats['P2_EN_EN']}")
    print(f"  OM -> OM (Duplicated)   : {stats['P2_OM_OM']}")
    print(f"  OM -> EN (Swapped)      : {stats['P2_OM_EN']}")
    print(f"  XX -> OM (Wrong source) : {stats['P2_XX_OM']}")
    print(f"  EN -> XX (Wrong target) : {stats['P2_EN_XX']}")
    print(f"  Amharic/Ge'ez leak      : {stats['P2_SCRIPT']}")
    print(f"  HTML / URL noise        : {stats['P2_NOISE']}")
    print(f"  Identical pairs         : {stats['P2_IDENTICAL']}")
    print(f"  Unknown lang (XX_XX)    : {stats['P2_XX_XX']}")

    print("\n" + "-"*51)
    print("PHASE 5 - ALIGNMENT & LENGTH RATIO")
    mean_r = ratio_df["ratio"].mean()
    print(f"  Expected ratio (OM/EN)  : 1.20 - 2.50")
    print(f"  Mean ratio found        : {mean_r:.2f}")
    print("\n  Distribution")
    ratios = ratio_df["ratio"].drop_nulls().to_list()
    total = len(ratios)
    bands = [
        ("1.0-1.5", 1.0, 1.5),
        ("1.5-2.0", 1.5, 2.0),
        ("2.0-2.5", 2.0, 2.5),
        ("2.5+   ", 2.5, float("inf")),
    ]
    for label, lo, hi in bands:
        count = sum(1 for r in ratios if lo <= r < hi)
        pct = (count / total * 100) if total > 0 else 0
        bar = int(pct / 6.25)
        filled = "█" * bar
        empty = "░" * (16 - bar)
        removed = "  <- removed" if lo >= 2.5 else ""
        print(f"  {label}  {filled}{empty}  {pct:.1f}%{removed}")

    print("\n" + "-"*51)
    print("PHASE 10 - FINAL DELIVERABLES")
    print(f"  cleaned_corpus    : {len(final_df)} pairs")
    print(f"  audit_log.csv     : {stats['REMOVE']} removed rows")
    print(f"  review_queue.csv  : generated")
    print("="*51)

    print("\nTHESIS WARNINGS")
    print("  [1] langid removed -> Oromo heuristic classifier applied")
    print("  [2] Vocabulary imbalance EN:OM -> Expected for agglutinative morphology")
    print("  [3] Domain Bias -> Check if Bible data dominates corpus")
    print("\n" + "="*51)


def main():
    input_path = input("Enter source corpus path: ").strip()
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    BASE = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE, "config", "pipeline_config.yaml")

    analyzer = DatasetAnalyzer()
    analyzed_df, stats = analyzer.analyze(input_path)

    proceed = input("\nProceed to Research Clean & Export? (y/n): ").lower()
    if proceed != 'y':
        return

    print("\nFormat: 1.JSONL | 2.CSV | 3.TSV")
    choice = input("Choice: ").strip()
    fmt_map = {"1": ("jsonl", ".jsonl"), "2": ("csv", ".csv"), "3": ("tsv", ".tsv")}
    selected_fmt, ext = fmt_map.get(choice, ("jsonl", ".jsonl"))

    output_path = os.path.join(BASE, "data", "output", f"cleaned_corpus{ext}")
    audit_path  = os.path.join(BASE, "data", "output", "audit_log.csv")
    review_path = os.path.join(BASE, "data", "output", "review_queue.csv")

    os.makedirs(os.path.join(BASE, "data", "output"), exist_ok=True)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    pipeline = NLPPipeline(config_data.get("pipeline", []))
    try:
        final_df, ratio_df = pipeline.run_research_mode(
            analyzed_df, output_path, audit_path, review_path, format=selected_fmt
        )
        generate_report(stats, final_df, ratio_df)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()