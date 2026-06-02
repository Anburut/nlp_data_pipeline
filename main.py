import yaml
import os
import polars as pl
from src.pipeline import NLPPipeline
from src.analyzer import DatasetAnalyzer


def export_corpus_stats(final_df, output_dir):
    source_tokens = final_df["Source"].str.split(" ").list.len()
    target_tokens = final_df["Target"].str.split(" ").list.len()

    all_source_tokens = " ".join(final_df["Source"].to_list()).split()
    all_target_tokens = " ".join(final_df["Target"].to_list()).split()

    source_vocab = set(all_source_tokens)
    target_vocab = set(all_target_tokens)

    total_src_tokens = len(all_source_tokens)
    total_tgt_tokens = len(all_target_tokens)

    stats = {
        "total_pairs":             len(final_df),
        "avg_source_tokens":       round(source_tokens.mean(), 2),
        "avg_target_tokens":       round(target_tokens.mean(), 2),
        "max_source_tokens":       source_tokens.max(),
        "max_target_tokens":       target_tokens.max(),
        "source_vocab_size":       len(source_vocab),
        "target_vocab_size":       len(target_vocab),
        "total_source_tokens":     total_src_tokens,
        "total_target_tokens":     total_tgt_tokens,
        "ttr_source":              round(len(source_vocab) / total_src_tokens, 4),
        "ttr_target":              round(len(target_vocab) / total_tgt_tokens, 4),
        "vocab_ratio_tgt_src":     round(len(target_vocab) / len(source_vocab), 4),
    }

    stats_path = os.path.join(output_dir, "corpus_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("CORPUS STATISTICS FOR THESIS\n")
        f.write("="*40 + "\n")
        for k, v in stats.items():
            f.write(f"  {k:<28}: {v}\n")

    return stats


def generate_report(stats, final_df, ratio_df, output_dir):
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
    print("PHASE 6 - CORPUS LINGUISTICS STATS")
    corpus_stats = export_corpus_stats(final_df, output_dir)
    print(f"  Source vocab size       : {corpus_stats['source_vocab_size']}")
    print(f"  Target vocab size       : {corpus_stats['target_vocab_size']}")
    print(f"  Vocab ratio (OM/EN)     : {corpus_stats['vocab_ratio_tgt_src']}")
    print(f"  Avg source tokens       : {corpus_stats['avg_source_tokens']}")
    print(f"  Avg target tokens       : {corpus_stats['avg_target_tokens']}")
    print(f"  TTR source (EN)         : {corpus_stats['ttr_source']}")
    print(f"  TTR target (OM)         : {corpus_stats['ttr_target']}")
    print(f"  Stats saved to          : corpus_stats.txt")

    print("\n" + "-"*51)
    print("PHASE 10 - FINAL DELIVERABLES")
    print(f"  cleaned_corpus    : {len(final_df)} pairs")
    print(f"  audit_log.csv     : {stats['REMOVE']} removed rows")
    print(f"  review_queue.csv  : generated")
    print(f"  corpus_stats.txt  : generated")
    print("="*51)

    print("\nTHESIS WARNINGS")
    print("  [1] langid removed -> Oromo heuristic classifier applied")
    print("  [2] Vocabulary imbalance EN:OM -> Expected for agglutinative morphology")
    print("  [3] Domain Bias -> Check if Bible data dominates corpus")
    print("  [4] Säleva & Lignos (2021) -> Address why Oromo differs from Nepali/Kazakh findings")
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

    output_dir  = os.path.join(BASE, "data", "output")
    output_path = os.path.join(output_dir, f"cleaned_corpus{ext}")
    audit_path  = os.path.join(output_dir, "audit_log.csv")
    review_path = os.path.join(output_dir, "review_queue.csv")

    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    pipeline = NLPPipeline(config_data.get("pipeline", []))
    try:
        final_df, ratio_df = pipeline.run_research_mode(
            analyzed_df, output_path, audit_path, review_path, format=selected_fmt
        )
        generate_report(stats, final_df, ratio_df, output_dir)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()