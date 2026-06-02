# 🌍 Afaan Oromoo NMT Corpus Firewall

> A data cleaning pipeline for English ↔ Oromo neural machine translation

Afaan Oromoo is spoken by over 40 million people, yet it remains severely underrepresented in NLP research. Most publicly available parallel corpora for the language are noisy — riddled with script leaks, misaligned translations, and contamination from other Ethiopian languages like Amharic.

This pipeline exists to fix that. It runs your raw parallel corpus through 10 targeted cleaning phases and outputs training data you can actually trust.

---

## Why this exists

Training an NMT model on dirty data doesn't just hurt performance — it quietly poisons it. A model that's seen Amharic characters in its Oromo training data will learn the wrong things. One trained on length-mismatched pairs will hallucinate. This pipeline catches those problems before they reach your model.

Think of it less as a cleaning script and more as a quality gate: data goes in, only the pairs that genuinely belong come out.

---

## What it checks

| Check | What it catches |
|---|---|
| **Bilingual pair validation** | Pairs where both sides are the same language, or the languages are swapped |
| **Script guard** | Ethiopic (Amharic) or Arabic characters that leaked into the Oromo column |
| **Length ratio filtering** | Translations that are suspiciously short or long relative to the source — a common sign of truncation or hallucination |
| **Qubee normalization** | Inconsistent apostrophe usage for glottal stops, which inflates vocabulary size for no reason |
| **Audit trail** | Every removed row gets a tagged reason, so you can explain your data decisions in a thesis or paper |

---

## Getting started

**Requirements:** Python 3.10+

```bash
pip install polars pyyaml langid
```

**Project layout:**

```
nlp_data_pipeline/
├── config/
│   └── pipeline_config.yaml
├── data/
│   └── output/
├── src/
│   ├── analyzer.py
│   ├── pipeline.py
│   ├── base.py
│   └── transformers/
│       ├── cleaning.py
│       ├── filtering.py
│       ├── linguistic.py
│       ├── tokenization.py
│       └── validation.py
└── main.py
```

---

## Running the pipeline

**Step 1 — Tune your settings**

Open `nlp_data_pipeline/config/pipeline_config.yaml` and adjust the thresholds for your corpus (length ratio bounds, which languages to allow, etc.).

**Step 2 — Run it**

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/nlp_data_pipeline
python nlp_data_pipeline/main.py
```

**Step 3 — Review the results**

The script will:
1. Ask for your raw `corpus_cleaned.csv`
2. Run the full analysis and print a quality report
3. Let you review flagged categories (e.g., how much `EN_SW` contamination was found)
4. Export in your chosen format — `jsonl` for training, `csv` for analysis

---

## The 10 phases

| # | Phase | What happens |
|---|---|---|
| 1 | Structural check | Drops null and empty rows |
| 2 | Language identification | Confirms each side of the pair is actually the right language |
| 3 | Script check | Scans for Ethiopic or Arabic characters in Oromo text |
| 4 | Signal verification | Cross-checks `langid` results against Oromo-specific heuristics, since automatic language ID often fails on low-resource languages |
| 5 | Orthography | Normalizes Qubee glottal stop markers |
| 6 | Syntactic order | Checks for SOV word order; SVO constructions often indicate a bad translation |
| 7 | Semantic alignment | Prunes pairs where the length ratio falls outside expected bounds |
| 8 | Token quality | Strips HTML tags, noise characters, and web-scraping artifacts |
| 9 | Deduplication | Removes exact and near-duplicate pairs |
| 10 | Export | Writes the final cleaned corpus, audit log, and review queue |

---

## Output files

After a run, you'll find three files in `data/output/`:

- **`cleaned_corpus.jsonl`** — the filtered, normalized training set
- **`audit_log.csv`** — every removed row with the reason it was flagged; useful for the data section of a thesis or paper
- **`review_queue.csv`** — pairs that were borderline; worth a human look before deciding to keep or discard

---

## License

See [LICENSE](LICENSE) for details.
