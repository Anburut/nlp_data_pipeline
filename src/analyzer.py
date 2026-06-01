import polars as pl
import re
from typing import Dict, Tuple


class DatasetAnalyzer:

    OROMO_WORDS = {
        "fi", "kan", "itti", "akka", "nama", "hin", "ta'u", "baay'ee",
        "miti", "irraa", "keessa", "keessaa", "rraa", "dhaan", "tti",
        "dha", "ture", "jira", "bara", "waan", "ta'e", "ni", "oo",
        "irratti", "waliin", "garuu", "yeroo", "gara", "kana",
        "isaa", "isaaf", "isaan", "yoo", "yihowaa", "waaqayyoo", "namni",
        "jedha", "jedhe", "jetee", "moo", "maali", "maaf", "akkam",
        "danda'a", "danda'u", "qaba", "qabna", "tahe", "tahee"
    }

    OROMO_SUFFIXES = [
        "dhaan", "rraa", "tti", "irratti", "keessa", "keessaa",
        "dha", "ture", "jira", "jedha", "jedhe", "danda'a",
        "tahe", "tahee", "moo", "oof", "aan", "een", "oon",
        "uun", "irraa", "irra", "rratti"
    ]

    ENGLISH_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "not",
        "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "that", "this", "it", "he", "she", "they", "we",
        "you", "i", "my", "his", "her", "our", "your", "their",
        "why", "what", "when", "where", "how", "who", "if", "let",
        "god", "jehovah", "work", "read", "take", "right", "however"
    }

    def __init__(self):
        self.ethiopic_pattern = re.compile(r'[\u1200-\u137F]')
        self.noise_pattern = re.compile(r'<[^>]*>|&[a-z]+;|https?://\S+')
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        self.cyrillic_pattern = re.compile(r'[\u0400-\u04FF]')

    def _is_english(self, text: str) -> bool:
        if not text or len(text.strip()) == 0:
            return False
        words = set(text.lower().split())
        hits = len(words & self.ENGLISH_WORDS)
        return hits >= 1

    def _is_oromo(self, text: str) -> bool:
        if not text or len(text.strip()) < 3:
            return False
        if self.ethiopic_pattern.search(text):
            return False
        if self.arabic_pattern.search(text):
            return False
        if self.cyrillic_pattern.search(text):
            return False
        text_lower = text.lower()
        words = set(text_lower.split())
        word_hits = len(words & self.OROMO_WORDS)
        suffix_hits = sum(1 for s in self.OROMO_SUFFIXES if s in text_lower)
        return (word_hits >= 1) or (suffix_hits >= 1)

    def classify_language(self, text: str, side: str = "target") -> str:
        if not text or len(text.strip()) == 0:
            return "empty"
        if self.ethiopic_pattern.search(text):
            return "amharic"
        if self.noise_pattern.search(text):
            return "noise"
        if side == "target":
            if self._is_oromo(text):
                return "oromo"
            if self._is_english(text):
                return "en"
            return "unknown"
        else:
            if self._is_english(text):
                return "en"
            if self._is_oromo(text):
                return "oromo"
            return "unknown"

    def analyze(self, input_path: str) -> Tuple[pl.DataFrame, Dict]:
        import csv
        with open(input_path, "r", encoding="utf-8") as f:
            f.readline()
            lines = f.readlines()

        reader = csv.reader(lines, delimiter=',')
        results = []
        stats = {
            "total": 0, "KEEP": 0, "REMOVE": 0, "REVIEW": 0,
            "P1_EMPTY": 0, "P1_ENCODING": 0, "P1_ALIGN": 0, "P1_DUP": 0,
            "P2_EN_EN": 0, "P2_OM_OM": 0, "P2_OM_EN": 0, "P2_XX_OM": 0,
            "P2_EN_XX": 0, "P2_XX_XX": 0, "P2_SCRIPT": 0, "P2_NOISE": 0,
            "P2_IDENTICAL": 0, "valid": 0
        }

        for row in reader:
            stats["total"] += 1
            if len(row) < 2 or not row[0] or not row[1]:
                status, src, tgt, reason = "REMOVE", row[0] if row else "", "", "P1_EMPTY"
                stats["P1_EMPTY"] += 1
            else:
                src = row[0].strip()
                tgt = ",".join(row[1:]).strip()
                src_lang = self.classify_language(src, side="source")
                tgt_lang = self.classify_language(tgt, side="target")

                if src_lang == "en" and tgt_lang == "oromo":
                    if self.ethiopic_pattern.search(tgt):
                        status, reason = "REMOVE", "P2_SCRIPT"
                        stats["P2_SCRIPT"] += 1
                    elif self.noise_pattern.search(tgt):
                        status, reason = "REMOVE", "P2_NOISE"
                        stats["P2_NOISE"] += 1
                    elif src.lower() == tgt.lower():
                        status, reason = "REMOVE", "P2_IDENTICAL"
                        stats["P2_IDENTICAL"] += 1
                    else:
                        status, reason = "KEEP", "VALID"
                        stats["valid"] += 1
                elif src_lang == "en" and tgt_lang == "en":
                    status, reason = "REMOVE", "P2_EN_EN"
                    stats["P2_EN_EN"] += 1
                elif src_lang == "oromo" and tgt_lang == "oromo":
                    status, reason = "REMOVE", "P2_OM_OM"
                    stats["P2_OM_OM"] += 1
                elif src_lang == "oromo" and tgt_lang == "en":
                    status, reason = "REMOVE", "P2_OM_EN"
                    stats["P2_OM_EN"] += 1
                elif src_lang != "en" and tgt_lang == "oromo":
                    status, reason = "REMOVE", "P2_XX_OM"
                    stats["P2_XX_OM"] += 1
                elif src_lang == "en" and tgt_lang != "oromo":
                    status, reason = "REMOVE", "P2_EN_XX"
                    stats["P2_EN_XX"] += 1
                else:
                    status, reason = "REMOVE", "P2_XX_XX"
                    stats["P2_XX_XX"] += 1

            stats[status] += 1
            results.append({
                "Source": src,
                "Target": tgt,
                "Decision": status,
                "Reason": reason
            })

        return pl.DataFrame(results), stats