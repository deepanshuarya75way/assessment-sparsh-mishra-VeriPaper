#!/usr/bin/env python
"""Build the VeriPaper open corpus for plagiarism search and model training.

Downloads real arXiv abstracts via the arXiv API (free, no key) and writes:
- data/corpus.jsonl          — plagiarism-search corpus
- data/ai_training.csv       — human (real abstracts) vs AI-variant rows
"""
import csv
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

random.seed(42)

ARXIV_QUERY = (
    "cat:cs.CL+OR+cat:cs.AI+OR+cat:cs.LG+OR+cat:stat.ML"
)


def fetch_arxiv_abstracts(target: int = 2500, batch: int = 100) -> list:
    """Fetch real abstracts from arXiv (human-written academic text)."""
    abstracts = []
    start = 0
    while len(abstracts) < target and start < 12000:
        params = {
            "search_query": ARXIV_QUERY,
            "start": str(start),
            "max_results": str(batch),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        # Build the URL manually: urlencode double-escapes the literal '+'
        # operators in the search_query, which arXiv interprets differently
        q = urllib.parse.quote(params["search_query"], safe="+")
        url = (f"https://export.arxiv.org/api/query?search_query={q}"
               f"&start={params['start']}&max_results={params['max_results']}"
               f"&sortBy=submittedDate&sortOrder=descending")
        req = urllib.request.Request(url, headers={"User-Agent": "VeriPaper/1.0 (corpus research; contact sparsh@example.com)"})
        xml = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    xml = resp.read().decode("utf-8", errors="ignore")
                if "<entry>" in xml:
                    break
            except Exception as exc:  # pragma: no cover
                print(f"  fetch error at start={start} attempt={attempt}: {exc}")
            time.sleep(4)
        if not xml or "<entry>" not in xml:
            print(f"  no entries at start={start}, skipping")
            start += batch
            continue

        # Parse with proper XML parsing (namespaced Atom)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        got = 0
        for entry in root.findall("a:entry", ns):
            entry_id = (entry.findtext("a:id", "", ns) or "").strip()
            title = re.sub(r"\s+", " ", entry.findtext("a:title", "", ns)).strip()
            summary = re.sub(r"\s+", " ", entry.findtext("a:summary", "", ns)).strip()
            if not (entry_id and title and summary):
                continue
            abstracts.append({"id": entry_id, "title": title, "text": summary})
            got += 1
        print(f"  start={start}: got {got} entries (total {len(abstracts)})")
        start += batch
        if got == 0:
            time.sleep(3)
    return abstracts[:target]


def make_ai_variants(texts: list) -> list:
    """Create AI-style variants of human abstracts via systematic rewriting.

    LLM paraphrases are the most common 'AI writing' signature; we simulate the
    common patterns (formulaic openers, hedging stacks, filler adjectives) so
    the detector learns them. This is a controlled, reproducible proxy.
    """
    fillers = [
        "In this comprehensive study, we delve into",
        "It is important to note that",
        "Furthermore, our extensive experiments demonstrate",
        "Moreover, the proposed framework significantly enhances",
        "In the rapidly evolving landscape of",
        "Leveraging state-of-the-art techniques, we propose",
        "Our findings reveal a fascinating tapestry of",
        "Through rigorous experimentation and thorough analysis,",
        "Notably, the results underscore the critical importance of",
        "Embarking on this investigation, we present",
    ]
    variants = []
    for text in texts:
        kind = random.choice(["opener", "hedge", "adjective"])
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if kind == "opener" and sentences:
            opener = random.choice(fillers)
            rest = " ".join(sentences[1:])
            variants.append(f"{opener} {rest}")
        elif kind == "hedge":
            hedge_map = {
                "shows": "appears to show",
                "improves": "appears to improve",
                "achieves": "potentially achieves",
                "demonstrates": "may demonstrate",
                "increases": "tends to increase",
            }
            modified = text
            for k, v in hedge_map.items():
                modified = re.sub(rf"\b{k}\b", v, modified)
            if modified != text:
                variants.append(modified)
            else:
                variants.append(random.choice(fillers) + " " + text)
        else:  # adjective inflation
            adj = ["robust", "novel", "extensive", "comprehensive", "significant"]
            modified = re.sub(r"\b(experiments|results|framework|method|approach)\b",
                              lambda m: random.choice(adj) + " " + m.group(1), text, count=3)
            variants.append(modified)
    return variants


def main():
    print("Fetching arXiv abstracts...")
    abstracts = fetch_arxiv_abstracts(target=2500)
    print(f"Fetched {len(abstracts)} abstracts")

    # Corpus for plagiarism search (dedupe by id)
    seen = set()
    corpus = []
    for a in abstracts:
        if a["id"] in seen or len(a["text"].split()) < 30:
            continue
        seen.add(a["id"])
        corpus.append({"title": a["title"], "text": a["text"], "corpus": "arxiv"})
    with open(DATA / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for item in corpus:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus)} corpus entries")

    # Training data: human (1) vs AI-variant (0)
    n_human = min(1800, len(corpus))
    human_rows = corpus[:n_human]
    ai_rows = make_ai_variants([c["text"] for c in corpus[:n_human]])
    assert len(human_rows) == len(ai_rows)
    with open(DATA / "ai_training.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        for row in human_rows:
            writer.writerow([row["text"], 1])
        for text in ai_rows:
            writer.writerow([text, 0])
    print(f"Wrote {len(human_rows)} human + {len(ai_rows)} AI-variant rows")


if __name__ == "__main__":
    main()
