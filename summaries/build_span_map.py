#!/usr/bin/env python3
"""Map every EDU span in the RST trees of a split to the EDU strings it covers.

Output structure:

    {doc_id: {"start-end": [edu_text, ...]}}

EDU numbers are 0-based and restart for each document, so spans are grouped by
document. Both ends of a span are inclusive: "2-4" covers EDUs 2, 3 and 4, and
"3-3" is the single EDU 3. Each distinct span appears once, however many tree
nodes cover it, and spans are ordered by start, then end.
"""

import argparse
import json
import re
from pathlib import Path

from build_edu_map import build_doc_map

SCRIPT_DIR = Path(__file__).resolve().parent
TREE_TOKEN = re.compile(r"\(text (\d+)\)|\(|\)|[^\s()]+")


def collect_spans(rst_tree):
    """Return the set of (start, end) EDU spans of all nodes in a tree string."""
    spans = set()
    stack = []  # (start, end) of the children seen so far, per open node

    def add_child(start, end):
        spans.add((start, end))
        if stack:
            if stack[-1] is None:
                stack[-1] = (start, end)
            elif stack[-1][1] + 1 == start:
                stack[-1] = (stack[-1][0], end)
            else:
                raise ValueError(f"non-contiguous children at EDU {start}")

    for match in TREE_TOKEN.finditer(rst_tree):
        token = match.group()
        if match.group(1) is not None:
            edu = int(match.group(1))
            add_child(edu, edu)
        elif token == "(":
            stack.append(None)
        elif token == ")":
            add_child(*stack.pop())
    if stack:
        raise ValueError("unbalanced parentheses in rst_tree")
    return spans


def build_span_map(docs):
    """Return {doc_id: {"start-end": [edu_text, ...]}} for a list of documents."""
    span_map = {}
    for doc in docs:
        if doc["doc_id"] in span_map:
            raise ValueError(f"duplicate doc_id {doc['doc_id']}")
        edus = build_doc_map(doc)
        span_map[doc["doc_id"]] = {
            f"{start}-{end}": edus[start : end + 1]
            for start, end in sorted(collect_spans(doc["rst_tree"]))
        }
    return span_map


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR.parent / "data" / "rstdt" / "train.json",
        help="split file to read",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "span_map.json",
        help="path of the JSON file to write",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        docs = json.load(f)
    span_map = build_span_map(docs)

    num_spans = sum(len(spans) for spans in span_map.values())
    print(f"{len(span_map)} docs, {num_spans} distinct spans")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(span_map, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
