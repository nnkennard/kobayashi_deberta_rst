#!/usr/bin/env python3
"""Map every EDU span in the RST trees of a split to the EDU strings it covers.

Output structure:

    {doc_id: {"start-end": [edu_text, ...]}}

EDU numbers are 0-based and restart for each document, so spans are grouped by
document. Both ends of a span are inclusive: "2-4" covers EDUs 2, 3 and 4, and
"3-3" is the single EDU 3.
"""

import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="../data/rstdt/train.json",
    help="split file to read",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="span_map.json",
    help="path of the JSON file to write",
)

TREE_TOKEN = re.compile(r"\(text (\d+)\)|\(|\)|[^\s()]+")
TREE_LEAF = re.compile(r"\(text (\d+)\)")

RELATION_TABLE = {
    "ROOT": "ROOT",
    "span": "span",
    "attribution": "Attribution",
    "attribution-negative": "Attribution",
    "background": "Background",
    "circumstance": "Background",
    "cause": "Cause",
    "result": "Cause",
    "cause-result": "Cause",
    "consequence": "Cause",
    "comparison": "Comparison",
    "preference": "Comparison",
    "analogy": "Comparison",
    "proportion": "Comparison",
    "condition": "Condition",
    "hypothetical": "Condition",
    "contingency": "Condition",
    "otherwise": "Condition",
    "contrast": "Contrast",
    "concession": "Contrast",
    "antithesis": "Contrast",
    "elaboration-additional": "Elaboration",
    "elaboration-general-specific": "Elaboration",
    "elaboration-part-whole": "Elaboration",
    "elaboration-process-step": "Elaboration",
    "elaboration-object-attribute": "Elaboration",
    "elaboration-set-member": "Elaboration",
    "example": "Elaboration",
    "definition": "Elaboration",
    "enablement": "Enablement",
    "purpose": "Enablement",
    "evaluation": "Evaluation",
    "interpretation": "Evaluation",
    "conclusion": "Evaluation",
    "comment": "Evaluation",
    "evidence": "Explanation",
    "explanation-argumentative": "Explanation",
    "reason": "Explanation",
    "list": "Joint",
    "disjunction": "Joint",
    "manner": "Manner-Means",
    "means": "Manner-Means",
    "problem-solution": "Topic-Comment",
    "question-answer": "Topic-Comment",
    "statement-response": "Topic-Comment",
    "topic-comment": "Topic-Comment",
    "comment-topic": "Topic-Comment",
    "rhetorical-question": "Topic-Comment",
    "summary": "Summary",
    }


def build_doc_map(doc):
    """Return the list of EDU texts for a document, checking its numbering."""
    edus = doc["edu_strings"]
    expected = list(range(len(edus)))

    start_numbers = [start[2] for start in doc["edu_start_indices"]]
    # TODO: Figure out what edu_start_indices is
    if start_numbers != expected:
        raise ValueError(
            f"{doc['doc_id']}: edu_start_indices disagree with edu_strings")

    # TODO: Figure out if this is needed at all
    tree_numbers = [int(n) for n in TREE_LEAF.findall(doc["rst_tree"])]
    if tree_numbers != expected:
        raise ValueError(
            f"{doc['doc_id']}: rst_tree leaves disagree with edu_strings")

    return edus


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
            f"{start}-{end}": edus[start:end + 1]
            for start, end in sorted(collect_spans(doc["rst_tree"]))
        }
    return span_map


def main():
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
