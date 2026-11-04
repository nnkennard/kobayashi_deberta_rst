#!/usr/bin/env python3
"""Extract the EDU spans and relations of the RST trees of a split.

Two JSON files are written. The span map gives the EDU strings each span covers:

    {doc_id: {"start-end": [edu_text, ...]}}

The relations file gives every relation and the EDU spans of its two subtrees:

    {doc_id: [{"relation": ..., "nuclearity": ...,
               "left_span": "start-end", "right_span": "start-end"}, ...]}

EDU numbers are 0-based and restart for each document, so spans are grouped by
document. Both ends of a span are inclusive: "2-4" covers EDUs 2, 3 and 4, and
"3-3" is the single EDU 3.

There is one relation per internal node of the binarized tree, in document order
and top-down. Trees are binarized and relations labelled as in src/data/tree.py,
and relations use the coarse-grained classes of RELATION_TABLE in
src/data/rstdt_relation.py.
"""

import argparse
import json
import re
from dataclasses import dataclass
from typing import Optional

TREE_TOKEN = re.compile(r"\(text (\d+)\)|\(|\)|[^\s()]+")
TREE_LEAF = re.compile(r"\(text (\d+)\)")

# Copy of RELATION_TABLE in src/data/rstdt_relation.py. That module can't be
# imported here without the parser's dependencies (nltk, torchtext).
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
    "restatement": "Summary",
    "temporal-before": "Temporal",
    "temporal-after": "Temporal",
    "temporal-same-time": "Temporal",
    "sequence": "Temporal",
    "inverted-sequence": "Temporal",
    "topic-shift": "Topic-Change",
    "topic-drift": "Topic-Change",
    "textualorganization": "Textual-organization",
    "same-unit": "Same-unit",
}


@dataclass
class Node:
    """An internal node of an RST tree. Leaves are plain EDU numbers (ints)."""

    label: Optional[str]
    children: list


def parse_tree(rst_tree):
    """Parse a tree string into its root Node."""
    root = None
    stack = []
    for match in TREE_TOKEN.finditer(rst_tree):
        token = match.group()
        if match.group(1) is not None:
            stack[-1].children.append(int(match.group(1)))
        elif token == "(":
            stack.append(Node(None, []))
        elif token == ")":
            node = stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                root = node
        else:
            stack[-1].label = token
    if stack or root is None:
        raise ValueError("unbalanced parentheses in rst_tree")
    return root


def edu_span(node):
    """Return the (start, end) EDU numbers covered by a node or leaf."""
    first = last = node
    while isinstance(first, Node):
        first = first.children[0]
    while isinstance(last, Node):
        last = last.children[-1]
    return first, last


def span_key(start, end):
    """Return the "start-end" key used for a span in the output files."""
    return f"{start}-{end}"


def collect_spans(tree):
    """Return the set of (start, end) EDU spans of all nodes in a parsed tree."""
    spans = set()

    def visit(node):
        if isinstance(node, Node):
            child_spans = [visit(child) for child in node.children]
            for (_, previous_end), (start, _) in zip(child_spans, child_spans[1:]):
                if previous_end + 1 != start:
                    raise ValueError(f"non-contiguous children at EDU {start}")
            span = (child_spans[0][0], child_spans[-1][1])
        else:
            span = (node, node)
        spans.add(span)
        return span

    visit(tree)
    return spans


def coarse_relation(relation):
    """Map a fine-grained RST-DT relation to its coarse class, as in re_categorize."""
    while relation[-2:] in ["-s", "-e", "-n"]:
        relation = relation[:-2]
    return RELATION_TABLE[relation]


def split_children(children):
    """Return the (left, right) subtrees of a node, binarizing it if needed."""
    if len(children) == 2:
        return children

    labels = [child.label for child in children]
    if all(label == labels[0] for label in labels):
        # Multinuclear relation: right-heavy.
        return children[0], Node(labels[0], children[1:])
    if len(children) == 3 and labels[1] == "nucleus:span":
        # Satellite, nucleus, satellite: left-heavy.
        return Node("nucleus:span", children[:2]), children[2]
    raise ValueError(f"cannot binarize node with children {labels}")


def iter_relations(node):
    """Yield a relation dict for each internal node of a parsed tree, top-down."""
    if len(node.children) == 1:
        if isinstance(node.children[0], Node):
            raise ValueError(f"non-EDU only child under {node.label}")
        return

    left, right = split_children(node.children)
    left_nuclearity, left_relation = left.label.split(":", maxsplit=1)
    right_nuclearity, right_relation = right.label.split(":", maxsplit=1)
    relation = left_relation if left_relation != "span" else right_relation
    yield {
        "relation": coarse_relation(relation),
        "nuclearity": f"{left_nuclearity}-{right_nuclearity}",
        "left_span": span_key(*edu_span(left)),
        "right_span": span_key(*edu_span(right)),
    }
    yield from iter_relations(left)
    yield from iter_relations(right)


def get_edus(doc):
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


def extract(docs):
    """Return the span map and relations of a list of documents, keyed by doc_id."""
    span_map = {}
    relations = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        if doc_id in span_map:
            raise ValueError(f"duplicate doc_id {doc_id}")
        edus = get_edus(doc)
        tree = parse_tree(doc["rst_tree"])
        span_map[doc_id] = {
            span_key(start, end): edus[start:end + 1]
            for start, end in sorted(collect_spans(tree))
        }
        relations[doc_id] = list(iter_relations(tree))
    return span_map, relations


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        help="path of the span map JSON file to write",
    )
    parser.add_argument(
        "-r",
        "--relations-output",
        type=str,
        default="relations.json",
        help="path of the relations JSON file to write",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, encoding="utf-8") as f:
        docs = json.load(f)
    span_map, relations = extract(docs)

    num_spans = sum(len(spans) for spans in span_map.values())
    num_relations = sum(len(doc_relations) for doc_relations in relations.values())
    print(f"{len(span_map)} docs, {num_spans} distinct spans, {num_relations} relations")

    write_json(span_map, args.output)
    write_json(relations, args.relations_output)


if __name__ == "__main__":
    main()
