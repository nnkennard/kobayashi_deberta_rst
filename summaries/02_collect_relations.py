#!/usr/bin/env python3
"""Gather the text and summary of both children of every relation.

One JSON file is written, with a record per relation across all documents:

    [{"doc_id": ..., "relation": ..., "nuclearity": ...,
      "left_span": "start-end", "left_text": ..., "left_summary": ...,
      "right_span": "start-end", "right_text": ..., "right_summary": ...}, ...]

EDU numbers in spans are 0-based, restart for each document and are inclusive at
both ends. A summary is the empty string if its span has no entry in the
summaries file.

A child's text is built from its single-EDU entries in the span map.
"""

import argparse
import json


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def span_text(edu_spans, span):
    """Return the text of a "start-end" span, with its EDUs joined by spaces."""
    start, end = (int(number) for number in span.split("-"))
    return " ".join(edu_spans[f"{i}-{i}"][0] for i in range(start, end + 1))


def build_records(relations, span_map, summaries):
    """Return the list of relation records for all documents."""
    records = []
    for doc_id, doc_relations in relations.items():
        edu_spans = span_map[doc_id]
        doc_summaries = summaries.get(doc_id, {})
        for relation in doc_relations:
            left, right = relation["left_span"], relation["right_span"]
            records.append({
                "doc_id": doc_id,
                "relation": relation["relation"],
                "left_summary": doc_summaries.get(left, ""),
                "right_summary": doc_summaries.get(right, ""),
                "nuclearity": relation["nuclearity"],
                "left_span": left,
                "left_text": span_text(edu_spans, left),
                "right_span": right,
                "right_text": span_text(edu_spans, right),
            })
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-r", "--relations", default="relations.json",
                        help="relations file to read")
    parser.add_argument("-m", "--span-map", default="span_map.json",
                        help="span map file to read")
    parser.add_argument("-s", "--summaries", default="summaries.json",
                        help="summaries file to read")
    parser.add_argument("-o", "--output", default="relation_records.json",
                        help="file to write")
    return parser.parse_args()


def main():
    args = parse_args()
    records = build_records(load_json(args.relations),
                            load_json(args.span_map),
                            load_json(args.summaries))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
