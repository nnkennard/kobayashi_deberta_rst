import collections
import json

TRAIN, TEST, VALID = "train test valid".split()
SPLITS = [TRAIN, TEST, VALID]

EDU_DECODE_RE = "/([0-9]+)\-([0-9]+)"

def summarize(text):
    return len(text)

def main():
    with open('span_map.json', 'r') as f:
        m = json.load(f)

    summaries = collections.defaultdict(dict)
    for doc_id, edu_spans in m.items():
        for key, span_texts in edu_spans.items():
            text = " ".join(span_texts)
            if len(text.split()) > 10:
                summaries[doc_id][key] = summarize(text)

    print(summaries)

if __name__ == "__main__":
    main()
