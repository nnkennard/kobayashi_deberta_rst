import collections
import json

import requests

TRAIN, TEST, VALID = "train test valid".split()
SPLITS = [TRAIN, TEST, VALID]

EDU_DECODE_RE = "/([0-9]+)\-([0-9]+)"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3.2"

def summarize(text):
    prompt = (
        "Summarize the following text into a single sentence of 20 words or "
        "less. Respond with only the summary sentence.\n\n"
        f"Text: {text}"
    )
    response = requests.post(
        OLLAMA_URL,
        json={"model": LLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

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
