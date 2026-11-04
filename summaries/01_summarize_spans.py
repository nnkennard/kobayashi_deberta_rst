import json
import os
import sys

import requests
from tqdm import tqdm

TRAIN, TEST, VALID = "train test valid".split()
SPLITS = [TRAIN, TEST, VALID]

EDU_DECODE_RE = r"/([0-9]+)\-([0-9]+)"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3.2"

CHECKPOINT_PATH = "summaries.json"
CHECKPOINT_EVERY = 50
MAX_CONSECUTIVE_FAILURES = 10

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

def load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_checkpoint(summaries, path):
    # Write then rename, so a crash mid-write can't corrupt the checkpoint.
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(summaries, f, indent=1)
    os.replace(tmp_path, path)

def main():
    with open('span_map.json', 'r') as f:
        m = json.load(f)

    summaries = load_checkpoint(CHECKPOINT_PATH)

    todo = []
    doc_count = 0
    for doc_id, edu_spans in m.items():
        for key, span_texts in edu_spans.items():
            text = " ".join(span_texts)
            if len(text.split()) > 10 and key not in summaries.get(doc_id, {}):
                todo.append((doc_id, key, text))
        doc_count += 1
        if doc_count == 20:
            break

    # Failed spans stay out of the checkpoint, so the next run retries them.
    failures = 0
    consecutive_failures = 0
    try:
        progress = tqdm(todo, desc="Summarizing", unit="span")
        for i, (doc_id, key, text) in enumerate(progress, 1):
            try:
                summary = summarize(text)
            except (requests.RequestException, KeyError) as e:
                failures += 1
                consecutive_failures += 1
                progress.set_postfix(failed=failures)
                tqdm.write(f"FAILED {doc_id} {key}: {e!r}", file=sys.stderr)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    sys.exit(f"Stopping after {consecutive_failures} consecutive failures")
                continue
            consecutive_failures = 0
            summaries.setdefault(doc_id, {})[key] = summary
            if i % CHECKPOINT_EVERY == 0:
                save_checkpoint(summaries, CHECKPOINT_PATH)
    finally:
        save_checkpoint(summaries, CHECKPOINT_PATH)

    print(f"Finished: {failures} failed, rerun to retry them", flush=True)

if __name__ == "__main__":
    main()
