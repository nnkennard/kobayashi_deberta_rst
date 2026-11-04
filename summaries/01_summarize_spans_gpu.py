"""Summarize RST-DT spans on a GPU server, without Ollama.

Runs the model in-process with Hugging Face transformers. Needs its own
environment, because the versions pinned in ../requirements.txt are too old:

    pip install torch transformers accelerate tqdm
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import pipeline

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

MIN_WORDS_EXCLUSIVE = 10
MAX_NEW_TOKENS = 64

CHECKPOINT_EVERY = 50
MAX_CONSECUTIVE_FAILURES = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="span_map.json")
    parser.add_argument(
        "--output",
        default="summaries_gpu.json",
        help="Also the checkpoint: spans already in this file are skipped.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Only process the first N documents, for a test run.",
    )
    return parser.parse_args()


def load_model(model_name):
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to torch; check the job's GPU allocation")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=dtype,
        device_map="auto",
    )


def summarize(pipe, text):
    prompt = (
        "Summarize the following text into a single sentence of 20 words or "
        "less. Respond with only the summary sentence.\n\n"
        f"Text: {text}"
    )
    out = pipe(
        [{"role": "user", "content": prompt}],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        # Unset the model's default sampling params, which greedy decoding ignores.
        temperature=None,
        top_p=None,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    return out[0]["generated_text"][-1]["content"].strip()


def load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_checkpoint(summaries, path):
    # Write then rename, so a crash mid-write can't corrupt the checkpoint.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(summaries, f, indent=1)
    os.replace(tmp_path, path)


def collect_todo(span_map, summaries, max_docs):
    todo = []
    for doc_index, (doc_id, edu_spans) in enumerate(span_map.items()):
        if max_docs is not None and doc_index >= max_docs:
            break
        for key, span_texts in edu_spans.items():
            text = " ".join(span_texts)
            if (
                len(text.split()) > MIN_WORDS_EXCLUSIVE
                and key not in summaries.get(doc_id, {})
            ):
                todo.append((doc_id, key, text))
    return todo


def main():
    args = parse_args()

    with open(args.input, "r") as f:
        span_map = json.load(f)

    summaries = load_checkpoint(args.output)
    todo = collect_todo(span_map, summaries, args.max_docs)
    pipe = load_model(args.model)

    # Failed spans stay out of the checkpoint, so the next run retries them.
    failures = 0
    consecutive_failures = 0
    try:
        progress = tqdm(todo, desc="Summarizing", unit="span", mininterval=30)
        for i, (doc_id, key, text) in enumerate(progress, 1):
            try:
                summary = summarize(pipe, text)
            except (RuntimeError, ValueError, KeyError, IndexError) as e:
                # RuntimeError includes CUDA out-of-memory errors.
                torch.cuda.empty_cache()
                failures += 1
                consecutive_failures += 1
                progress.set_postfix(failed=failures)
                tqdm.write(f"FAILED {doc_id} {key}: {e!r}", file=sys.stderr)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    sys.exit(
                        f"Stopping after {consecutive_failures} consecutive failures"
                    )
                continue
            consecutive_failures = 0
            summaries.setdefault(doc_id, {})[key] = summary
            if i % CHECKPOINT_EVERY == 0:
                save_checkpoint(summaries, args.output)
    finally:
        save_checkpoint(summaries, args.output)

    print(f"Finished: {failures} failed, rerun to retry them", flush=True)


if __name__ == "__main__":
    main()
