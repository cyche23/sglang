#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import requests


def make_prompt(common_len: int, tail_len: int, tail_name: str) -> str:
    common_words = [f"shared_partial_l3_prefix_{i % 997}" for i in range(common_len)]
    tail_words = [f"{tail_name}_partial_l3_tail_{i % 997}" for i in range(tail_len)]
    return (
        "You are validating radix prefix reuse. Keep the answer short.\n\n"
        + " ".join(common_words)
        + "\n\n"
        + " ".join(tail_words)
        + "\n\nSummarize the final tail marker in one sentence."
    )


def send_generate(
    base_url: str,
    prompt: str,
    max_new_tokens: int,
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    start = time.perf_counter()
    response = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
    latency = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        data = data[0]
    meta = data.get("meta_info", {})
    return {
        "latency_s": latency,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "cached_tokens": meta.get("cached_tokens", 0),
        "finish_reason": meta.get("finish_reason"),
        "output_nonempty": bool(data.get("text", "")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exercise UnifiedRadixCache partial L3 split with A/B/A prompts."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--common-len", type=int, default=768)
    parser.add_argument("--tail-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    requests.get(f"{base_url}/health", timeout=10).raise_for_status()

    prompt_a = make_prompt(args.common_len, args.tail_len, "old")
    prompt_b = make_prompt(args.common_len, args.tail_len, "new")

    first_a = send_generate(base_url, prompt_a, args.max_new_tokens, args.timeout)
    partial_b = send_generate(base_url, prompt_b, args.max_new_tokens, args.timeout)
    second_a = send_generate(base_url, prompt_a, args.max_new_tokens, args.timeout)

    result = {
        "base_url": base_url,
        "common_len_words_requested": args.common_len,
        "tail_len_words_requested": args.tail_len,
        "max_new_tokens": args.max_new_tokens,
        "first_a_populate_old_tail": first_a,
        "partial_b_common_prefix_new_tail": partial_b,
        "second_a_old_tail_again": second_a,
        "expected_server_log_signals": [
            "UnifiedRadixCache partial-l3-split",
            "UnifiedRadixCache L3 hit",
            "UnifiedRadixCache L3 read",
            "UnifiedRadixCache L3 restore",
        ],
        "failure_log_signal": "partial-l3-insert-miss",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
