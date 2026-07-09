#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import requests


def make_prompt(prompt_len: int, variant: str) -> str:
    words = [
        f"{variant}_unified_radix_cache_{i % 997}"
        for i in range(max(prompt_len, 1))
    ]
    return (
        "You are evaluating a KV cache reuse baseline. "
        "Keep the answer concise.\n\n"
        + " ".join(words)
        + "\n\nSummarize the repeated context in one sentence."
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
    text = data.get("text", "")
    return {
        "latency_s": latency,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "cached_tokens": meta.get("cached_tokens", 0),
        "output_nonempty": bool(text),
        "finish_reason": meta.get("finish_reason"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce a UnifiedRadixCache L3 write/restore case."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2 to observe restore.")

    base_url = f"http://{args.host}:{args.port}"
    requests.get(f"{base_url}/health", timeout=10).raise_for_status()

    recompute_prompt = make_prompt(args.prompt_len, "recompute")
    reuse_prompt = make_prompt(args.prompt_len, "reuse")

    recompute = send_generate(
        base_url, recompute_prompt, args.max_new_tokens, args.timeout
    )
    populate = send_generate(base_url, reuse_prompt, args.max_new_tokens, args.timeout)
    restore_runs = [
        send_generate(base_url, reuse_prompt, args.max_new_tokens, args.timeout)
        for _ in range(args.repeat - 1)
    ]

    restore_latencies = [item["latency_s"] for item in restore_runs]
    best_restore = min(restore_latencies) if restore_latencies else None
    ratio = (
        best_restore / recompute["latency_s"]
        if best_restore is not None and recompute["latency_s"] > 0
        else None
    )

    result = {
        "base_url": base_url,
        "prompt_len_words_requested": args.prompt_len,
        "repeat": args.repeat,
        "max_new_tokens": args.max_new_tokens,
        "recompute": recompute,
        "populate": populate,
        "restore_runs": restore_runs,
        "best_restore_latency_s": best_restore,
        "restore_recompute_ratio": ratio,
        "observability_note": (
            "Read server logs for authoritative L3 write/hit/read/restore/eviction "
            "counters emitted by UnifiedRadixCache."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
