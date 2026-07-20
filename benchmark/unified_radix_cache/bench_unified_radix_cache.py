#!/usr/bin/env python3
"""Deterministic TTFT benchmark for the UnifiedRadixCache L3 path.

The restore scenario creates a working set larger than the configured device KV
pool, backs it with L3, applies cache pressure, and then revisits the prefixes at
fixed concurrency. The unique scenario measures write-side interference without
prefix reuse.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def make_shared_prompt(family: int, words: int, tail: str) -> str:
    shared = " ".join(
        f"urc_family_{family}_shared_context_{i % 257}" for i in range(words)
    )
    return (
        "You are evaluating deterministic prefix reuse. Read the context and "
        "answer with one short token.\n\n"
        f"{shared}\n\nrequest_tail_{tail}\nAnswer:"
    )


def make_unique_prompt(request_id: int, words: int) -> str:
    body = " ".join(
        f"urc_unique_{request_id}_{i}_{(request_id * 997 + i) % 10007}"
        for i in range(words)
    )
    return (
        "You are evaluating a no-reuse inference workload. Read the unique "
        f"context.\n\n{body}\n\nAnswer with one short token:"
    )


async def discover_model(session: aiohttp.ClientSession, base_url: str) -> str:
    async with session.get(f"{base_url}/v1/models") as response:
        response.raise_for_status()
        payload = await response.json()
    models = payload.get("data") or []
    if not models:
        raise RuntimeError("/v1/models returned no models")
    return str(models[0]["id"])


def extract_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    text = choice.get("text")
    return text if isinstance(text, str) else ""


async def send_completion(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    rid: str,
    max_tokens: int,
) -> Dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "rid": rid,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first_token_at = None
    output_parts: List[str] = []
    usage: Dict[str, Any] = {}
    status = "ok"
    error = None

    try:
        async with session.post(url, json=body) as response:
            if response.status != 200:
                status = "http_error"
                error = (await response.text())[:1000]
            else:
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(payload.get("usage"), dict):
                            usage = payload["usage"]
                        text = extract_text(payload)
                        if text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            output_parts.append(text)
    except Exception as exc:  # benchmark must preserve every failed record
        status = "exception"
        error = repr(exc)

    finish = time.perf_counter()
    return {
        "rid": rid,
        "status": status,
        "ttft_s": first_token_at - start if first_token_at is not None else None,
        "e2e_s": finish - start,
        "output_nonempty": bool("".join(output_parts)),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "error": error,
    }


async def run_bounded(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    jobs: List[tuple[str, str]],
    max_tokens: int,
    concurrency: int,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(rid: str, prompt: str) -> Dict[str, Any]:
        async with semaphore:
            return await send_completion(session, url, model, prompt, rid, max_tokens)

    return await asyncio.gather(*(run_one(rid, prompt) for rid, prompt in jobs))


def summarize(records: List[Dict[str, Any]], duration_s: float) -> Dict[str, Any]:
    successful = [
        record
        for record in records
        if record["status"] == "ok"
        and record["ttft_s"] is not None
        and record["output_nonempty"]
    ]
    ttfts = [record["ttft_s"] for record in successful]
    e2e = [record["e2e_s"] for record in successful]
    prompt_tokens = [
        record["prompt_tokens"]
        for record in successful
        if record["prompt_tokens"] is not None
    ]
    return {
        "request_count": len(records),
        "successful_count": len(successful),
        "duration_s": duration_s,
        "request_throughput": len(successful) / duration_s if duration_s else None,
        "ttft_s": {
            "mean": statistics.mean(ttfts) if ttfts else None,
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
            "max": max(ttfts) if ttfts else None,
        },
        "e2e_s": {
            "mean": statistics.mean(e2e) if e2e else None,
            "p50": percentile(e2e, 50),
            "p95": percentile(e2e, 95),
        },
        "prompt_tokens": {
            "min": min(prompt_tokens) if prompt_tokens else None,
            "max": max(prompt_tokens) if prompt_tokens else None,
            "mean": statistics.mean(prompt_tokens) if prompt_tokens else None,
        },
    }


async def async_main(args: argparse.Namespace) -> Dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 32))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        model = await discover_model(session, base_url)
        url = f"{base_url}/v1/completions"

        warmup = await send_completion(
            session,
            url,
            model,
            "UnifiedRadixCache benchmark warmup. Answer with one token:",
            f"urc-{args.run_label}-warmup",
            args.max_tokens,
        )

        preparation: List[Dict[str, Any]] = []
        if args.scenario == "restore":
            for family in range(args.prefix_count):
                preparation.append(
                    await send_completion(
                        session,
                        url,
                        model,
                        make_shared_prompt(family, args.prompt_words, "populate"),
                        f"urc-{args.run_label}-populate-{family}",
                        args.max_tokens,
                    )
                )
            await asyncio.sleep(args.settle_seconds)
            for churn_id in range(args.churn_count):
                preparation.append(
                    await send_completion(
                        session,
                        url,
                        model,
                        make_unique_prompt(100000 + churn_id, args.prompt_words),
                        f"urc-{args.run_label}-churn-{churn_id}",
                        args.max_tokens,
                    )
                )
            await asyncio.sleep(args.settle_seconds)

            rng = random.Random(args.seed)
            families = []
            while len(families) < args.requests:
                cycle = list(range(args.prefix_count))
                rng.shuffle(cycle)
                families.extend(cycle)
            jobs = [
                (
                    f"urc-{args.run_label}-measure-{index}",
                    make_shared_prompt(
                        family,
                        args.prompt_words,
                        f"measure_{index}",
                    ),
                )
                for index, family in enumerate(families[: args.requests])
            ]
        else:
            jobs = [
                (
                    f"urc-{args.run_label}-measure-{index}",
                    make_unique_prompt(index, args.prompt_words),
                )
                for index in range(args.requests)
            ]

        measure_start = time.perf_counter()
        records = await run_bounded(
            session,
            url,
            model,
            jobs,
            args.max_tokens,
            args.concurrency,
        )
        measure_duration = time.perf_counter() - measure_start

    return {
        "schema_version": "unified_radix_cache.benchmark.v1",
        "scenario": args.scenario,
        "run_label": args.run_label,
        "base_url": base_url,
        "model": model,
        "config": {
            "prefix_count": args.prefix_count,
            "churn_count": args.churn_count,
            "prompt_words": args.prompt_words,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "settle_seconds": args.settle_seconds,
            "seed": args.seed,
        },
        "warmup": warmup,
        "preparation": preparation,
        "summary": summarize(records, measure_duration),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--scenario", choices=["restore", "unique"], required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-count", type=int, default=12)
    parser.add_argument("--churn-count", type=int, default=6)
    parser.add_argument("--prompt-words", type=int, default=192)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.prefix_count <= 0 or args.requests <= 0 or args.concurrency <= 0:
        parser.error("prefix-count, requests, and concurrency must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(async_main(args))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
