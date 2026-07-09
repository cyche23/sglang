# Environment Notes for Local SGLang Development and Validation

This document describes the local SGLang development and validation environment.

The key rule is:

**Agents do not need to enter the Docker container to edit source code. The container is mainly used to launch and validate `sglang_server` with the correct runtime environment.**

## 1. Device and project context

Target device:

* Device: Jetson AGX Orin
* Runtime platform: Jetson Linux / L4T R36.x style environment
* SGLang baseline version: `v0.5.4`
* Main working branch: `local-v0.5.4-baseline`

Research context:

* The project targets on-device LLM serving experiments.
* Current focus includes:

  * KV cache reuse
  * KV cache offloading
  * HiCache / hierarchical cache behavior
  * Jetson unified-memory-aware serving baseline
  * SSD-backed KV cache experiments

Related project background may be found in:

```bash
docs/agent-runtime-on-device.md
```

## 2. Source-code editing rule

Source-code editing should be performed in the normal local repository workspace available to the agent.

Agents do **not** need to enter the Docker container just to edit files.

Before editing, always check:

```bash
git branch --show-current
git status --short
```

Expected branch:

```bash
local-v0.5.4-baseline
```

Rules:

1. Work on the current local SGLang repository.
2. Keep changes minimal and reviewable.
3. Do not modify unrelated files.
4. Do not reformat unrelated files.
5. Do not change dependencies unless explicitly instructed.
6. Do not modify `sgl-kernel`, CUDA kernels, model-loading logic, or unrelated scheduler code unless the task explicitly requires it.
7. New experimental behavior should be disabled by default and enabled only by explicit flags or configs.

## 3. Container role

The Docker container provides the runtime environment needed to launch and validate SGLang server on Jetson.

The container should be used for:

* launching `sglang_server`
* sending local validation requests
* running demo clients that depend on the container runtime
* checking whether modified SGLang code works under the actual Jetson SGLang environment

The container should **not** be treated as the primary source-editing environment unless explicitly instructed.

Existing container name:

```bash
sglang-dev-v054
```

Start the container with:

```bash
docker start sglang-dev-v054
```

Enter the container only when runtime validation is needed:

```bash
docker exec -it sglang-dev-v054 bash
```

If the container is already running, only `docker exec` is needed.

Do not delete, recreate, or reinstall the container unless explicitly instructed.

## 4. Source path inside the container

Inside the container, the SGLang source tree is available at:

```bash
/codes/sglang
```

The container has read/write access to this path.

However, for normal agent work:

* edit code from the local repository workspace
* use `/codes/sglang` inside the container only to launch server validation and run runtime commands

Inside the container, validation commands should generally start with:

```bash
cd /codes/sglang
```

Before launching validation, inspect:

```bash
cd /codes/sglang
git branch --show-current
git status --short
python3 -c "import sglang; print(sglang.__file__)"
python3 -m sglang.launch_server --help | head -n 80
```

If `/codes/sglang` does not reflect the latest source edits, stop and report the mismatch instead of validating stale code.

## 5. Model paths inside the container

Common model directory:

```bash
/models
```

Common model path used in prior validation:

```bash
/models/Qwen3-1.7B/origin
```

Another possible model path:

```bash
/models/Qwen3-8B
```

Before assuming a model path, check:

```bash
ls -lah /models
```

Prefer the smaller model for smoke tests unless the task requires a larger model:

```bash
/models/Qwen3-1.7B/origin
```

## 6. Server launch rule

SGLang server must be launched inside the container for runtime validation.

Start the container:

```bash
docker start sglang-dev-v054
```

Open a shell inside the container:

```bash
docker exec -it sglang-dev-v054 bash
```

Inside the container:

```bash
cd /codes/sglang

python3 -m sglang.launch_server \
  --model-path /models/Qwen3-1.7B/origin \
  --host 0.0.0.0 \
  --port 8000
```

For experimental cache features, add task-specific flags explicitly:

```bash
cd /codes/sglang

python3 -m sglang.launch_server \
  --model-path /models/Qwen3-1.7B/origin \
  --host 0.0.0.0 \
  --port 8000 \
  [TASK_SPECIFIC_FLAGS]
```

Rules:

1. Do not claim server validation passed unless the server actually starts.
2. Do not silently change the port.
3. If port `8000` is occupied, inspect and report the issue.
4. If a different port is used, use the same port in all client commands and final reports.

Check port usage:

```bash
ss -ltnp | grep 8000
```

## 7. Client request validation

Client requests should also be sent from inside the same container unless explicitly instructed otherwise.

Open a second container shell:

```bash
docker exec -it sglang-dev-v054 bash
```

Inside the container:

```bash
cd /codes/sglang
curl http://127.0.0.1:8000/v1/models
```

A successful response should list the loaded model.

Example chat completion request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-1.7B/origin",
    "messages": [
      {"role": "user", "content": "Hello, briefly introduce KV cache."}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

If the model id differs from the model path, first query:

```bash
curl http://127.0.0.1:8000/v1/models
```

Then use the returned model id in later requests.