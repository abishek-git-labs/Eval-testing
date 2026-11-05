# Eval-testing
A repo to understand how to load small LLM models on your local machine and evaluate the models. And along with that we are using a different model to judge the answer of the evaluation done by another model


# LLM-as-a-Judge 

Teaching project for evaluation Session 3: datasets, evaluators, LLM-as-a-judge and judge validation.
A local model answers questions, and a local judge (Qwen3-8B or Gemma 3 4B) scores them using LM Studio.
LangSmith stores traces, datasets and experiments.

## Prerequisites
- macOS or Linux, Python 3.10+ (developed on 3.14)
- [LM Studio](https://lmstudio.ai) with these models downloaded:
  - `qwen3-8b` (Q4_K_M)
  - `gemma-3-4b-instruct` (Q4_K_M), optional second judge
- A free [LangSmith](https://smith.langchain.com) account and API key

## Setup
```bash
cd "<directory>"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
```
(If above pip cmd does not work do this - pip install openai langsmith python-dotenv truststore)
```

Create `.env` in the project root. Never commit it.
```
LANGSMITH_API_KEY=your_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=llm-judge-demo
EVAL_MODEL=qwen3-8b
```

### LM Studio
1. Load a model, then open the Developer tab and start the server at `http://localhost:1234/v1`.
2. Use temperature 0, a blank system prompt, max concurrent predictions 1, and Thinking/reasoning off.
3. LM Studio runs one model at a time. **Load the model you name in a command before running it**, and reload when switching judges.

## Project layout
| Path | Purpose |
|---|---|
| `src/config.py` | Paths, `.env`, model settings, `EVAL_PROFILES` |
| `src/datasets/` | Policy `.txt` files and datasets (JSON) |
| `src/results/` | Saved answers and results |
| `src/evals/run_eval.py` | Generates answers (streaming, tokens, TTFT, tracing) |
| `src/evals/evaluate_results.py` | Exact-match baseline |
| `src/evals/judge_eval.py` | Local pass/fail judge |
| `src/evals/langsmith_eval.py` | 4-criterion 1–5 rubric judge, uploaded to LangSmith |
| `tests/` | Offline tests |

## Profiles
- `willow`: Willow & Pine returns policy, 80 cases (default)
- `it-helpdesk`: IT helpdesk policy, 12 cases (3 correct controls, 9 flawed answers)

Add a profile by adding a policy and dataset to `src/datasets/` and an entry in `EVAL_PROFILES` in `src/config.py`.

## Run (always from the project root, with the venv active)
```bash
# 1. Generate answers (Qwen loaded)
python -m src.evals.run_eval --profile willow --model qwen3-8b --limit 10

# 2. Exact-match baseline
python -m src.evals.evaluate_results --results-file src/results/eval_results_<timestamp>.json

# 3. Optional local pass/fail judge
python -m src.evals.judge_eval --profile willow --results-file src/results/eval_results_<timestamp>.json --limit 10

# 4. LangSmith rubric judge (pass the filename only, not a path)
python -m src.evals.langsmith_eval --profile willow \
  --answers-file eval_results_<timestamp>.json --limit 10 \
  --judge-model qwen3-8b --experiment-prefix qwen-judge

# 5. Same answers, different judge (load Gemma in LM Studio first)
python -m src.evals.langsmith_eval --profile willow \
  --answers-file eval_results_<timestamp>.json --limit 10 \
  --judge-model gemma-3-4b-instruct --experiment-prefix gemma-judge
```
`--limit` must not exceed the number of saved answers. Reusing the same answers file and limit across judges gives a fair comparison.
To trace into a different project: `LANGSMITH_PROJECT=other-name python -m src.evals.run_eval ...`

## Where to look in LangSmith
- **Datasets & Experiments**: judge scores and per-case feedback
- **Tracing → your project**: generation traces (`answer_llm`, `judge_llm`)
- **Run → Attributes → Metadata**: TTFT, latency, tokens/sec

## Metrics
- Rubric 1–5 maps to 0–1 as `(rating-1)/4`. These are categories, not probabilities.
- `ttft_seconds`: time to the first content token
- `latency_seconds`: whole-request time
- `end_to_end_output_tokens_per_second`: completion tokens ÷ full request time. It includes network and prompt processing, so it is not a decoding benchmark.
- Token counts are recorded only when LM Studio returns them. They are never estimated.

## Validating the judge
Rate a sample yourself, then compare to the judge's scores. Agreement is computed manually. Use held-out cases, and expect failure modes such as verbosity bias and shared-model blind spots.


## Troubleshooting
| Problem | Fix |
|---|---|
| Connection refused | Start the LM Studio server and load the model |
| Model not found | The ID must match, e.g. `qwen3-8b` or `gemma-3-4b-instruct` |
| `No saved answer for case-XX` | Lower `--limit` to the number of saved answers |
| Import errors | Run from the project root using `python -m ...` |
| Metrics missing in LangSmith | Open Attributes → Metadata. Older runs aren't updated. |
| SSL errors on macOS | `truststore` is already used. Never disable verification. |

## Privacy and limitations
- Policy text, questions, answers and traces are uploaded to LangSmith. Use fictional data only.
- Never commit `.env`.
- Not included: retrieval/RAG, embedding-based similarity, CI gating.
- The answerer and judge may share blind spots when both are Qwen.
