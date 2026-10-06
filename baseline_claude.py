"""Prompted frontier model baseline. Reads out/<split>.jsonl and writes out/preds_<tag>_<split>.jsonl.

Usage (set ANTHROPIC_API_KEY first)
  python baseline_claude.py --split val --limit 50          # cheap smoke test
  python baseline_claude.py --split val                     # full validation run
  python baseline_claude.py --split test --shots 3          # few shot, run once at the very end

The script resumes where it left off if interrupted, and logs tokens and latency per call so that
cost and speed can be compared fairly against the fine tuned model.
"""
import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--model", default="claude-sonnet-5-5")
ap.add_argument("--shots", type=int, default=0, help="number of positive training examples shown as few shot demos")
ap.add_argument("--cover_fields", action="store_true",
                help="few shot demos are the shortest training example for each of the four fields")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--tag", default=None)
args = ap.parse_args()

tag = args.tag or (f"{args.model}_cover" if args.cover_fields else f"{args.model}_{args.shots}shot")
out_path = f"out/preds_{tag}_{args.split}.jsonl"
rows = [json.loads(l) for l in open(f"out/{args.split}.jsonl")]
if args.limit:
    rows = rows[: args.limit]

# few shot demos come from the training split only
demos = []
if args.cover_fields:
    train = [json.loads(l) for l in open("out/train.jsonl")]
    used = set()
    for key in ["governing_law", "expiration_date", "renewal_term", "agreement_date"]:
        cands = [r for r in train if json.loads(r["completion"])[key] and id(r) not in used]
        pick = min(cands, key=lambda r: len(r["prompt"]) + len(r["completion"]))
        used.add(id(pick))
        demos += [{"role": "user", "content": pick["prompt"]}, {"role": "assistant", "content": pick["completion"]}]
elif args.shots:
    train = [json.loads(l) for l in open("out/train.jsonl")]
    positives = [r for r in train if r["completion"] != json.dumps({k: [] for k in json.loads(r["completion"])})]
    random.Random(0).shuffle(positives)
    for r in positives[: args.shots]:
        demos += [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}]

done = set()
if os.path.exists(out_path):
    done = {(d["contract_id"], d["chunk_index"]) for d in map(json.loads, open(out_path))}
todo = [r for r in rows if (r["contract_id"], r["chunk_index"]) not in done]
print(f"{len(rows)} chunks, {len(done)} already done, {len(todo)} to run, model {args.model}, shots {args.shots}, cover_fields {args.cover_fields}")

client = anthropic.Anthropic()

def call(row):
    messages = demos + [{"role": "user", "content": row["prompt"]}]
    for attempt in range(6):
        try:
            t0 = time.time()
            resp = client.messages.create(model=args.model, max_tokens=700, messages=messages)
            return {
                "contract_id": row["contract_id"], "chunk_index": row["chunk_index"],
                "output": "".join(b.text for b in resp.content if b.type == "text"),
                "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                "latency_s": time.time() - t0,
            }
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(2 ** attempt)
    raise RuntimeError("giving up on one chunk after repeated API errors")

with open(out_path, "a") as fh, ThreadPoolExecutor(args.workers) as pool:
    futures = [pool.submit(call, r) for r in todo]
    for i, f in enumerate(as_completed(futures), 1):
        fh.write(json.dumps(f.result()) + "\n")
        fh.flush()
        if i % 50 == 0:
            print(f"{i}/{len(todo)}")
print("wrote", out_path)
