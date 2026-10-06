"""Score a predictions file and report quality, cost, and speed in one table.

Usage
  python score_preds.py out/preds_<tag>_val.jsonl --split val --price_in 3 --price_out 15
Prices are dollars per million tokens, copy them from the provider pricing page on the day you run.
For the fine tuned model pass --gpu_hourly instead, the dollars per hour of the GPU you rented.
"""
import argparse, json
from evaluate import aggregate, score, KEYS

ap = argparse.ArgumentParser()
ap.add_argument("preds")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--price_in", type=float, default=None)
ap.add_argument("--price_out", type=float, default=None)
ap.add_argument("--gpu_hourly", type=float, default=None)
ap.add_argument("--trim_dates", action="store_true", help="trim agreement_date quotes down to the date text, applied to any model")
args = ap.parse_args()

preds = [json.loads(l) for l in open(args.preds)]
covered = {p["contract_id"] for p in preds}
res = score(aggregate([(p["contract_id"], p["output"]) for p in preds], trim_dates=args.trim_dates), split=args.split)

# only meaningful when every chunk of the split was run, so flag partial runs loudly
total_chunks = sum(1 for _ in open(f"out/{args.split}.jsonl"))
if len(preds) < total_chunks:
    print(f"WARNING partial run, {len(preds)} of {total_chunks} chunks, scores are not comparable yet\n")

print(f"{'field':18s} {'span F1':>8s} {'presence P':>11s} {'presence R':>11s} {'presence F1':>12s}")
for k in KEYS:
    r = res[k]
    print(f"{k:18s} {r['span_f1']:8.3f} {r['presence_precision']:11.3f} {r['presence_recall']:11.3f} {r['presence_f1']:12.3f}")
print(f"{'macro span F1':18s} {res['macro_span_f1']:8.3f}\n")

n = len(preds)
lat = sorted(p["latency_s"] for p in preds if "latency_s" in p)
if lat:
    print(f"latency per chunk, median {lat[len(lat)//2]:.2f}s, p95 {lat[int(.95*len(lat))]:.2f}s")
if args.price_in is not None and "input_tokens" in preds[0]:
    cost = sum(p["input_tokens"] for p in preds) / 1e6 * args.price_in + sum(p["output_tokens"] for p in preds) / 1e6 * args.price_out
    print(f"cost, ${cost:.2f} total, ${cost / n * 1000:.2f} per 1000 chunks")
if args.gpu_hourly is not None and "seconds_total" in preds[0]:
    secs = sum(p["seconds_total"] for p in preds)
    cost = secs / 3600 * args.gpu_hourly
    print(f"GPU cost, ${cost:.2f} total, ${cost / n * 1000:.2f} per 1000 chunks")
