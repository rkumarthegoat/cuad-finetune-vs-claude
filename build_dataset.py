"""Build the contract level split and chunked dataset for the CUAD extraction project."""
import json, random, re
from collections import Counter

SEED = 42
CHUNK_CHARS = 2500
OVERLAP_CHARS = 500
FIELDS = {
    "Governing Law": "governing_law",
    "Expiration Date": "expiration_date",
    "Renewal Term": "renewal_term",
    "Agreement Date": "agreement_date",
}
norm = lambda t: " ".join(t.split())

raw = json.load(open("CUADv1.json"))["data"]

# 1. gather contracts with gold spans (offsets into the raw text) for the four fields
contracts = []
mismatch = 0
for item in raw:
    para = item["paragraphs"][0]
    ctx = para["context"]
    gold = {v: [] for v in FIELDS.values()}
    for qa in para["qas"]:
        key = qa["id"].split("__")[-1]
        if key in FIELDS and not qa["is_impossible"]:
            for a in qa["answers"]:
                s, t = a["answer_start"], a["text"]
                if ctx[s:s + len(t)] != t:
                    mismatch += 1
                    continue
                gold[FIELDS[key]].append((s, s + len(t)))
    contracts.append({"title": item["title"], "text": ctx, "gold": gold})
print("offset mismatches skipped:", mismatch)

# 2. split by contract, so no contract ever appears in two splits
rng = random.Random(SEED)
order = list(range(len(contracts)))
rng.shuffle(order)
n = len(order)
n_train, n_val = int(0.70 * n), int(0.15 * n)
split_of = {}
for rank, i in enumerate(order):
    split_of[i] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
print("contracts per split:", Counter(split_of.values()))

# 3. chunk each contract on whitespace boundaries with overlap
def chunk_bounds(text):
    n, start, out = len(text), 0, []
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        if end < n:
            cut = text.rfind(" ", end - 300, end)
            end = cut if cut > start else end
        out.append((start, end))
        if end >= n:
            break
        nxt = end - OVERLAP_CHARS
        sp = text.find(" ", nxt)
        start = sp + 1 if 0 <= sp < end else nxt
    return out

chunks, cut_spans, total_spans = [], 0, 0
for ci, c in enumerate(contracts):
    bounds = chunk_bounds(c["text"])
    for k, (cs, ce) in enumerate(bounds):
        target = {}
        for f, spans in c["gold"].items():
            target[f] = [norm(c["text"][s:e]) for s, e in spans if s >= cs and e <= ce]
        chunks.append({
            "contract_id": ci, "title": c["title"], "split": split_of[ci],
            "chunk_index": k, "n_chunks": len(bounds), "text": norm(c["text"][cs:ce]),
            "target": target,
        })
    for f, spans in c["gold"].items():
        for s, e in spans:
            total_spans += 1
            if not any(s >= cs and e <= ce for cs, ce in bounds):
                cut_spans += 1
print("gold spans split across every chunk (unrecoverable):", cut_spans, "of", total_spans)

# 4. write contract level gold (for evaluation) and chunk level examples
with open("out/contracts_gold.jsonl", "w") as fh:
    for ci, c in enumerate(contracts):
        gold = {f: [norm(c["text"][s:e]) for s, e in sp] for f, sp in c["gold"].items()}
        fh.write(json.dumps({"contract_id": ci, "title": c["title"], "split": split_of[ci],
                             "gold": gold}) + "\n")
with open("out/chunks_all.jsonl", "w") as fh:
    for ch in chunks:
        fh.write(json.dumps(ch) + "\n")

# 5. summary
by = Counter(ch["split"] for ch in chunks)
pos = Counter(ch["split"] for ch in chunks if any(ch["target"].values()))
print("chunks per split:", dict(by))
print("chunks with at least one gold span:", dict(pos))
for sp in ("train", "val", "test"):
    cnt = Counter()
    for c_i, c in enumerate(contracts):
        if split_of[c_i] == sp:
            for f, spans in c["gold"].items():
                cnt[f] += len(spans) > 0
    print(sp, "contracts with field:", dict(cnt))
lens = [len(ch["text"]) for ch in chunks]
print("chunk chars min/median/max:", min(lens), sorted(lens)[len(lens)//2], max(lens))
