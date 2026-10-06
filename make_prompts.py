"""Turn chunk level examples into prompt/completion pairs shared by both models."""
import json, random
from collections import Counter

SEED = 42
NEG_PER_POS = 2   # training only, negatives kept per positive chunk
KEYS = ["governing_law", "expiration_date", "renewal_term", "agreement_date"]

INSTRUCTION = (
    "You are reading one excerpt of a commercial contract. Find any sentences or clauses in the excerpt "
    "that state the governing law, the expiration date or term length, the renewal terms, or the date the "
    "agreement was signed. Copy each one exactly as written. Reply with only a JSON object with the keys "
    + ", ".join(KEYS) + ", each holding a list of exact quotes from the excerpt. Use an empty list when the "
    "excerpt has nothing for that key.\n\nExcerpt:\n"
)

def prompt(chunk_text):
    return INSTRUCTION + chunk_text

def completion(target):
    return json.dumps({k: target[k] for k in KEYS}, ensure_ascii=False)

chunks = [json.loads(l) for l in open("out/chunks_all.jsonl")]
rng = random.Random(SEED)
files = {s: open(f"out/{s}.jsonl", "w") for s in ("train", "val", "test")}
counts = Counter()
train_pos = [c for c in chunks if c["split"] == "train" and any(c["target"].values())]
train_neg = [c for c in chunks if c["split"] == "train" and not any(c["target"].values())]
keep_train = train_pos + rng.sample(train_neg, min(len(train_neg), NEG_PER_POS * len(train_pos)))
rng.shuffle(keep_train)
for c in chunks:
    if c["split"] != "train":
        files[c["split"]].write(json.dumps({"contract_id": c["contract_id"], "chunk_index": c["chunk_index"],
            "prompt": prompt(c["text"]), "completion": completion(c["target"])}) + "\n")
        counts[c["split"]] += 1
for c in keep_train:
    files["train"].write(json.dumps({"contract_id": c["contract_id"], "chunk_index": c["chunk_index"],
        "prompt": prompt(c["text"]), "completion": completion(c["target"])}) + "\n")
    counts["train"] += 1
print("examples written", dict(counts), "| train positives", len(train_pos), "negatives kept", len(keep_train) - len(train_pos))
