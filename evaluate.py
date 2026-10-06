"""Contract level scoring used for every model. Predictions are chunk level JSON outputs."""
import json, re, sys
from collections import Counter, defaultdict

KEYS = ["governing_law", "expiration_date", "renewal_term", "agreement_date"]
tok = lambda s: re.findall(r"\w+", s.lower())

def token_f1(pred_spans, gold_spans):
    p, g = Counter(t for s in pred_spans for t in tok(s)), Counter(t for s in gold_spans for t in tok(s))
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)

_M = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE_PATTERNS = [rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{_M},?\s+\d{{4}}",
                  rf"{_M}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}", r"\d{1,2}/\d{1,2}/\d{2,4}", rf"{_M}\s+\d{{4}}"]

def trim_to_date(span):
    """Optional postprocessing for agreement_date. Keeps only the date text inside a longer quote."""
    for p in _DATE_PATTERNS:
        m = re.search(p, span, flags=re.I)
        if m:
            return m.group(0)
    return span

def parse(text):
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return {k: [s for s in obj.get(k, []) if isinstance(s, str)] for k in KEYS}
    except Exception:
        return {k: [] for k in KEYS}

def aggregate(chunk_preds, trim_dates=False):
    """chunk_preds is a list of (contract_id, completion_text). Merge and dedupe per contract."""
    out = defaultdict(lambda: {k: [] for k in KEYS})
    for cid, text in chunk_preds:
        for k, spans in parse(text).items():
            for s in spans:
                if trim_dates and k == "agreement_date":
                    s = trim_to_date(s)
                if s not in out[cid][k]:
                    out[cid][k].append(s)
    return out

def score(pred_by_contract, gold_path="out/contracts_gold.jsonl", split="test"):
    gold = {g["contract_id"]: g["gold"] for g in map(json.loads, open(gold_path)) if g["split"] == split}
    res = {}
    for k in KEYS:
        tp = fp = fn = 0
        f1s = []
        for cid, g in gold.items():
            pred = pred_by_contract.get(cid, {}).get(k, [])
            has_gold, has_pred = bool(g[k]), bool(pred)
            tp += has_gold and has_pred
            fp += (not has_gold) and has_pred
            fn += has_gold and not has_pred
            if has_gold:
                f1s.append(token_f1(pred, g[k]))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        res[k] = {"span_f1": sum(f1s) / len(f1s), "presence_precision": pr, "presence_recall": rc,
                  "presence_f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "n_gold_contracts": len(f1s)}
    res["macro_span_f1"] = sum(res[k]["span_f1"] for k in KEYS) / len(KEYS)
    return res

if __name__ == "__main__":
    # sanity checks, a perfect oracle should score 1.0 and an empty predictor should score 0.0
    chunks = [json.loads(l) for l in open("out/chunks_all.jsonl") if json.loads(l)["split"] == "test"]
    oracle = aggregate([(c["contract_id"], json.dumps(c["target"])) for c in chunks])
    empty = aggregate([(c["contract_id"], "{}") for c in chunks])
    print("oracle macro span F1", round(score(oracle)["macro_span_f1"], 3))
    print("empty macro span F1", round(score(empty)["macro_span_f1"], 3))
