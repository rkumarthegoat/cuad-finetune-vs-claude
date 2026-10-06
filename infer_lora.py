"""Run the fine tuned adapter over a split and write predictions in the same format as the baseline.

Usage
  python infer_lora.py --split val
  python infer_lora.py --split test      # run once, at the very end
"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--adapter", default="lora_out/adapter")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--tag", default="lora")
args = ap.parse_args()

cfg = json.load(open(args.adapter + "/train_config.json"))
tok = AutoTokenizer.from_pretrained(args.adapter, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(
    cfg["base_model"], torch_dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

rows = [json.loads(l) for l in open(f"out/{args.split}.jsonl")]
if args.limit:
    rows = rows[: args.limit]
texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False)
         for r in rows]
order = sorted(range(len(rows)), key=lambda i: len(texts[i]))   # similar lengths per batch, less padding

out_path = f"out/preds_{args.tag}_{args.split}.jsonl"
with open(out_path, "w") as fh:
    for b in range(0, len(order), args.batch):
        idx = order[b: b + args.batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=600, do_sample=False, pad_token_id=tok.pad_token_id)
        secs = time.time() - t0
        for j, i in enumerate(idx):
            new = gen[j, enc["input_ids"].shape[1]:]
            fh.write(json.dumps({
                "contract_id": rows[i]["contract_id"], "chunk_index": rows[i]["chunk_index"],
                "output": tok.decode(new, skip_special_tokens=True),
                "latency_s": secs, "seconds_total": secs / len(idx)}) + "\n")
        fh.flush()
        print(f"{min(b + args.batch, len(order))}/{len(order)}")
print("wrote", out_path)
