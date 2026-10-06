"""LoRA fine tune of a small open model on out/train.jsonl. Written for a single Colab GPU.

Colab setup
  !pip install -q transformers peft accelerate
Usage
  python train_lora.py                                   # defaults to Qwen2.5 1.5B Instruct
  python train_lora.py --model Qwen/Qwen2.5-3B-Instruct  # needs a bigger GPU such as an L4 or A100
"""
import argparse, json, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--out", default="lora_out")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--max_len", type=int, default=1536)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--val_examples", type=int, default=300, help="size of the validation loss subset")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16   # half precision base weights, peft keeps the LoRA weights in fp32

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

class SFTData(Dataset):
    """Loss is computed only on the completion tokens, never on the prompt."""
    def __init__(self, path, limit=None):
        rows = [json.loads(l) for l in open(path)]
        if limit:
            random.Random(0).shuffle(rows); rows = rows[:limit]
        self.items, self.dropped = [], 0
        for r in rows:
            prompt_text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                                  add_generation_prompt=True, tokenize=False)
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            comp_ids = tok(r["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = prompt_ids + comp_ids
            if len(ids) > args.max_len:
                self.dropped += 1
                continue
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                               "labels": [-100] * len(prompt_ids) + comp_ids})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

train_ds = SFTData("out/train.jsonl")
val_ds = SFTData("out/val.jsonl", limit=args.val_examples)
print(f"train {len(train_ds)} (dropped {train_ds.dropped} too long), val subset {len(val_ds)}")

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, lr_scheduler_type="cosine", warmup_steps=20,
        bf16=use_bf16, fp16=not use_bf16 and torch.cuda.is_available(),
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        report_to="none", seed=args.seed, remove_unused_columns=False),
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
)
trainer.train()
model.save_pretrained(args.out + "/adapter")
tok.save_pretrained(args.out + "/adapter")
json.dump({"base_model": args.model, **{k: v for k, v in vars(args).items() if k != "model"}},
          open(args.out + "/adapter/train_config.json", "w"), indent=2)
print("saved adapter to", args.out + "/adapter")
