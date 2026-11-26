# -*- coding: utf-8 -*-
"""Qwen2.5-0.5B inference optimization: fp16 baseline vs int4 (bnb nf4)."""
import argparse, json, math, os, time, random, io
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def format_chat(tokenizer, ex):
    inp = ex["instruction"]
    if ex.get("input") and str(ex["input"]).strip() and str(ex["input"]) != "<noinput>":
        inp = inp + "\n" + str(ex["input"])
    msgs = [{"role":"user","content":inp},{"role":"assistant","content":ex["output"]}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

def format_prompt(tokenizer, ex):
    inp = ex["instruction"]
    if ex.get("input") and str(ex["input"]).strip() and str(ex["input"]) != "<noinput>":
        inp = inp + "\n" + str(ex["input"])
    msgs = [{"role":"user","content":inp}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

class InstructDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        ex = self.data[i]
        text = format_chat(self.tokenizer, ex)
        enc = self.tokenizer(text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels[input_ids == pad_id] = -100
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"][0], "labels": labels}

def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

def compute_ppl(model, dataloader, device):
    model.eval()
    total_loss = 0; n = 0
    with torch.no_grad():
        for b in dataloader:
            input_ids = b["input_ids"].to(device)
            attn = b["attention_mask"].to(device)
            labels = b["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            total_loss += out.loss.item() * input_ids.size(0)
            n += input_ids.size(0)
    return total_loss / max(n, 1), math.exp(total_loss / max(n, 1))

def gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def bench_generation(model, tokenizer, prompt_ids, max_new_tokens=128):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, out.shape[1] - prompt_ids.shape[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="PROJECT_ROOT/hf_cache/Qwen2.5-0.5B")
    ap.add_argument("--data", default="PROJECT_ROOT/datasets/alpaca_gpt4_data_zh.json")
    ap.add_argument("--output", default="PROJECT_ROOT/output_optimize")
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--n_gen", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    set_seed(42)
    device = args.device if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = json.load(io.open(args.data, encoding="utf-8"))
    random.seed(42); random.shuffle(data)
    eval_data = data[:args.n_eval]
    gen_data = data[args.n_eval:args.n_eval+args.n_gen]
    eval_ds = InstructDataset(eval_data, tokenizer, args.max_len)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate)
    gen_prompts = [tokenizer(format_prompt(tokenizer, ex), return_tensors="pt").input_ids.to(device) for ex in gen_data]

    size_path = os.path.join(args.model, "model.safetensors")
    disk_size_mb = os.path.getsize(size_path) / 1024 / 1024

    # Only fp16 and int4 (int8 via bnb on Windows is pathologically slow, see NOTES)
    arms = [
        {"name": "fp16_baseline", "dtype": torch.float16, "quant": None},
        {"name": "int4_nf4_bnb",  "dtype": None,          "quant": "int4"},
    ]

    results = []
    for arm in arms:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        name = arm["name"]
        print(f"\\n=== {name} ===", flush=True)
        try:
            if arm["quant"] == "int4":
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                             device_map={"": 0}, trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=arm["dtype"]).to(device)
            model.eval()

            _ = model.generate(gen_prompts[0], max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)

            loss, ppl = compute_ppl(model, eval_dl, device)
            decode_times = []; decode_toks = []
            for pids in gen_prompts:
                t, ntok = bench_generation(model, tokenizer, pids, max_new_tokens=args.max_new_tokens)
                decode_times.append(t); decode_toks.append(ntok)
            decode_tps = sum(decode_toks) / sum(decode_times) if sum(decode_times) > 0 else 0
            vram = gpu_mem_mb()

            samples = []
            for pids in gen_prompts[:3]:
                with torch.no_grad():
                    out = model.generate(pids, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                text = tokenizer.decode(out[0][pids.shape[1]:], skip_special_tokens=True)
                samples.append({"prompt": tokenizer.decode(pids[0], skip_special_tokens=True)[:200], "generated": text})

            results.append({
                "arm": name,
                "quantization": arm["quant"] or "none",
                "disk_size_mb": round(disk_size_mb, 2),
                "peak_vram_mb": round(vram, 2),
                "ppl": round(ppl, 4),
                "ppl_loss": round(loss, 4),
                "decode_tokens_per_s": round(decode_tps, 2),
                "samples": samples,
            })
            print(f"  PPL={ppl:.4f}  decode={decode_tps:.2f} tok/s  vram={vram:.0f}MB", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"arm": name, "error": str(e)})

    summary_path = os.path.join(args.output, "benchmark.json")
    with io.open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "arms": results, "n_eval": args.n_eval, "note_int8": "bitsandbytes int8 on Windows + bf16 was pathologically slow (>25 min without completing eval); excluded from final comparison."}, f, ensure_ascii=False, indent=2)
    print(f"\\nSaved {summary_path}", flush=True)

if __name__ == "__main__":
    main()
