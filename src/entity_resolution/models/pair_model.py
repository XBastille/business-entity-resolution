import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "FacebookAI/xlm-roberta-large"


def text(frame):
    """The pair model's view of a record: "name | address"."""
    return (frame["business_name"].fill_null("") + " | " + frame["business_address"].fill_null("")).to_numpy()


def encode(tok, a, b, max_len, device):
    return tok(list(a), list(b), truncation="longest_first", max_length=max_len, padding=True,
               return_tensors="pt").to(device)


def logits(model, tok, a, b, max_len=128, batch=512):
    """Cross-encoder logits for text pairs (record text a, business text b), batched in length order."""
    device = next(model.parameters()).device
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    out = np.empty(len(order), dtype=np.float32)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(order), batch):
            rows = order[i:i + batch]
            out[rows] = model(**encode(tok, a[rows], b[rows], max_len, device)).logits[:, 0].float().cpu().numpy()
    return out


def train(a, b, y, out_dir, model_name=MODEL, batch=64, lr=1e-5, max_len=128, save_every=5000, seed=3407):
    """Fine-tune a pretrained encoder as a binary pair judge for one epoch (bf16, AdamW, 5% warmup then linear decay).

    Resumes from out_dir/checkpoint.pt; saves the model to out_dir/model and returns it with its tokenizer. 20,000
    pairs are held back and reported as a sanity check.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "the pair model needs a GPU"
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if (out_dir / "model" / "config.json").exists():
        tok = AutoTokenizer.from_pretrained(out_dir / "model")
        return AutoModelForSequenceClassification.from_pretrained(out_dir / "model", num_labels=1).to(device), tok
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    order = np.random.default_rng(seed).permutation(len(y))
    n_hold = min(20_000, len(order) // 50)
    hold, order = order[:n_hold], order[n_hold:]
    steps = math.ceil(len(order) / batch)
    warm = max(1, int(0.05 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ckpt, start = out_dir / "checkpoint.pt", 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["step"]
    print(f"pair model: {len(order):,} pairs, {steps:,} steps of {batch}, from step {start:,}", flush=True)
    model.train()
    t0, seen = time.perf_counter(), 0
    for step in range(start, steps):
        rows = order[step * batch:(step + 1) * batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**encode(tok, a[rows], b[rows], max_len, device)).logits[:, 0]
        loss = loss_fn(out.float(), torch.tensor(y[rows], dtype=torch.float32, device=device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        seen += len(rows)
        if (step + 1) % 1000 == 0:
            rate = seen / (time.perf_counter() - t0)
            print(f"step {step + 1:,}/{steps:,}  loss {loss.item():.4f}  {rate:,.0f} pairs/s", flush=True)
        if (step + 1) % save_every == 0:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1}, tmp)
            os.replace(tmp, ckpt)
    model.save_pretrained(out_dir / "model")
    tok.save_pretrained(out_dir / "model")
    p = 1 / (1 + np.exp(-logits(model, tok, a[hold], b[hold], max_len)))
    print(f"pair model held-back accuracy {np.mean((p > 0.5) == (y[hold] == 1)):.4f}", flush=True)
    return model, tok
