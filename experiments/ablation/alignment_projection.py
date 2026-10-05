"""Train frozen-feature alignment towers. Input: UNI2-h and scFoundation cell features. Output: normalized 128-D tower checkpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml


class Tower(nn.Module):
    def __init__(self, din, dh, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, dout))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def chunk_stats(x, device, chunk=400000):
    n, d = x.shape
    s = torch.zeros(d, dtype=torch.float64, device=device)
    ss = torch.zeros(d, dtype=torch.float64, device=device)
    for i in range(0, n, chunk):
        c = x[i:i + chunk].to(device).float()
        s += c.sum(0).double()
        ss += (c * c).sum(0).double()
    mu = s / n
    var = (ss / n - mu * mu).clamp(min=0)
    return mu.float(), var.sqrt().float() + 1e-6


def load_pairs(slides, max_cells):
    sizes = []
    for slide in slides:
        with np.load(slide['features'], allow_pickle=True) as z:
            sizes.append(len(z['E1536']))
    total = sum(sizes)
    if total == 0:
        raise ValueError('No training cells')
    frac = min(1.0, max_cells / total)
    counts = [int(round(n * frac)) for n in sizes]
    he = np.empty((sum(counts), 1536), np.float16)
    scf = np.empty((sum(counts), 3072), np.float16)
    rng = np.random.RandomState(0)
    off = 0
    for slide, n, count in zip(slides, sizes, counts):
        sel = np.sort(rng.choice(n, count, replace=False)) if frac < 1 else np.arange(n)
        with np.load(slide['features'], allow_pickle=True) as z:
            e = z['E1536']
            if e.shape != (n, 1536):
                raise ValueError('Expected frozen UNI2-h features [N,1536]')
            rows = z['expr_rows'].astype(np.int64) if 'expr_rows' in z else np.arange(n)
            source = np.load(slide['scf'], mmap_mode='r')
            if source.ndim != 2 or source.shape[1] != 3072 or (len(rows) and (rows.min() < 0 or rows.max() >= len(source))):
                raise ValueError('scFoundation features must cover every expression row and have 3072 columns')
            he[off:off + count] = e[sel].astype(np.float16)
            scf[off:off + count] = source[rows[sel]].astype(np.float16)
        off += count
    keep = (np.abs(scf).sum(1) > 0) & (np.abs(he).astype(np.float32).sum(1) > 0)
    he, scf = np.ascontiguousarray(he[keep]), np.ascontiguousarray(scf[keep])
    if len(he) < 2 or not np.isfinite(he).all() or not np.isfinite(scf).all():
        raise ValueError('Need at least two finite, nonzero training pairs')
    return torch.from_numpy(he), torch.from_numpy(scf)


def train_towers(he, scf, args):
    dev = torch.device(args.device)
    he_mu, he_sd = chunk_stats(he, dev)
    scf_mu, scf_sd = chunk_stats(scf, dev)
    he_tower = Tower(1536, args.hidden, 128).to(dev)
    scf_tower = Tower(3072, args.hidden, 128).to(dev)
    ls = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(list(he_tower.parameters()) + list(scf_tower.parameters()) + [ls],
                           lr=args.lr, weight_decay=1e-4)
    for ep in range(args.epochs):
        perm = torch.randperm(len(he))
        total, batches = 0.0, 0
        for i in range(0, len(he) - 1, args.bs):
            b = perm[i:i + args.bs]
            if len(b) < 2:
                continue
            hb = (he[b].to(dev).float() - he_mu) / he_sd
            sb = (scf[b].to(dev).float() - scf_mu) / scf_sd
            zi, zg = he_tower(hb), scf_tower(sb)
            labels = torch.arange(len(b), device=dev)
            logits = ls.clamp(max=np.log(100)).exp() * zi @ zg.t()
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        print(f'epoch {ep + 1}: loss={total / max(batches, 1):.4f}', flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, tower, mu, sd, din in [('he', he_tower, he_mu, he_sd, 1536),
                                     ('scf', scf_tower, scf_mu, scf_sd, 3072)]:
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in tower.state_dict().items()},
                        mu=mu.cpu(), sd=sd.cpu(), in_dim=din, hidden=args.hidden, out_dim=128,
                        n_cells=len(he), epochs=args.epochs), out / f'clip_{name}_tower.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides', required=True, help='YAML: training slides with features and scf paths')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max_cells', type=int, default=4000000)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if a.epochs < 1 or a.bs < 2 or a.max_cells < 2 or a.lr <= 0:
        ap.error('Require epochs>=1, bs>=2, max_cells>=2 and lr>0')
    torch.manual_seed(0)
    np.random.seed(0)
    slides = yaml.safe_load(Path(a.slides).read_text())['slides']
    he, scf = load_pairs(slides, a.max_cells)
    train_towers(he, scf, a)


if __name__ == '__main__':
    main()
