"""Donor-grouped train / held-out split of each evaluation tissue's Stage-2 training slides. Input: training slides with
donor labels, target slides, and the head gene table. Output: split JSON and Stage-2 slide lists."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd


def head_panel(directory, mapping):
    from voice.panel import read_genes, real_gene_mask
    d = Path(directory); genes = read_genes(d / 'genes.tsv')
    h5 = d / 'features.h5'
    mask, _ = real_gene_mask(genes, str(h5) if h5.exists() else None)
    return {mapping[g] for g, m in zip(genes, mask) if m and g in mapping}


def n_cells(directory):
    with np.load(Path(directory) / 'patch_cell_boundaries.npz', allow_pickle=True) as z:
        return int(len(z['expr_rows']))


def bits(ids):
    out = 0
    for g in ids: out |= 1 << int(g)
    return out


def choose(groups, slides, panel, cells, rest, h_full, test, val_frac):
    """Exhaustive search over donor-group assignments (2^groups).
    Hard: the union of trained genes is unchanged; donor groups are never split.
    Lexicographic objective: train keeps all same-tissue test-gene coverage, val covers as many test genes as
    possible, val fraction by slides then by cells closest to val_frac."""
    G = sorted(groups); best = None
    tot_cells = sum(cells[s] for s in slides)
    for mask in itertools.product([0, 1], repeat=len(G)):
        if not any(mask) or all(mask): continue
        val = [s for g, b in zip(G, mask) if b for s in groups[g]]; trn = [s for s in slides if s not in val]
        tr_bits = 0
        for s in trn: tr_bits |= panel[s]
        if rest | tr_bits != h_full: continue
        va_bits = 0
        for s in val: va_bits |= panel[s]
        s1, s2 = (test & tr_bits).bit_count(), (test & va_bits).bit_count()
        fs = len(val) / len(slides); fc = sum(cells[s] for s in val) / tot_cells
        key = (-s1, -s2, round(abs(fs - val_frac), 6), round(abs(fc - val_frac), 6))
        if best is None or key < best[0]: best = (key, sorted(val), sorted(trn), s1, s2, fs, fc)
    if best is None: raise SystemExit('no split keeps the trained gene set unchanged')
    return best[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides', required=True, help='CSV: slide, dir, donor, tissue (empty for non-evaluation tissues); '
                                                    'every Stage-2 training slide')
    ap.add_argument('--targets', required=True, help='CSV: tissue, dir — one target slide per evaluation tissue')
    ap.add_argument('--global_genes', required=True, help='head gene table (gene_symbol, global_gene_index)')
    ap.add_argument('--val_frac', type=float, default=0.4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from voice.panel import sym2glob_map
    mapping, _ = sym2glob_map(a.global_genes)
    sl = pd.read_csv(a.slides, dtype=str).fillna(''); tg = pd.read_csv(a.targets, dtype=str)
    if sl.slide.duplicated().any(): raise SystemExit('duplicate slide names')
    dirs = dict(zip(sl.slide, sl.dir)); panel = {s: bits(head_panel(d, mapping)) for s, d in dirs.items()}
    cells = {s: n_cells(d) for s, d in dirs.items()}
    h_full = 0
    for b in panel.values(): h_full |= b
    res = dict(val_frac=a.val_frac, tissues={}, n_trained_genes=h_full.bit_count())
    held = []
    for t, d in zip(tg.tissue, tg.dir):
        sub = sl[sl.tissue == t]
        if len(sub) < 2: raise SystemExit(f'{t}: needs at least two training slides')
        slides = list(sub.slide); groups = {}
        for s, g in zip(sub.slide, sub.donor): groups.setdefault(g or s, []).append(s)
        if len(groups) < 2: raise SystemExit(f'{t}: all slides share one donor; no donor-held-out split exists')
        rest = 0
        for s in sl.slide:
            if s not in set(slides): rest |= panel[s]
        test = bits(head_panel(d, mapping)); full_cov = 0
        for s in slides: full_cov |= panel[s]
        val, trn, s1, s2, fs, fc = choose(groups, slides, panel, cells, rest, h_full, test, a.val_frac)
        held += val
        res['tissues'][t] = dict(target=d, train=trn, val=val, donor_groups=groups,
                                 val_frac_slides=round(fs, 4), val_frac_cells=round(fc, 4),
                                 train_cells=sum(cells[s] for s in trn), val_cells=sum(cells[s] for s in val),
                                 test_genes_in_head=(test & h_full).bit_count(),
                                 test_genes_same_tissue=(test & full_cov).bit_count(),
                                 test_genes_train=s1, test_genes_val=s2)
        print(f'{t}: {len(slides)} slides / {len(groups)} donors -> {len(val)} held out '
              f'({fs:.2f} slides, {fc:.2f} cells); same-tissue test genes all {(test & full_cov).bit_count()} '
              f'train {s1} val {s2}', flush=True)
    stage2 = [s for s in sl.slide if s not in set(held)]
    h_split = 0
    for s in stage2: h_split |= panel[s]
    assert h_split == h_full
    res.update(heldout=sorted(held), n_stage2_slides=len(stage2), cells_all=sum(cells.values()),
               cells_stage2=sum(cells[s] for s in stage2))
    (out / 'split.json').write_text(json.dumps(res, indent=1))
    (out / 'stage2_slides.txt').write_text('\n'.join(stage2) + '\n')
    (out / 'heldout_slides.txt').write_text('\n'.join(sorted(held)) + '\n')
    print(f'Stage-2 corpus {len(stage2)} slides, {res["cells_stage2"]:,} of {res["cells_all"]:,} cells; '
          f'trained genes {h_full.bit_count()} unchanged', flush=True)


if __name__ == '__main__':
    main()
