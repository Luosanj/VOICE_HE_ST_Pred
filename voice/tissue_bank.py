"""Read tissue-pooled, prestandardized reference cells from compressed shards."""
from pathlib import Path
import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = 'voice-tissue-bank-v1'


def is_tissue_bank(directory):
    path = Path(directory) / 'manifest.json'
    return path.is_file() and json.loads(path.read_text()).get('schema') == SCHEMA


class TissueBank:
    def __init__(self, directory, device='cuda', weights_id=None):
        self.root = Path(directory)
        self.meta = json.loads((self.root / 'manifest.json').read_text())
        if self.meta.get('schema') != SCHEMA:
            raise ValueError('Choose a tissue directory containing its manifest.json')
        if weights_id and weights_id != self.meta['weights_id']:
            raise ValueError('The reference bank and query use different model weights')
        if not (self.meta['features_l2_normalized'] and self.meta['expression_prestandardized']):
            raise ValueError('Expected normalized features and prestandardized expression')
        self.device = torch.device(device)
        self.panels = [np.asarray(p, np.int64) for p in self.meta['gene_sets']]
        self.columns = [{int(g): j for j, g in enumerate(p)} for p in self.panels]
        self.parts = []
        self.features = {}
        offset = 0
        try:
            for entry in self.meta['parts']:
                file = self.root / entry['file']
                if Path(entry['file']).name != entry['file']:
                    raise ValueError('Reference shard must be inside the tissue directory')
                handle = h5py.File(file, 'r')
                if handle.attrs.get('schema') != SCHEMA or handle['E'].shape != (entry['n_cells'], 1536):
                    handle.close()
                    raise ValueError('Invalid reference shard')
                groups = {}
                for key, group in handle['gene_sets'].items():
                    pid = int(key)
                    if not np.array_equal(group['genes'][:], self.panels[pid]):
                        raise ValueError('Reference gene identities do not match the manifest')
                    groups[pid] = dict(rows=group['rows'][:], Y=group['Y'])
                self.parts.append(dict(handle=handle, E=handle['E'], groups=groups,
                                       start=offset, end=offset+entry['n_cells']))
                offset += entry['n_cells']
            if offset != self.meta['n_cells']:
                raise ValueError('Reference cell count differs from manifest')
            feature_bytes = offset * 1536 * 2
            # Cache all features when the device has room; otherwise stream shards.
            budget = 1024 ** 3
            if self.device.type == 'cuda':
                free, _ = torch.cuda.mem_get_info(self.device)
                budget = max(0, int((free - 2 * 1024 ** 3) * .7))
            if feature_bytes <= budget:
                for i, part in enumerate(self.parts):
                    self.features[i] = torch.from_numpy(part['E'][:]).to(self.device)
        except Exception:
            self.close()
            raise

    def close(self):
        self.features.clear()
        for part in self.parts:
            part['handle'].close()

    def feature_tensor(self, index):
        if index in self.features:
            return self.features[index]
        return torch.from_numpy(self.parts[index]['E'][:]).to(self.device)

    def expression(self, ids, genes, eligible):
        """Gather prestandardized values in neighbour order, without a second z-score."""
        flat = np.asarray(ids, np.int64).reshape(-1)
        result = np.empty((len(flat), len(genes)), np.float16)
        filled = np.zeros(len(flat), bool)
        for part in self.parts:
            where = np.flatnonzero((flat >= part['start']) & (flat < part['end']))
            if not len(where):
                continue
            local = flat[where] - part['start']
            for pid in eligible:
                if pid not in part['groups']:
                    continue
                group = part['groups'][pid]
                rows = group['rows']
                positions = np.searchsorted(rows, local)
                valid = positions < len(rows)
                valid[valid] &= rows[positions[valid]] == local[valid]
                if not valid.any():
                    continue
                unique, inverse = np.unique(positions[valid], return_inverse=True)
                cols = np.array([self.columns[pid][int(g)] for g in genes])
                lo, hi = int(cols.min()), int(cols.max()) + 1
                values = group['Y'][unique, lo:hi][:, cols-lo]
                result[where[valid]] = values[inverse]
                filled[where[valid]] = True
        if not filled.all():
            raise ValueError('A selected reference cell has no expression for the requested genes')
        return result.reshape(*np.asarray(ids).shape, len(genes))

    @torch.no_grad()
    def retrieve(self, Eq, panel_q, K=200, tau=.03, qchunk=128, verbose=True):
        if K < 1 or tau <= 0:
            raise ValueError('K and tau must be positive')
        Eq = np.asarray(Eq, np.float32)
        panel_q = np.asarray(panel_q, np.int64)
        if Eq.ndim != 2 or Eq.shape[1] != 1536 or not np.isfinite(Eq).all():
            raise ValueError('Expected finite [cells,1536] query features')
        signatures = {}
        for qi, gene in enumerate(panel_q):
            eligible = tuple(i for i, columns in enumerate(self.columns) if int(gene) in columns)
            if eligible:
                signatures.setdefault(eligible, []).append(qi)
        needed = set(pid for eligible in signatures for pid in eligible)
        covered = np.array(sorted(qi for group in signatures.values() for qi in group), np.int64)
        output = np.full((len(Eq), len(panel_q)), np.nan, np.float32)
        for start in range(0, len(Eq), qchunk):
            stop = min(start+qchunk, len(Eq))
            queries = F.normalize(torch.from_numpy(np.ascontiguousarray(Eq[start:stop])), dim=1).half().to(self.device)
            candidates = {}
            for part_id, part in enumerate(self.parts):
                features = self.feature_tensor(part_id)
                scores = queries @ features.T
                for pid, group in part['groups'].items():
                    if pid not in needed:
                        continue
                    rows = torch.as_tensor(group['rows'], device=self.device)
                    distances, indices = torch.topk(scores[:, rows], min(K, len(rows)), dim=1)
                    indices = rows[indices] + part['start']
                    if pid in candidates:
                        old_d, old_i = candidates[pid]
                        all_d = torch.cat([old_d, distances], 1)
                        all_i = torch.cat([old_i, indices], 1)
                        distances, take = torch.topk(all_d, min(K, all_d.shape[1]), dim=1)
                        indices = torch.gather(all_i, 1, take)
                    candidates[pid] = distances, indices
                del features, scores
            for eligible, positions in signatures.items():
                distances = torch.cat([candidates[pid][0] for pid in eligible], 1)
                indices = torch.cat([candidates[pid][1] for pid in eligible], 1)
                distances, take = torch.topk(distances, min(K, distances.shape[1]), dim=1)
                indices = torch.gather(indices, 1, take).cpu().numpy()
                weights = torch.softmax(distances.float()/tau, dim=1)
                for g0 in range(0, len(positions), 128):
                    cols = positions[g0:g0+128]
                    values = self.expression(indices, panel_q[cols], eligible)
                    values = torch.from_numpy(values).to(self.device).float()
                    prediction = torch.einsum('nk,nkg->ng', weights, values)
                    output[start:stop, cols] = prediction.cpu().numpy()
            if verbose:
                print(f'[R] {stop:,}/{len(Eq):,} cells; tissue={self.meta["tissue"]}', flush=True)
        return output, covered


def crossR(Eq, panel_q, directory, K=200, tau=.03, weights_id=None, device='cuda',
           verbose=True, names=None, exclude=None):
    if names is not None or exclude:
        raise ValueError('The pooled tissue bank has no source identities for slide-level selection or exclusion')
    bank = TissueBank(directory, device, weights_id)
    try:
        return bank.retrieve(Eq, panel_q, K, tau, verbose=verbose)
    finally:
        bank.close()
