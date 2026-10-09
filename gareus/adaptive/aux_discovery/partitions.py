"""Discovery / evaluation partitions (c10 diagnose.py + eval_partition.py, prereg_v2 values).
Bootstraps keep lineage multiplicity (c10 bootstrap_fix.py)."""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor

from .settings import AuxDiscoverySettings


def preprocess(X, families, train, s: AuxDiscoverySettings):
    X = np.asarray(X, dtype=np.float64)
    mean = X[train].mean(0); sd_raw = X[train].std(0)
    keep = sd_raw >= s.sd_drop
    sd = np.maximum(sd_raw, s.sd_floor)  # exactly what Z used; FrozenPartition stores this
    Z = (X[:, keep] - mean[keep]) / sd[keep]
    fam = np.asarray(families)[keep]
    for f in np.unique(fam):
        Z[:, fam == f] /= np.sqrt((fam == f).sum())
    return Z, keep, mean, sd


def standardise_s(cv, train):
    cv = np.asarray(cv, dtype=np.float32)
    return (cv - cv[train].mean(0)) / cv[train].std(0)


def knn_residual(S, Z, train, k):
    knn = KNeighborsRegressor(n_neighbors=int(k)).fit(S[train], Z[train])
    M = np.empty_like(Z)
    for a in range(0, len(Z), 4000):
        M[a:a + 4000] = knn.predict(S[a:a + 4000])
    return Z - M, knn


def pca_project(R, train, var, cap):
    pca = PCA().fit(R[train])
    nc = int(min(cap, np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var) + 1))
    return pca.transform(R)[:, :nc], pca, nc


def s_bins(S, train, n):
    edges = [np.quantile(S[train, a], np.linspace(0, 1, n + 1)[1:-1]) for a in range(2)]
    return np.digitize(S[:, 0], edges[0]) * n + np.digitize(S[:, 1], edges[1])


def lineage_bootstrap_rows(lineage_of_rows, rng) -> np.ndarray:
    lins, inv = np.unique(lineage_of_rows, return_inverse=True)
    rows_by = [np.nonzero(inv == i)[0] for i in range(len(lins))]
    pick = rng.integers(0, len(lins), len(lins))
    return np.concatenate([rows_by[i] for i in pick])


@dataclass
class PartitionChoice:
    k: Optional[int]
    gmm: object = None
    labels: Optional[np.ndarray] = None
    table: List[dict] = field(default_factory=list)


def choose_partition(Pm, train, holdout, lineage, s: AuxDiscoverySettings, seed) -> PartitionChoice:
    rng = np.random.default_rng(int(seed))
    tr_rows = np.nonzero(train)[0]
    table, best = [], None
    for k in range(int(s.k_min), int(s.k_max) + 1):
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0, reg_covar=1e-6).fit(Pm[train])
        lab_v = g.predict(Pm[holdout])
        refit = GaussianMixture(k, covariance_type="full", n_init=3, random_state=1, reg_covar=1e-6).fit(Pm[holdout])
        ari_val = float(adjusted_rand_score(lab_v, refit.predict(Pm[holdout])))
        boots = []
        for r in range(int(s.n_boot_partition)):
            rows = tr_rows[lineage_bootstrap_rows(lineage[train], rng)]
            gb = GaussianMixture(k, covariance_type="full", n_init=1, random_state=10 + r, reg_covar=1e-6).fit(Pm[rows])
            boots.append(adjusted_rand_score(lab_v, gb.predict(Pm[holdout])))
        share = np.bincount(lab_v, minlength=k) / max(1, lab_v.size)
        n_ok = int((share >= s.group_min_share).sum())
        row = {"k": k, "ari_val": round(ari_val, 4), "boot_median": round(float(np.median(boots)), 4),
               "n_groups_ok": n_ok}
        table.append(row)
        if ari_val >= s.ari_min and np.median(boots) >= s.ari_min and n_ok >= 2:
            best = (k, g)
    if best is None:
        return PartitionChoice(None, None, None, table)
    k, g = best
    return PartitionChoice(k, g, g.predict(Pm), table)


def _cond_table(y, x, K, nx, alpha=1.0):
    t = np.full((nx, K), float(alpha))
    np.add.at(t, (x, y), 1.0)
    return t / t.sum(1, keepdims=True)


def _ll(p, y, x=None):
    x = np.zeros_like(y) if x is None else x
    return float(np.mean(np.log(p[x, y])))


def hidden_fraction(lab, bins, train, holdout, K, nbins) -> float:
    pm = _cond_table(lab[train], np.zeros(train.sum(), int), K, 1)
    ll_marg = _ll(pm, lab[holdout])
    ll_s = _ll(_cond_table(lab[train], bins[train], K, nbins), lab[holdout], bins[holdout])
    return float(1.0 - (ll_s - ll_marg) / (-ll_marg)) if ll_marg < 0 else 0.0


def co_occurrence(lab, bins, holdout, K, s: AuxDiscoverySettings) -> List[dict]:
    out = []
    lv, bv = lab[holdout], bins[holdout]
    for b in np.unique(bv):
        m = bv == b
        if m.mean() < s.cooc_bin_min:
            continue
        share = np.bincount(lv[m], minlength=K) / m.sum()
        groups = np.nonzero(share >= s.cooc_share)[0]
        if groups.size >= 2:
            out.append({"bin": int(b), "groups": groups.tolist(), "share": share[groups].round(4).tolist()})
    return out


def lineage_info(lab, bins, lineage, step, K, nbins, c) -> float:
    lin_id = np.unique(lineage, return_inverse=True)[1]
    order = np.lexsort((step, lin_id))
    lin_s, lab_s, b_s = lin_id[order], lab[order], bins[order]
    n = lin_s.size
    start = np.searchsorted(lin_s, lin_s, side="left")
    size = np.searchsorted(lin_s, lin_s, side="right") - start
    first = (np.arange(n) - start) < size // 2
    A, Tm = first, ~first
    if not Tm.any():
        return 0.0
    pA = _cond_table(lab_s[A], b_s[A], K, nbins)
    keyA = lin_s[A].astype(np.int64) * nbins + b_s[A]
    ukeys, inv = np.unique(keyA, return_inverse=True)
    cnt = np.zeros((ukeys.size, K))
    np.add.at(cnt, (inv, lab_s[A]), 1.0)
    keyT = lin_s[Tm].astype(np.int64) * nbins + b_s[Tm]
    pos = np.minimum(np.searchsorted(ukeys, keyT), max(ukeys.size - 1, 0))
    hit = (ukeys[pos] == keyT) if ukeys.size else np.zeros(keyT.size, bool)
    nB = np.zeros((keyT.size, K))
    if ukeys.size:
        nB[hit] = cnt[pos[hit]]
    yT, bT = lab_s[Tm], b_s[Tm]
    pB = (nB + c * pA[bT]) / (nB.sum(1, keepdims=True) + c)
    lt = np.log(pB[np.arange(yT.size), yT])
    return float(np.mean(lt) - _ll(pA, yT, bT))


def info_gain(z, y, bins, train, holdout, K, nbins) -> float:
    u = (z - z[train].mean()) / (z[train].std() + 1e-12)
    onehot = np.eye(nbins)[bins]
    base = onehot; full = np.c_[onehot, u, u ** 2, u ** 3]

    def _heldout_ll(F):
        m = LogisticRegression(max_iter=3000, C=1.0).fit(F[train], y[train])
        pr = np.full((holdout.sum(), K), 1e-12)
        pr[:, m.classes_] = np.maximum(m.predict_proba(F[holdout]), 1e-12)
        return float(np.mean(np.log(pr[np.arange(holdout.sum()), y[holdout]])))
    return _heldout_ll(full) - _heldout_ll(base)


@dataclass
class FrozenPartition:
    feature_names: List[str]
    keep: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    families: List[str]
    s_mean: np.ndarray
    s_sd: np.ndarray
    knn: object
    pca: object
    n_components: int
    gmm: object

    def predict(self, X_raw, cv) -> np.ndarray:
        X = np.asarray(X_raw, dtype=np.float64)
        Z = (X[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        fam = np.asarray(self.families)[self.keep]
        for f in np.unique(fam):
            Z[:, fam == f] /= np.sqrt((fam == f).sum())
        S = (np.asarray(cv, np.float32) - self.s_mean) / self.s_sd
        R = Z - self.knn.predict(S)
        return self.gmm.predict(self.pca.transform(R)[:, : self.n_components])

    def to_file(self, path: Path) -> str:
        data = pickle.dumps(self, protocol=4)
        Path(path).write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        Path(str(path) + ".sha256").write_text(sha + "\n")
        return sha

    @staticmethod
    def from_file(path: Path) -> "FrozenPartition":
        data = Path(path).read_bytes()
        sha = Path(str(path) + ".sha256").read_text().strip()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"{path}: sha256 mismatch")
        return pickle.loads(data)


@dataclass
class PartitionResult:
    status: str
    choice: PartitionChoice
    frozen: Optional[FrozenPartition]
    bins: np.ndarray
    hidden_fraction: Optional[float] = None
    co_occurrence: List[dict] = field(default_factory=list)
    lineage_info: Optional[float] = None


def fit_partition(X, families, cv, train, holdout, lineage, step, s: AuxDiscoverySettings, *, seed: int = 0,
                  feature_names: Optional[Sequence[str]] = None) -> PartitionResult:
    Z, keep, mean, sd = preprocess(X, families, train, s)
    S = standardise_s(cv, train)
    R, knn = knn_residual(S, Z, train, s.knn_k)
    Pm, pca, nc = pca_project(R, train, s.pca_var, s.pca_max)
    bins = s_bins(S, train, s.s_bins)
    choice = choose_partition(Pm, train, holdout, lineage, s, seed)
    if choice.k is None:
        return PartitionResult("insufficient_evidence", choice, None, bins)
    K, nb = choice.k, s.s_bins ** 2
    hf = hidden_fraction(choice.labels, bins, train, holdout, K, nb)
    co = co_occurrence(choice.labels, bins, holdout, K, s)
    li = lineage_info(choice.labels, bins, lineage, step, K, nb, s.lineage_dirichlet_c)
    s_mean = np.asarray(cv, np.float32)[train].mean(0); s_sd = np.asarray(cv, np.float32)[train].std(0)
    frozen = FrozenPartition(list(feature_names or [f"f{i}" for i in range(np.asarray(X).shape[1])]), keep, mean,
                             sd, list(families), s_mean, s_sd,
                             knn, pca, nc, choice.gmm)
    triggered = (hf >= s.hidden_min and bool(co)) or li >= s.lineage_info_min
    return PartitionResult("ok" if triggered else "keep", choice, frozen, bins, hf, co, li)
