"""Drift detectors: single-test baselines and the proposed multi-signal engine.

All detectors are label-free except ErrorRateDetector (supervised reference).
Each detector exposes .fit(ref_X, ref_proba) and .check(X, proba) -> dict(alarm=bool, ...).
"""
import numpy as np
from scipy.stats import ks_2samp

EPS = 1e-4


def psi(ref, cur, bins=10):
    """Population Stability Index with quantile bins taken from the reference."""
    ref = np.asarray(ref, float); cur = np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        edges = np.linspace(ref.min() - EPS, ref.max() + EPS, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + EPS
    c = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def confidence(proba):
    return np.abs(np.asarray(proba) - 0.5) * 2  # 0 = unsure, 1 = certain (binary)


class _Base:
    name = "base"

    def fit(self, ref_X, ref_proba, ref_acc=None):
        self.ref_X = np.asarray(ref_X, float)
        self.ref_p = np.asarray(ref_proba, float)
        self.ref_acc = ref_acc
        return self


class KSDetector(_Base):
    """Per-feature two-sample KS test, Bonferroni-corrected (standard practice, cf. [P8])."""
    name = "KS"

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def check(self, X, proba, acc=None):
        X = np.asarray(X, float); d = X.shape[1]
        p = [ks_2samp(self.ref_X[:, j], X[:, j]).pvalue for j in range(d)]
        return {"alarm": bool(min(p) < self.alpha / d), "min_p": float(min(p))}


class PSIDetector(_Base):
    """Alarm when any feature PSI exceeds the common 0.2 industry threshold."""
    name = "PSI"

    def __init__(self, thr=0.2):
        self.thr = thr

    def check(self, X, proba, acc=None):
        X = np.asarray(X, float)
        v = [psi(self.ref_X[:, j], X[:, j]) for j in range(X.shape[1])]
        return {"alarm": bool(max(v) > self.thr), "max_psi": float(max(v))}


class ErrorRateDetector(_Base):
    """Supervised reference: alarm if window accuracy falls `drop` below post-training accuracy.
    Needs true labels at check time, which production rarely has."""
    name = "ErrorRate (labels)"

    def __init__(self, drop=0.05):
        self.drop = drop

    def check(self, X, proba, acc=None):
        return {"alarm": bool(acc is not None and acc < self.ref_acc - self.drop)}


class MultiSignalDetector(_Base):
    """Proposed engine. Three model-agnostic, label-free signals:
      S_in   : >= `min_feat_share` of features show KS-significant AND PSI > psi_thr shift
      S_pred : PSI of the model's predicted-probability distribution > pred_thr
      S_conf : mean prediction confidence drops by > conf_drop vs reference
    A window "votes" drift when >= `votes` signals fire; the alarm needs `persist`
    consecutive drift windows. S_in alone can never trigger (votes>=2), so pure
    feature drift that the model ignores is suppressed.
    """
    name = "Multi-signal"

    def __init__(self, psi_thr=0.2, pred_thr=0.1, conf_drop=0.05, min_feat_share=0.25,
                 votes=2, persist=2, alpha=0.05):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self._streak = 0

    def fit(self, ref_X, ref_proba, ref_acc=None):
        super().fit(ref_X, ref_proba, ref_acc)
        self.ref_conf = confidence(self.ref_p).mean()
        self._streak = 0
        return self

    def signals(self, X, proba):
        X = np.asarray(X, float); d = X.shape[1]
        shifted = 0
        for j in range(d):
            if (ks_2samp(self.ref_X[:, j], X[:, j]).pvalue < self.alpha / d
                    and psi(self.ref_X[:, j], X[:, j]) > self.psi_thr):
                shifted += 1
        p_psi = psi(self.ref_p, proba)
        c_drop = self.ref_conf - confidence(proba).mean()
        return {"S_in": shifted / d >= self.min_feat_share, "S_pred": p_psi > self.pred_thr,
                "S_conf": c_drop > self.conf_drop, "feat_share": shifted / d,
                "pred_psi": p_psi, "conf_drop": float(c_drop)}

    def check(self, X, proba, acc=None):
        s = self.signals(X, proba)
        n = s["S_in"] + s["S_pred"] + s["S_conf"]
        self._streak = self._streak + 1 if n >= self.votes else 0
        s.update(votes=int(n), alarm=self._streak >= self.persist)
        return s
