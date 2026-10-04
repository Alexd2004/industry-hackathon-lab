"""Review policy (Tier 2, step 9): turns stack scores into three bands.

verify: score >= t_verify, sent to a human. t_verify is the cutoff that keeps false-teen at or below
        cap - margin on the audit slice (metrics.cap_threshold: the conservative (1 - cap) quantile of
        adult scores, so ties never push the rate over the cap).
soft:   t_soft <= score < t_verify, softer action. t_soft is the (1 - soft_recall) quantile of teen
        scores (method="lower"), so the soft band and above reach at least soft_recall of teens.
none:   below t_soft.

Thresholds come from honest out-of-fold scores only (cache/stack_oof.csv, written by stack.py), never
from test. Decisions: the cap wins over review_budget (the verify band is never truncated, budget_binding
reports when it is over budget); the 8-30% cap clamp lives in loop.py and A2, not here, because the
ladder rows use 5% and 10%. The A1 PSI key is added to policy.yaml in step 11.

Run: python -m softsignal.policy
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from softsignal.data import ROOT, load_data
from softsignal.features import ID_COL, TARGET
from softsignal.metrics import DEFAULT_CAP, as_binary, cap_threshold, prf
from softsignal.stack import STACK_OOF, Stack

POLICY_FILE = ROOT / "policy.yaml"
POLICY_KEYS = {
    "cap_false_teen": float,
    "review_budget": float,
    "soft_recall": float,
    "min_audit_adults": int,
    "audit_per_batch": int,
}
MIN_CLASS = 5  # fewer audit adults (or teens) than this: keep the prior threshold
BANDS = ("verify", "soft", "none")


def load_policy(path: Path = POLICY_FILE) -> dict:
    """policy.yaml as a dict. Raises on missing or unknown keys, wrong types or out-of-range values."""
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a mapping of policy keys")
    if set(raw) != set(POLICY_KEYS):
        raise ValueError(f"{path} keys differ: missing {sorted(set(POLICY_KEYS) - set(raw))}, "
                         f"unknown {sorted(set(raw) - set(POLICY_KEYS))}")
    out = {}
    for key, kind in POLICY_KEYS.items():
        v = raw[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and v != int(v)):
            raise ValueError(f"{key} must be a {kind.__name__}, got {v!r}")
        out[key] = kind(v)
    for key in ("cap_false_teen", "review_budget", "soft_recall"):
        if not 0.0 <= out[key] <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1, got {out[key]}")
    for key in ("min_audit_adults", "audit_per_batch"):
        if out[key] < 1:
            raise ValueError(f"{key} must be at least 1, got {out[key]}")
    return out


@dataclass(frozen=True)
class Thresholds:
    """cap is the nominal cap; the verify cutoff targets cap - margin. flags are strings (see pick_thresholds)."""

    t_verify: float
    t_soft: float
    cap: float
    margin: float
    n_adults: int
    n_teens: int
    flags: tuple[str, ...] = ()


def load_audit_slice(train: pd.DataFrame, path: Path = STACK_OOF) -> pd.DataFrame:
    """Train labels joined to their nested-OOF stack scores: columns ID_COL, TARGET, stack_oof.

    The static audit slice is the whole train set. Raises if the cache does not cover exactly
    the train ids (a stale cache).
    """
    oof = pd.read_csv(path, dtype={ID_COL: str})
    if oof[ID_COL].duplicated().any() or set(oof[ID_COL]) != set(train[ID_COL].astype(str)):
        raise ValueError(f"{path} does not match the train ids: rerun python -m softsignal.stack")
    ids = train[[ID_COL, TARGET]].astype({ID_COL: str})
    return ids.merge(oof, on=ID_COL, how="left", validate="one_to_one")


def pick_thresholds(
    scores,
    y,
    cap: float = DEFAULT_CAP,
    soft_recall: float = 0.90,
    margin: float = 0.0,
    prior: Thresholds | None = None,
) -> Thresholds:
    """t_verify and t_soft from out-of-fold scores and revealed labels (teen = 1).

    margin aims the verify cutoff at cap - margin ("cap minus 1 pt" is margin=0.01).
    Flags: insufficient_adults (fewer than MIN_CLASS adults: both thresholds are the prior's, and
    prior is required); insufficient_teens (t_soft is the prior's, else t_verify); soft_capped
    (t_soft would sit above t_verify, so it is lowered to t_verify and the soft band is empty).
    """
    scores, y = np.asarray(scores, dtype=float), as_binary(y)
    if scores.shape != y.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {y.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite (no NaN or inf)")
    if not 0.0 <= cap <= 1.0 or not 0.0 <= margin <= cap:
        raise ValueError(f"need 0 <= margin <= cap <= 1, got cap={cap}, margin={margin}")
    if not 0.0 < soft_recall <= 1.0:
        raise ValueError("soft_recall must be in (0, 1]")
    adults, teens = scores[y == 0], scores[y == 1]
    n_a, n_t = len(adults), len(teens)
    if n_a < MIN_CLASS:
        if prior is None:
            raise ValueError(f"only {n_a} audit adults and no prior thresholds to keep")
        return Thresholds(prior.t_verify, prior.t_soft, cap, margin, n_a, n_t, ("insufficient_adults",))
    flags = []
    t_verify = cap_threshold(scores, y, cap - margin)
    if n_t < MIN_CLASS:
        flags.append("insufficient_teens")
        t_soft = prior.t_soft if prior is not None else t_verify
    else:
        t_soft = float(np.quantile(teens, 1.0 - soft_recall, method="lower"))
    if t_soft > t_verify:
        flags.append("soft_capped")
        t_soft = t_verify
    return Thresholds(t_verify, t_soft, cap, margin, n_a, n_t, tuple(flags))


def assign_bands(scores, th: Thresholds) -> np.ndarray:
    """'verify' / 'soft' / 'none' per account. A NaN score (a new account with no score yet) is 'soft'."""
    s = np.asarray(scores, dtype=float)
    bands = np.where(s >= th.t_verify, "verify", np.where(s >= th.t_soft, "soft", "none"))
    return np.where(np.isnan(s), "soft", bands)


def band_summary(bands, review_budget: float, y=None) -> dict:
    """Counts per band and the budget flag. With labels (teen = 1) also false-teen and recall at the verify
    band and at verify + soft. budget_binding is True when the verify band is over review_budget of the batch."""
    bands = np.asarray(bands)
    n = len(bands)
    out = {f"n_{b}": int((bands == b).sum()) for b in BANDS}
    out["verify_share"] = out["n_verify"] / n if n else 0.0
    out["budget_binding"] = bool(out["verify_share"] > review_budget)
    if y is not None:
        y = as_binary(y)
        for name, mask in (("verify", bands == "verify"), ("soft_up", bands != "none")):
            _, rec, ft, _ = prf(y, mask.astype(int))
            out[f"rec_{name}"], out[f"ft_{name}"] = rec, ft
    return out


def main() -> None:
    pol = load_policy()
    train, test = load_data(on_param_mismatch="error")
    audit = load_audit_slice(train)
    oof, y = audit["stack_oof"].to_numpy(), audit[TARGET].to_numpy()
    p_te, y_te = Stack.fit(train).score(test), test[TARGET].to_numpy()
    print(f"audit slice: {len(audit)} train accounts (nested OOF), review_budget {pol['review_budget']:.0%}, "
          f"soft_recall {pol['soft_recall']:.0%}")
    rows = []
    for cap in (pol["cap_false_teen"], 0.10, 0.05):
        th = pick_thresholds(oof, y, cap=cap, soft_recall=pol["soft_recall"])
        for name, s, yy in (("oof", oof, y), ("test", p_te, y_te)):
            rows.append({"cap": cap, "t_verify": th.t_verify, "t_soft": th.t_soft, "set": name,
                         **band_summary(assign_bands(s, th), pol["review_budget"], yy), "flags": ",".join(th.flags)})
    pd.set_option("display.width", 200)
    cols = ["cap", "set", "t_verify", "t_soft", "ft_verify", "rec_verify", "rec_soft_up", "verify_share",
            "budget_binding", "flags"]
    print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))
    print("OOF rows meet the cap by construction. Test applies the OOF cutoff to the full-train stack, so its")
    print("false-teen can land above the cap (step 8 measured 0.176 at 15%). WP targets (to re-measure, different")
    print("split): recall 92% at 15%, 88% at 10%, 74% at 6%.")


if __name__ == "__main__":
    main()
