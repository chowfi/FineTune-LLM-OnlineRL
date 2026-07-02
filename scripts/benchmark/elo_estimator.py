"""Bradley-Terry MLE with Rao-Kupper draws + bootstrap CIs.

Game outcomes are encoded from White's perspective as ``"win" | "loss" | "draw"``.
Each game is a dict::

    {"white": <player_id>, "black": <player_id>, "result": "win"|"loss"|"draw"}

Rao-Kupper model::

    log_diff = (R_w - R_b) * ln(10) / 400              # = ln(gamma_w / gamma_b)
    P(W wins) = 1 / (1 + theta * exp(-log_diff))
    P(B wins) = 1 / (1 + theta * exp(+log_diff))
    P(draw)   = 1 - P(W wins) - P(B wins)               # = 0 iff theta == 1

theta >= 1 controls draw frequency; theta == 1 reduces to plain Bradley-Terry.

Fitting:
- ``fit_ratings(games, fixed={...})`` jointly optimises the free ratings and
  log(theta) under the L-BFGS-B optimiser. Returns ``(ratings_dict, theta,
  nll)``.
- ``bootstrap_ci(games, fixed, target_player, n_reps)`` resamples games with
  replacement (case bootstrap), refits, returns ``(lo, hi)`` for the
  ``target_player``'s Elo at the 2.5 / 97.5 percentiles.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


LN10 = math.log(10.0)


def _result_signs(result: str) -> Tuple[bool, bool, bool]:
    r = result.lower()
    return r == "win", r == "loss", r == "draw"


def _neg_log_likelihood(
    free_params: np.ndarray,
    games: List[Dict[str, str]],
    free_player_idx: Dict[str, int],
    fixed_ratings: Dict[str, float],
) -> float:
    """``free_params`` = [<ratings for free players>..., log_theta_minus_zero]."""
    log_theta = float(free_params[-1])
    theta = math.exp(log_theta) if log_theta >= 0.0 else 1.0
    nll = 0.0
    for g in games:
        w_id = g["white"]
        b_id = g["black"]
        rw = (
            float(free_params[free_player_idx[w_id]])
            if w_id in free_player_idx
            else float(fixed_ratings[w_id])
        )
        rb = (
            float(free_params[free_player_idx[b_id]])
            if b_id in free_player_idx
            else float(fixed_ratings[b_id])
        )
        log_diff = (rw - rb) * LN10 / 400.0

        if g["result"] == "win":
            nll += math.log1p(theta * math.exp(-log_diff))
        elif g["result"] == "loss":
            nll += math.log1p(theta * math.exp(log_diff))
        else:
            # draw
            p_w = 1.0 / (1.0 + theta * math.exp(-log_diff))
            p_b = 1.0 / (1.0 + theta * math.exp(log_diff))
            p_d = max(1.0 - p_w - p_b, 1e-12)
            nll -= math.log(p_d)
    # Soft prior on log_theta to keep theta in [1, ~5] range (helps when there
    # are very few draws and the likelihood is flat for large theta).
    nll += 0.5 * (log_theta - math.log(1.5)) ** 2 / (1.0**2)
    return nll


def fit_ratings(
    games: List[Dict[str, str]],
    *,
    fixed_ratings: Optional[Dict[str, float]] = None,
    initial_ratings: Optional[Dict[str, float]] = None,
    initial_theta: float = 1.5,
    max_iter: int = 200,
) -> Tuple[Dict[str, float], float, float]:
    """Return ``(ratings_dict, theta, final_nll)``.

    All players appearing in ``games`` are rated. Players present in
    ``fixed_ratings`` are anchored (not optimised). At least one player must
    be fixed, otherwise ratings are only identified up to a shift.
    """
    fixed_ratings = dict(fixed_ratings or {})
    initial_ratings = dict(initial_ratings or {})
    if not games:
        return dict(fixed_ratings), initial_theta, 0.0

    all_players: List[str] = []
    seen = set()
    for g in games:
        for key in ("white", "black"):
            pid = g[key]
            if pid not in seen:
                seen.add(pid)
                all_players.append(pid)

    free_players = [p for p in all_players if p not in fixed_ratings]
    if not free_players:
        # Nothing to fit; just compute the NLL at the fixed ratings + initial theta.
        x = np.array([math.log(initial_theta)], dtype=float)
        free_idx: Dict[str, int] = {}
        return (
            dict(fixed_ratings),
            initial_theta,
            _neg_log_likelihood(x, games, free_idx, fixed_ratings),
        )

    free_player_idx = {p: i for i, p in enumerate(free_players)}
    n_free = len(free_players)

    x0 = np.zeros(n_free + 1, dtype=float)
    for p, i in free_player_idx.items():
        x0[i] = float(initial_ratings.get(p, 1500.0))
    x0[-1] = math.log(max(1.0, float(initial_theta)))

    bounds = [(-3000.0, 5000.0)] * n_free + [(math.log(1.0), math.log(10.0))]

    res = minimize(
        _neg_log_likelihood,
        x0=x0,
        args=(games, free_player_idx, fixed_ratings),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )

    fitted = dict(fixed_ratings)
    for p, i in free_player_idx.items():
        fitted[p] = float(res.x[i])
    theta = float(math.exp(res.x[-1]))
    return fitted, theta, float(res.fun)


def bootstrap_ci(
    games: List[Dict[str, str]],
    *,
    fixed_ratings: Dict[str, float],
    target_player: str,
    n_reps: int = 500,
    seed: int = 0,
    initial_theta: float = 1.5,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Case-bootstrap CI for ``target_player``'s Elo.

    Returns ``{"point": <Elo>, "lo": <Elo>, "hi": <Elo>, "theta": <theta>,
    "n_reps": <int>}``.
    """
    point_ratings, point_theta, _ = fit_ratings(
        games,
        fixed_ratings=fixed_ratings,
        initial_theta=initial_theta,
    )
    if target_player not in point_ratings:
        raise KeyError(f"{target_player} not in fitted ratings")

    rng = np.random.default_rng(int(seed))
    n = len(games)
    samples: List[float] = []
    for _ in range(int(n_reps)):
        idx = rng.integers(0, n, size=n)
        resample = [games[int(i)] for i in idx]
        try:
            rs, _, _ = fit_ratings(
                resample,
                fixed_ratings=fixed_ratings,
                initial_ratings=point_ratings,
                initial_theta=point_theta,
            )
            if target_player in rs:
                samples.append(rs[target_player])
        except Exception:
            continue

    if not samples:
        return {
            "point": point_ratings[target_player],
            "lo": float("nan"),
            "hi": float("nan"),
            "theta": point_theta,
            "n_reps": 0,
        }
    arr = np.asarray(samples, dtype=float)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "point": point_ratings[target_player],
        "lo": float(np.quantile(arr, alpha)),
        "hi": float(np.quantile(arr, 1.0 - alpha)),
        "theta": point_theta,
        "n_reps": int(arr.size),
    }


def predicted_score(rating_w: float, rating_b: float, theta: float = 1.5) -> float:
    """Expected score for White (W=1, D=0.5, L=0)."""
    log_diff = (rating_w - rating_b) * LN10 / 400.0
    p_w = 1.0 / (1.0 + theta * math.exp(-log_diff))
    p_b = 1.0 / (1.0 + theta * math.exp(log_diff))
    p_d = max(1.0 - p_w - p_b, 0.0)
    return p_w + 0.5 * p_d


def games_from_results(
    results: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Pass-through for storage symmetry / cheap type-check."""
    out: List[Dict[str, str]] = []
    for r in results:
        if "white" not in r or "black" not in r or "result" not in r:
            raise ValueError(f"bad game row: {r!r}")
        if r["result"] not in {"win", "loss", "draw"}:
            raise ValueError(f"unknown result: {r['result']!r}")
        out.append({"white": r["white"], "black": r["black"], "result": r["result"]})
    return out


__all__ = [
    "fit_ratings",
    "bootstrap_ci",
    "predicted_score",
    "games_from_results",
]
