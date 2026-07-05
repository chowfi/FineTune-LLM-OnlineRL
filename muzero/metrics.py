"""wandb logging + per-loop aggregation of self-play game summaries."""

from __future__ import annotations

import numpy as np

from muzero.config import MuZeroConfig


def aggregate_game_summaries(summaries: list) -> dict:
    n = max(len(summaries), 1)
    wins = sum(1 for s in summaries if s["ally_won"])
    draws = sum(1 for s in summaries if s["draw"])
    promotions = sum(1 for s in summaries if s["promoted"])

    root_entropies = [s.get("mean_root_entropy") for s in summaries]
    root_entropies = [v for v in root_entropies if v is not None]

    ally_cp_aucs = [s.get("mean_ally_cp") for s in summaries]
    ally_cp_aucs = [v for v in ally_cp_aucs if v is not None]

    values, cps = [], []
    for s in summaries:
        for v, cp in s.get("value_cp_pairs") or []:
            values.append(v)
            cps.append(cp)
    correlation = 0.0
    if len(values) >= 2 and np.std(values) > 0 and np.std(cps) > 0:
        corr = np.corrcoef(values, cps)[0, 1]
        correlation = float(corr) if np.isfinite(corr) else 0.0

    total_blunders = sum(s.get("blunders", 0) for s in summaries)
    total_cp_moves = sum(s.get("cp_moves", 0) for s in summaries)
    search_kls = [s.get("mean_search_kl") for s in summaries]
    search_kls = [v for v in search_kls if v is not None]

    return {
        "selfplay/win_rate": wins / n,
        "selfplay/draw_rate": draws / n,
        "selfplay/loss_rate": (len(summaries) - wins - draws) / n,
        "selfplay/repetition_draw_rate": sum(
            1 for s in summaries if s["result"] == "draw_repetition"
        )
        / n,
        "selfplay/truncation_rate": sum(1 for s in summaries if s["truncated"]) / n,
        "selfplay/mean_plies": sum(s["plies"] for s in summaries) / n,
        "selfplay/promotions": promotions,
        "selfplay/era": max((s["era"] for s in summaries), default=0),
        "selfplay/games": len(summaries),
        "selfplay/mean_root_entropy": (
            float(np.mean(root_entropies)) if root_entropies else 0.0
        ),
        "selfplay/mean_ally_cp_auc": (
            float(np.mean(ally_cp_aucs)) if ally_cp_aucs else 0.0
        ),
        "selfplay/value_cp_correlation": correlation,
        "selfplay/red_win_rate": sum(1 for s in summaries if s["result"] == "red_win")
        / n,
        "selfplay/black_win_rate": sum(
            1 for s in summaries if s["result"] == "black_win"
        )
        / n,
        # Move quality: fraction of moves whose mover-perspective eval dropped
        # past the blunder threshold. Should decline steadily if learning.
        "selfplay/blunder_rate": total_blunders / max(total_cp_moves, 1),
        # Wins by actual checkmate (not adjudication) — the "learned to
        # convert" signal that predicts repetition draws falling.
        "selfplay/mate_win_rate": sum(
            1
            for s in summaries
            if s["result"] in ("red_win", "black_win") and not s["truncated"]
        )
        / n,
        # KL(root visits || raw prior): how much search improves on the raw
        # policy. Drifts down as the policy internalizes the search.
        "selfplay/mean_search_kl": (float(np.mean(search_kls)) if search_kls else 0.0),
    }


class MetricsLogger:
    def __init__(self, cfg: MuZeroConfig, enabled: bool = True):
        self.enabled = enabled
        self.wandb = None
        if enabled:
            import wandb

            self.wandb = wandb
            wandb.init(project=cfg.wandb_project, config=vars(cfg))

    def log(self, metrics: dict, step: int):
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
