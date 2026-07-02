"""wandb logging + per-loop aggregation of self-play game summaries."""

from __future__ import annotations

from muzero.config import MuZeroConfig


def aggregate_game_summaries(summaries: list) -> dict:
    n = max(len(summaries), 1)
    wins = sum(1 for s in summaries if s["ally_won"])
    draws = sum(1 for s in summaries if s["draw"])
    ally_cps = [
        (s["final_red_cp"] if s["ally_side"] == "w" else -s["final_red_cp"])
        for s in summaries
        if s["final_red_cp"] is not None
    ]
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
        "selfplay/mean_final_ally_cp": (
            sum(ally_cps) / len(ally_cps) if ally_cps else 0.0
        ),
        "selfplay/promotions": sum(1 for s in summaries if s["promoted"]),
        "selfplay/era": max((s["era"] for s in summaries), default=0),
        "selfplay/games": len(summaries),
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
