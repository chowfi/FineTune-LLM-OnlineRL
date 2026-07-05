import pytest

from muzero.metrics import aggregate_game_summaries


def test_aggregate_game_summaries():
    summaries = [
        {
            "result": "red_win",
            "ally_side": "w",
            "ally_won": True,
            "draw": False,
            "plies": 40,
            "truncated": False,
            "promoted": False,
            "era": 0,
        },
        {
            "result": "draw_repetition",
            "ally_side": "b",
            "ally_won": False,
            "draw": True,
            "plies": 60,
            "truncated": False,
            "promoted": False,
            "era": 0,
        },
        {
            "result": "black_win",
            "ally_side": "w",
            "ally_won": False,
            "draw": False,
            "plies": 30,
            "truncated": True,
            "promoted": False,
            "era": 0,
        },
    ]
    m = aggregate_game_summaries(summaries)
    assert m["selfplay/win_rate"] == 1 / 3
    assert m["selfplay/draw_rate"] == 1 / 3
    assert m["selfplay/loss_rate"] == 1 / 3
    assert m["selfplay/repetition_draw_rate"] == 1 / 3
    assert m["selfplay/truncation_rate"] == 1 / 3
    assert m["selfplay/mean_plies"] == 130 / 3
    # Old-style summaries lack the new §10 diagnostic keys entirely; the
    # aggregator must fall back gracefully instead of raising.
    assert m["selfplay/mean_root_entropy"] == 0.0
    assert m["selfplay/mean_ally_cp_auc"] == 0.0
    assert m["selfplay/value_cp_correlation"] == 0.0


def _base_summary(**overrides):
    base = {
        "result": "red_win",
        "ally_side": "w",
        "ally_won": True,
        "draw": False,
        "plies": 40,
        "truncated": False,
        "promoted": False,
        "era": 0,
        "mean_root_entropy": 0.5,
        "value_cp_pairs": [],
        "mean_ally_cp": 100.0,
        "games_this_era": 1,
    }
    base.update(overrides)
    return base


def test_aggregate_game_summaries_new_diagnostics():
    summaries = [
        _base_summary(
            mean_root_entropy=0.4,
            mean_ally_cp=100.0,
            value_cp_pairs=[(1.0, 100.0), (2.0, 200.0)],
        ),
        _base_summary(
            mean_root_entropy=0.6,
            mean_ally_cp=None,  # game had no ally moves with an engine cp
            value_cp_pairs=[(3.0, 300.0)],
            promoted=True,
        ),
    ]
    m = aggregate_game_summaries(summaries)
    assert m["selfplay/mean_root_entropy"] == 0.5  # mean(0.4, 0.6)
    assert m["selfplay/mean_ally_cp_auc"] == 100.0  # None skipped, only 100.0 kept
    assert m["selfplay/value_cp_correlation"] == pytest.approx(1.0)  # perfectly linear
    assert m["selfplay/promotions"] == 1


def test_red_black_win_rates():
    summaries = [
        {
            "result": "red_win",
            "ally_side": "w",
            "ally_won": True,
            "draw": False,
            "plies": 40,
            "truncated": False,
            "promoted": False,
            "era": 0,
        },
        {
            "result": "black_win",
            "ally_side": "w",
            "ally_won": False,
            "draw": False,
            "plies": 40,
            "truncated": False,
            "promoted": False,
            "era": 0,
        },
        {
            "result": "draw_repetition",
            "ally_side": "b",
            "ally_won": False,
            "draw": True,
            "plies": 40,
            "truncated": False,
            "promoted": False,
            "era": 0,
        },
    ]
    m = aggregate_game_summaries(summaries)
    assert m["selfplay/red_win_rate"] == 1 / 3
    assert m["selfplay/black_win_rate"] == 1 / 3


def test_blunder_mate_and_search_kl_metrics():
    base = {
        "ally_side": "w",
        "ally_won": False,
        "draw": False,
        "plies": 40,
        "promoted": False,
        "era": 0,
    }
    summaries = [
        {
            **base,
            "result": "red_win",
            "truncated": False,  # genuine mate win
            "blunders": 2,
            "cp_moves": 10,
            "mean_search_kl": 0.4,
        },
        {
            **base,
            "result": "black_win",
            "truncated": True,  # adjudicated win
            "blunders": 1,
            "cp_moves": 10,
            "mean_search_kl": 0.2,
        },
        {
            **base,
            "result": "draw_repetition",
            "draw": True,
            "truncated": False,
            "blunders": 0,
            "cp_moves": 20,
            "mean_search_kl": None,
        },
    ]
    m = aggregate_game_summaries(summaries)
    assert m["selfplay/blunder_rate"] == 3 / 40
    assert m["selfplay/mate_win_rate"] == 1 / 3  # truncation win excluded
    assert m["selfplay/mean_search_kl"] == (0.4 + 0.2) / 2  # None skipped
