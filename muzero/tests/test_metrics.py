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
            "final_red_cp": 250.0,
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
            "final_red_cp": 0.0,
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
            "final_red_cp": None,
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
    assert m["selfplay/mean_final_ally_cp"] == 125.0  # (250 + 0)/2, None skipped
