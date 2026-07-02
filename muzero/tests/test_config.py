from muzero.config import MuZeroConfig


def test_defaults_match_spec():
    cfg = MuZeroConfig()
    assert cfg.num_simulations == 800
    assert cfg.unroll_steps == 8
    assert cfg.num_workers == 3
    assert cfg.games_per_worker == 28
    assert cfg.buffer_games == 5000
    assert cfg.games_per_train_loop == 512
    assert len(cfg.opening_book) == 10
    assert cfg.input_planes == 14 * cfg.history_length + 3 == 115
    assert len(cfg.loss_weights) == 6


def test_input_planes_derived_from_history_length():
    cfg = MuZeroConfig(history_length=4)
    assert cfg.input_planes == 59
