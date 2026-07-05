import numpy as np
import pytest

from muzero.encoding import (
    absolute_visits,
    board_planes,
    encode_observation,
    flip_action,
    flip_board,
    index_to_move,
    material_balance,
    mirror_action,
    mirror_board,
    move_to_index,
)


def test_move_index_round_trip():
    for move in ["a0a1", "h9g7", "e6e5", "i9i0"]:
        idx = move_to_index(move)
        assert 0 <= idx < 8100
        assert index_to_move(idx) == move


def test_all_indices_decode_and_reencode():
    for idx in range(0, 8100, 173):
        assert move_to_index(index_to_move(idx)) == idx


def test_index_formula():
    # a0 -> square 0, a1 -> square 9: index = 0 * 90 + 9
    assert move_to_index("a0a1") == 9


def test_edge_indices_round_trip():
    for idx in (0, 8099):
        assert move_to_index(index_to_move(idx)) == idx


def test_bad_inputs_raise():
    for bad_move in ["", "z0a1", "a10a1", "a0a1x"]:
        with pytest.raises(ValueError):
            move_to_index(bad_move)
    for bad_idx in (-1, 8100, 10000):
        with pytest.raises(ValueError):
            index_to_move(bad_idx)


def _start_board():
    board = np.zeros((10, 9), dtype=np.int8)
    back = [8, 6, 4, 2, 1, 3, 5, 7, 9]  # r n b a k a b n r piece ids
    for c, pid in enumerate(back):
        board[0, c] = -pid  # black top
        board[9, c] = pid  # red bottom
    board[2, 1], board[2, 7] = -10, -11  # black cannons
    board[7, 1], board[7, 7] = 10, 11  # red cannons
    for c in range(0, 9, 2):
        board[3, c] = -(12 + c // 2)  # black pawns
        board[6, c] = 12 + c // 2  # red pawns
    return board


def test_board_planes_start_position():
    planes = board_planes(_start_board())
    assert planes.shape == (14, 10, 9)
    assert planes.sum() == 32  # 16 pieces per side
    assert planes[0, 9, 4] == 1.0  # red king plane
    assert planes[7, 0, 4] == 1.0  # black king plane


def test_encode_observation_shape_and_padding():
    board = _start_board()
    obs = encode_observation([board], "w", 1, 0, history_length=8)
    assert obs.shape == (114, 10, 9)
    assert obs[:14].sum() == 0  # padded oldest history slot is empty
    assert obs[98:112].sum() == 32  # newest slot holds the board


def test_material_balance():
    board = _start_board()
    assert material_balance(board) == 0.0
    board[0, 0] = 0  # remove a black rook
    assert material_balance(board) == 9.0


def test_encode_observation_broadcast_planes():
    board = _start_board()
    obs = encode_observation([board], "w", 1, 50, history_length=8)
    assert np.allclose(obs[112], 1.0 / 3.0)  # repetition_count=1
    assert np.allclose(obs[113], 0.5)  # no_progress=50
    obs_b = encode_observation([board], "b", 3, 200, history_length=8)
    assert np.allclose(obs_b[112], 1.0)  # clamped at 3
    assert np.allclose(obs_b[113], 1.0)  # clamped at 100


def test_encode_observation_canonicalizes_black_to_move():
    board = _start_board()
    board[6, 0] = 0  # remove a red pawn so the position is asymmetric
    obs_w = encode_observation([board], "w", 1, 0, history_length=8)
    obs_b = encode_observation([board], "b", 1, 0, history_length=8)
    assert obs_w.shape == obs_b.shape == (114, 10, 9)
    # red to move: newest slot is the absolute board
    np.testing.assert_array_equal(obs_w[98:112], board_planes(board))
    # black to move: newest slot is the flipped board (mover's canonical frame)
    np.testing.assert_array_equal(obs_b[98:112], board_planes(flip_board(board)))
    # rep / no-progress broadcast planes moved down to 112 / 113
    assert np.allclose(obs_b[112], 1.0 / 3.0)
    assert np.allclose(obs_b[113], 0.0)


def test_flip_action_involutions_exhaustive():
    idx = np.arange(8100, dtype=np.int64)
    assert np.array_equal(flip_action(flip_action(idx)), idx)
    assert np.array_equal(mirror_action(mirror_action(idx)), idx)
    # the two mirrors act on different axes, so they commute
    assert np.array_equal(
        flip_action(mirror_action(idx)), mirror_action(flip_action(idx))
    )


def test_flip_action_scalar_semantics():
    # black pawn push (3,0)->(4,0) flips to red pawn push (6,0)->(5,0)
    assert index_to_move(flip_action(move_to_index("a3a4"))) == "a6a5"
    # left-right mirror: file a -> file i
    assert index_to_move(mirror_action(move_to_index("a3a4"))) == "i3i4"
    assert isinstance(flip_action(move_to_index("a3a4")), int)


def test_flip_action_rejects_bad_indices():
    for bad in (-1, 8100):
        with pytest.raises(ValueError):
            flip_action(bad)
        with pytest.raises(ValueError):
            mirror_action(bad)
    # an out-of-range element anywhere in an array is rejected too
    with pytest.raises(ValueError):
        flip_action(np.array([0, -1, 50]))
    with pytest.raises(ValueError):
        mirror_action(np.array([0, -1, 50]))
    # non-integer input is a TypeError, not silent truncation
    with pytest.raises(TypeError):
        flip_action(np.array([1.5]))
    with pytest.raises(TypeError):
        mirror_action(np.array([1.5]))


def test_encode_observation_offsets_generalize_history_length():
    board = _start_board()
    obs = encode_observation([board], "w", 1, 50, history_length=4)
    assert obs.shape == (58, 10, 9)
    assert np.allclose(obs[56], 1.0 / 3.0)  # rep plane at 14*H
    assert np.allclose(obs[57], 0.5)  # no-progress plane at 14*H + 1


def test_flip_board_involution_and_color_swap():
    board = _start_board()
    # the start position is vertically AND horizontally symmetric at the
    # piece-type level (raw ids differ left-right, e.g. rook id 8 vs 9,
    # so we compare via board_planes which is what the network sees)
    np.testing.assert_array_equal(flip_board(board), board)
    np.testing.assert_array_equal(
        board_planes(mirror_board(board)), board_planes(board)
    )
    # asymmetric position: remove a red pawn at (6,0)
    board[6, 0] = 0
    fb = flip_board(board)
    np.testing.assert_array_equal(flip_board(fb), board)  # involution
    assert fb[3, 0] == 0  # the gap lands at the flipped square...
    assert fb[6, 0] > 0  # ...and black's pawn (color-swapped positive) sits at (6,0)
    mb = mirror_board(board)
    np.testing.assert_array_equal(mirror_board(mb), board)
    assert mb[6, 8] == 0 and mb[6, 0] > 0


def test_absolute_visits():
    visits = {move_to_index("a6a5"): 7, move_to_index("e6e5"): 3}
    assert absolute_visits(visits, "w") is visits  # red: identity, no copy
    unflipped = absolute_visits(visits, "b")
    assert unflipped == {
        flip_action(move_to_index("a6a5")): 7,
        flip_action(move_to_index("e6e5")): 3,
    }
