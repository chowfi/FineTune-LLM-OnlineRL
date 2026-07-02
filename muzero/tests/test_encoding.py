from muzero.encoding import index_to_move, move_to_index


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
