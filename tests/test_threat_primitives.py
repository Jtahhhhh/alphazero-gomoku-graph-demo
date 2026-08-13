import numpy as np

from azgomoku.game import GomokuState
from azgomoku.tactics import (
    classify_threat_move,
    creates_five,
    mandatory_defenses,
    threat_moves,
    winning_completions,
)


def state(rows, to_play=1, k=5):
    return GomokuState(np.asarray(rows, dtype=np.int8), to_play=to_play, win_length=k)


def empty(size=7, k=5):
    return state(np.zeros((size, size), dtype=np.int8), k=k)


def with_stones(size, stones, k=5):
    board = np.zeros((size, size), dtype=np.int8)
    for row, col, player in stones:
        board[row, col] = player
    return state(board, k=k)


def test_five_and_winning_completions_are_replay_verified():
    s = with_stones(7, [(3, col, 1) for col in range(1, 5)])
    assert creates_five(s, 3 * 7, 1)
    assert creates_five(s, 3 * 7 + 5, 1)
    assert not creates_five(s, 0, 1)
    threats = winning_completions(s, 1)
    assert {item.completion for item in threats} == {21, 26}
    assert {item.relation for item in threats} == {"horizontal"}


def test_single_four_has_one_mandatory_block():
    s = with_stones(7, [(2, col, 1) for col in range(4)])
    defense = mandatory_defenses(s, 1)
    assert defense.completions == (18,)
    assert defense.blocking_moves == (18,)
    assert not defense.unstoppable


def test_open_four_and_cross_double_four_are_unstoppable_by_one_block():
    open_four = with_stones(7, [(3, col, 1) for col in range(1, 5)])
    assert mandatory_defenses(open_four, 1).unstoppable

    cross = with_stones(
        7,
        [(3, col, 1) for col in (1, 2, 4)]
        + [(row, 3, 1) for row in (1, 2, 4)],
    )
    classified = classify_threat_move(cross, 3 * 7 + 3, 1)
    assert classified.creates_double_four
    assert {threat.completion for threat in classified.fours} == {21, 26, 3, 38}


def test_three_and_open_three_are_move_candidates_not_certificates():
    s = with_stones(7, [(3, 2, 1), (3, 4, 1)])
    classified = classify_threat_move(s, 3 * 7 + 3, 1)
    assert classified.creates_three
    assert classified.creates_open_three
    assert {3 * 7 + col for col in (1, 5)} <= set(classified.three_extensions)
    assert not classified.creates_five


def test_four_three_and_method_candidate_sets():
    s = with_stones(
        7,
        [(3, col, 1) for col in (0, 1, 2)]
        + [(row, 3, 1) for row in (1, 5)],
    )
    move = 3 * 7 + 3
    classified = classify_threat_move(s, move, 1)
    assert classified.creates_four
    assert classified.creates_three
    assert classified.creates_four_three
    assert move in {item.move for item in threat_moves(s, 1, "vcf")}

    three_only = with_stones(7, [(3, 2, 1), (3, 4, 1)])
    target = 3 * 7 + 3
    assert target not in {item.move for item in threat_moves(three_only, 1, "vcf")}
    assert target in {item.move for item in threat_moves(three_only, 1, "vct")}


def test_primitives_cover_all_relations_and_reject_illegal_moves():
    fixtures = [
        ([(2, col, 1) for col in range(4)], 2 * 7 + 4, "horizontal"),
        ([(row, 2, 1) for row in range(4)], 4 * 7 + 2, "vertical"),
        ([(i, i, 1) for i in range(4)], 4 * 7 + 4, "diagonal_down"),
        ([(i, 4 - i, 1) for i in range(4)], 4 * 7, "diagonal_up"),
    ]
    for stones, move, relation in fixtures:
        s = with_stones(7, stones)
        assert creates_five(s, move, 1)
        assert relation in {item.relation for item in winning_completions(s, 1)}

    occupied = with_stones(7, [(0, 0, 1)])
    assert not creates_five(occupied, 0, 1)
    assert not classify_threat_move(occupied, 0, 1).creates_four
