from investigation.e3a2_expand_gold import MID_RANGE,_seed,_state_at


def test_mid_range_and_seed_derivation_are_stable():
    assert list(MID_RANGE)==[5,6,7,8,9]
    assert _seed(1000,3,2)==_seed(1000,3,2)
    assert _seed(1000,3,2)!=_seed(1000,3,3)


def test_state_at_replays_requested_prefix():
    state=_state_at([0,1,6,2,12],3)
    assert state.last_move==6 and state.to_play==-1
    assert int((state.board!=0).sum())==3
