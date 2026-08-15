from investigation.e3a1_measure import render_budget_svg,summarize_production


def test_production_summary_splits_labels_phase_and_wall_time(tmp_path):
    rows=[
        {"board_size":6,"ply":7,"ply_bucket":"5-9","status":"exact_complete","wall_elapsed_ms":10,"partial_replay_pass":True},
        {"board_size":6,"ply":12,"ply_bucket":"10+","status":"exact_partial","wall_elapsed_ms":20,"partial_replay_pass":True},
        {"board_size":6,"ply":13,"ply_bucket":"10+","status":"unknown","wall_elapsed_ms":30,"partial_replay_pass":True},
    ]
    summary=summarize_production(rows,tmp_path,3,{"6":2000})
    assert summary["boards"]["6"]["complete"]==1
    assert summary["boards"]["6"]["partial_replay_pass"]==1
    assert summary["distribution"]["6:10+"]=={"complete":0,"partial":1,"unknown":1,"total":2}
    assert summary["complete_ply_stats"]["6"]["median"]==7
    assert summary["wall_time_total_s"]==.06


def test_budget_svg_contains_all_board_curves():
    summary={"budgets_ms":[500,2000],"boards":{str(size):{"rates":[{"rate":0.},{"rate":.5}]} for size in (6,10,15)}}
    svg=render_budget_svg(summary)
    assert "6x6" in svg and "10x10" in svg and "15x15" in svg
