import copy,json
from investigation.h3_evaluate import fixed_targets


def test_h1_solver_targets_do_not_depend_on_model_action():
    record={"state_id":"fixed","solver":{"value":1,"optimal_actions":[3,7]},"valid_proofs":[{"action":3,"critical_cells":[1,2],"critical_relations":["horizontal"],"windows":[[1,2]]},{"action":7,"critical_cells":[7],"critical_relations":["vertical"],"windows":[[7]]}]}
    before=copy.deepcopy(record); choosing_3=fixed_targets(record); choosing_7=fixed_targets(record)
    assert choosing_3==choosing_7 and record==before
    assert choosing_3["optimal_actions"]==(3,7) and json.loads(choosing_3["valid_proofs"])==record["valid_proofs"]
