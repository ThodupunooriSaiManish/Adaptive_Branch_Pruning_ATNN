import numpy as np
from ATNN import Model

def test_dormant_branch_reactivation():
    print("==================================================")
    print("TEST 1: Reactivation of Matching Dormant Branch")
    print("==================================================")
    
    model = Model(
        featureNum=10,
        hNeuronNum=16,
        cNeuronNum=3
    )

    # Create two test branches in the tree
    model.add_child_node(model.model, 1)
    model.add_child_node(model.model, 2)
    model.branchList = [1, 2]
    model.branchNum = 2
    model.activeBranch = 2
    model.branchStatus = {0: "active", 1: "active", 2: "active"}
    model.branchUsage = {0: 500, 1: 300, 2: 400}
    model.branchLastUsed = {0: 1000, 1: 600, 2: 950}
    model.trainTimes = 1000

    # Prune branch 1 to make it dormant
    model.prune_branch(1)
    assert model.branchStatus[1] == "dormant", "Branch 1 should be dormant after pruning"
    print(f"Branch statuses before drift: {model.branchStatus}")
    print(f"Active branch before drift: {model.activeBranch}")

    # Record initial weights, Fisher matrices, and loss history of Branch 1
    node_b1 = model.nodeList[1][0]
    initial_hW = node_b1.hW.copy()
    initial_cW = node_b1.cW.copy()
    initial_hb = node_b1.hb.copy()
    initial_cb = node_b1.cb.copy()
    node_b1.Fisher_hW = np.ones_like(node_b1.hW) * 0.42
    node_b1.Fisher_hb = np.ones_like(node_b1.hb) * 0.42
    initial_Fisher_hW = node_b1.Fisher_hW.copy()
    initial_Fisher_hb = node_b1.Fisher_hb.copy()

    model.lossList[1] = [0.2] * 1000

    # Simulate drift alert where Branch 1 best matches the current concept
    # Branch 2 (active) has high loss (drift)
    # Branch 1 (dormant) has low loss within its historical threshold:
    # prev_var + prev_mean (0.04 + 0.22 = 0.26) <= mean + confid * var (0.2 + 3 * 0.05 = 0.35)
    model.lossStatisticsList = {
        0: {"mean": 0.80, "var": 0.10, "prev_mean": 1.20, "prev_var": 0.20},
        1: {"mean": 0.20, "var": 0.05, "prev_mean": 0.22, "prev_var": 0.04},
        2: {"mean": 0.50, "var": 0.10, "prev_mean": 2.50, "prev_var": 0.50}
    }
    model.driftAlert = 1
    model.alertNum = model.splitLen

    branches_before = list(model.branchList)

    # Trigger concept detection
    model.concept_detection()

    print(f"Branch statuses after concept_detection: {model.branchStatus}")
    print(f"Active branch after concept_detection: {model.activeBranch}")

    # Verify dormant branch reactivation
    assert model.branchStatus[1] == "active", "Branch 1 must be reactivated to 'active' status!"
    assert model.activeBranch == 1, "Branch 1 must become the active branch!"
    assert model.branchList == branches_before, "No new branch should be created when matching a dormant branch!"

    # Verify all learned weights and Fisher matrices are preserved
    assert np.array_equal(node_b1.hW, initial_hW), "Branch 1 hW weights must be preserved!"
    assert np.array_equal(node_b1.cW, initial_cW), "Branch 1 cW weights must be preserved!"
    assert np.array_equal(node_b1.hb, initial_hb), "Branch 1 hb biases must be preserved!"
    assert np.array_equal(node_b1.cb, initial_cb), "Branch 1 cb biases must be preserved!"
    assert np.array_equal(node_b1.Fisher_hW, initial_Fisher_hW), "Branch 1 Fisher_hW must be preserved!"
    assert np.array_equal(node_b1.Fisher_hb, initial_Fisher_hb), "Branch 1 Fisher_hb must be preserved!"
    assert len(model.lossList[1]) == 1000, "Branch 1 loss history must be preserved!"
    assert model.driftAlert == 0, "driftAlert should be reset to 0"

    print("TEST 1 PASSED: Matching dormant branch reactivated with all weights and states preserved.\n")


def test_genuinely_new_concept_creates_branch():
    print("==================================================")
    print("TEST 2: Genuinely New Concept Adds New Branch")
    print("==================================================")

    model = Model(
        featureNum=10,
        hNeuronNum=16,
        cNeuronNum=3
    )

    model.add_child_node(model.model, 1)
    model.add_child_node(model.model, 2)
    model.branchList = [1, 2]
    model.branchNum = 2
    model.activeBranch = 2
    model.branchStatus = {0: "active", 1: "active", 2: "active"}
    model.trainTimes = 1000

    # Prune branch 1 so it is dormant
    model.prune_branch(1)
    assert model.branchStatus[1] == "dormant"

    # Simulate drift where NO branch matches (all exceed confidence bounds)
    # Branch 1 (lowest prev_mean): prev_var + prev_mean = 0.5 + 3.0 = 3.5 > 0.2 + 3 * 0.05 = 0.35
    model.lossStatisticsList = {
        0: {"mean": 0.80, "var": 0.10, "prev_mean": 3.50, "prev_var": 0.50},
        1: {"mean": 0.20, "var": 0.05, "prev_mean": 3.00, "prev_var": 0.50},
        2: {"mean": 0.50, "var": 0.10, "prev_mean": 3.20, "prev_var": 0.50}
    }
    model.driftAlert = 1
    model.alertNum = model.splitLen

    branches_before = list(model.branchList)

    # Trigger concept detection
    model.concept_detection()

    print(f"Branch list after new concept: {model.branchList}")
    print(f"Active branch after new concept: {model.activeBranch}")
    print(f"Branch statuses: {model.branchStatus}")

    # Verify a genuinely new branch was created
    assert len(model.branchList) == len(branches_before) + 1, "A new branch should be created for new concept!"
    assert model.activeBranch == 3, "New branch 3 must become active!"
    assert model.branchStatus[3] == "active", "New branch status must be 'active'!"
    assert model.branchStatus[1] == "dormant", "Dormant branch 1 must remain dormant when not matching!"

    print("TEST 2 PASSED: New branch created only for genuinely new concept.\n")


def test_reactivate_branch_method():
    print("==================================================")
    print("TEST 3: Direct reactivate_branch() Method")
    print("==================================================")

    model = Model(
        featureNum=10,
        hNeuronNum=16,
        cNeuronNum=3
    )
    model.add_child_node(model.model, 1)
    model.branchList = [1]
    model.branchNum = 1
    model.activeBranch = 0
    model.branchStatus = {0: "active", 1: "active"}

    # Deactivate branch 1
    model.prune_branch(1)
    assert model.branchStatus[1] == "dormant"

    # Reactivate branch 1
    res = model.reactivate_branch(1)
    assert res is True, "reactivate_branch should return True on success"
    assert model.branchStatus[1] == "active", "branchStatus must be updated to 'active'"
    assert model.activeBranch == 1, "activeBranch must be updated to 1"

    # Try to reactivate already active branch
    res_already_active = model.reactivate_branch(1)
    assert res_already_active is False, "reactivating an already active branch should return False"

    # Try to reactivate nonexistent branch
    res_nonexistent = model.reactivate_branch(999)
    assert res_nonexistent is False, "reactivating a nonexistent branch should return False"

    print("TEST 3 PASSED: reactivate_branch() API behaves as expected.\n")


def test_forward_propagation_after_reactivation():
    print("==================================================")
    print("TEST 4: Forward Propagation After Reactivation")
    print("==================================================")

    model = Model(
        featureNum=10,
        hNeuronNum=16,
        cNeuronNum=3
    )
    model.add_child_node(model.model, 1)
    model.branchList = [1]
    model.branchNum = 1
    model.activeBranch = 0
    model.branchStatus = {0: "active", 1: "active"}

    # Make branch 1 dormant
    model.prune_branch(1)
    model.driftAlert = 0

    dummy_input = np.random.randn(10, 1)

    # When dormant and no driftAlert, branch 1 output is not computed
    model.forward_propagation(model.model, dummy_input)
    output_dormant = model.get_model_output()
    assert 1 not in output_dormant, "Dormant branch should be skipped when driftAlert=0"

    # Now reactivate branch 1
    model.reactivate_branch(1)
    model.forward_propagation(model.model, dummy_input)
    output_active = model.get_model_output()

    assert 1 in output_active, "Reactivated branch must be computed in get_model_output"
    assert model.activeBranch in output_active, "Active branch output must be available"
    print(f"Reactivated branch output shape: {output_active[1].shape}")

    print("TEST 4 PASSED: Reactivated branch successfully evaluates in forward pass.\n")


if __name__ == "__main__":
    test_dormant_branch_reactivation()
    test_genuinely_new_concept_creates_branch()
    test_reactivate_branch_method()
    test_forward_propagation_after_reactivation()
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

