import numpy as np
from ATNN import Model

def test_fifo_deletes_oldest_eligible_active_branch():
    print("==================================================")
    print("TEST 1: Standard FIFO Deletion of Active Branches")
    print("==================================================")
    model = Model(featureNum=10, hNeuronNum=16, cNeuronNum=3)
    model.maxBranchNum = 2

    # Add Branch 1 and Branch 2
    model.add_branch()
    assert model.branchList == [1]
    assert model.activeBranch == 1

    model.add_branch()
    assert model.branchList == [1, 2]
    assert model.activeBranch == 2

    # Both are active
    assert model.branchStatus[1] == "active"
    assert model.branchStatus[2] == "active"

    # Add Branch 3 (exceeding maxBranchNum=2)
    # Branch 1 is the oldest active branch, Branch 3 is activeBranch
    model.add_branch()

    # Branch 1 should have been deleted by FIFO
    assert 1 not in model.branchList, "Oldest active branch 1 should be removed by FIFO"
    assert 1 not in model.nodeList, "Branch 1 nodes should be deleted"
    assert 1 not in model.branchStatus, "Branch 1 status should be removed"
    assert model.branchList == [2, 3], f"Expected branchList [2, 3], got {model.branchList}"
    assert len(model.branchList) == model.maxBranchNum, "Branch limit should be respected"
    assert model.activeBranch == 3, "Newest branch 3 must be the active branch"

    print("TEST 1 PASSED: FIFO correctly deleted oldest eligible active branch.\n")


def test_dormant_branch_never_deleted_by_fifo():
    print("==================================================")
    print("TEST 2: Dormant Branch Preservation Under FIFO")
    print("==================================================")
    model = Model(featureNum=10, hNeuronNum=16, cNeuronNum=3)
    model.maxBranchNum = 2

    # Add Branch 1 and Branch 2
    model.add_branch()
    model.add_branch()
    assert model.branchList == [1, 2]

    # Deactivate / prune Branch 1 to make it dormant
    model.prune_branch(1)
    assert model.branchStatus[1] == "dormant"

    # Attach weights, Fisher matrices, and loss history to dormant Branch 1
    node_b1 = model.nodeList[1][0]
    initial_hW = node_b1.hW.copy()
    initial_cW = node_b1.cW.copy()
    initial_hb = node_b1.hb.copy()
    initial_cb = node_b1.cb.copy()
    node_b1.Fisher_hW = np.ones_like(node_b1.hW) * 0.77
    node_b1.Fisher_hb = np.ones_like(node_b1.hb) * 0.77
    initial_Fisher_hW = node_b1.Fisher_hW.copy()
    initial_Fisher_hb = node_b1.Fisher_hb.copy()
    model.lossList[1] = [0.15] * 1000
    model.lossStatisticsList[1] = {"mean": 0.15, "var": 0.02, "prev_mean": 0.16, "prev_var": 0.01}

    # Now add Branch 3 (exceeding maxBranchNum=2)
    # Branch 1 is dormant (MUST NOT BE DELETED)
    # Branch 2 is active (ELIGIBLE FOR FIFO DELETION)
    # Branch 3 is activeBranch (PROTECTED)
    model.add_branch()

    print(f"Branch list after adding Branch 3: {model.branchList}")
    print(f"Branch statuses: {model.branchStatus}")

    # Dormant Branch 1 must be preserved
    assert 1 in model.branchList, "Dormant branch 1 must NEVER be deleted by FIFO!"
    assert 1 in model.nodeList, "Dormant branch 1 nodes must be preserved in nodeList!"
    assert model.branchStatus[1] == "dormant", "Branch 1 status must remain dormant"

    # Active Branch 2 should have been deleted by FIFO
    assert 2 not in model.branchList, "Active branch 2 should be deleted by FIFO"
    assert 2 not in model.nodeList, "Active branch 2 should be deleted from nodeList"

    # New branch 3 should be active and present
    assert model.branchList == [1, 3], f"Expected branchList [1, 3], got {model.branchList}"
    assert model.activeBranch == 3, "Branch 3 must be the active branch"
    assert len(model.branchList) == model.maxBranchNum, "maxBranchNum should be respected"

    # Verify weights, Fisher matrices, and loss history are fully preserved
    assert np.array_equal(node_b1.hW, initial_hW), "Dormant branch hW must be preserved"
    assert np.array_equal(node_b1.cW, initial_cW), "Dormant branch cW must be preserved"
    assert np.array_equal(node_b1.hb, initial_hb), "Dormant branch hb must be preserved"
    assert np.array_equal(node_b1.cb, initial_cb), "Dormant branch cb must be preserved"
    assert np.array_equal(node_b1.Fisher_hW, initial_Fisher_hW), "Dormant branch Fisher_hW must be preserved"
    assert np.array_equal(node_b1.Fisher_hb, initial_Fisher_hb), "Dormant branch Fisher_hb must be preserved"
    assert len(model.lossList[1]) == 1000, "Dormant branch lossList must be preserved"
    assert model.lossStatisticsList[1]["mean"] == 0.15, "Dormant branch lossStatisticsList must be preserved"

    print("TEST 2 PASSED: Dormant branch was preserved with all weights and states intact.\n")


def test_active_branch_and_base_branch_protected():
    print("==================================================")
    print("TEST 3: Active and Base Branch Protection")
    print("==================================================")
    model = Model(featureNum=10, hNeuronNum=16, cNeuronNum=3)
    model.maxBranchNum = 1

    # Base branch (0) is in nodeList
    assert 0 in model.nodeList
    assert model.model.branchType == 0

    # Add Branch 1
    model.add_branch()
    assert model.branchList == [1]
    assert model.activeBranch == 1

    # Base branch 0 must never be removed
    assert 0 in model.nodeList, "Base branch 0 must never be deleted"
    assert 0 not in model.branchList, "Base branch 0 should not be in branchList"

    # Add Branch 2 when maxBranchNum=1
    # Branch 1 is active, Branch 2 becomes activeBranch
    model.add_branch()
    assert model.activeBranch == 2
    assert 2 in model.branchList
    assert 2 in model.nodeList
    assert 0 in model.nodeList, "Base branch 0 must remain protected"

    print("TEST 3 PASSED: Active and base branches are strictly protected.\n")


def test_all_candidates_dormant_preserves_all_reusable_branches():
    print("==================================================")
    print("TEST 4: All Candidates Dormant (Knowledge Preservation)")
    print("==================================================")
    model = Model(featureNum=10, hNeuronNum=16, cNeuronNum=3)
    model.maxBranchNum = 2

    # Add Branch 1 and Branch 2
    model.add_branch()
    model.add_branch()
    assert model.branchList == [1, 2]

    # Both Branch 1 and Branch 2 are pruned/dormant
    model.prune_branch(1)
    model.branchStatus[2] = "dormant"
    assert model.branchStatus[1] == "dormant"
    assert model.branchStatus[2] == "dormant"

    # Now add Branch 3 (exceeding maxBranchNum=2)
    # Both older branches are dormant reusable branches
    # Branch 3 is activeBranch
    model.add_branch()

    print(f"Branch list when all candidates are dormant: {model.branchList}")
    print(f"Branch statuses: {model.branchStatus}")

    # Neither dormant branch should be deleted
    assert 1 in model.branchList, "Dormant branch 1 must not be destroyed"
    assert 2 in model.branchList, "Dormant branch 2 must not be destroyed"
    assert 3 in model.branchList, "Active branch 3 must not be destroyed"
    assert model.branchList == [1, 2, 3], "All branches retained to prevent silent destruction of reusable knowledge"
    assert model.activeBranch == 3

    print("TEST 4 PASSED: Reusable dormant branches are not destroyed even when exceeding limit.\n")


def test_reactivation_of_fifo_preserved_dormant_branch():
    print("==================================================")
    print("TEST 5: Reactivation of FIFO-Preserved Dormant Branch")
    print("==================================================")
    model = Model(featureNum=10, hNeuronNum=16, cNeuronNum=3)
    model.maxBranchNum = 2

    # Add Branch 1 and Branch 2
    model.add_branch()
    model.add_branch()

    # Prune Branch 1
    model.prune_branch(1)
    assert model.branchStatus[1] == "dormant"

    # Add Branch 3 (Branch 2 deleted by FIFO, Branch 1 preserved)
    model.add_branch()
    assert model.branchList == [1, 3]
    assert model.branchStatus[1] == "dormant"
    assert model.activeBranch == 3

    # Now concept drift recurs matching Branch 1
    model.lossStatisticsList = {
        0: {"mean": 0.80, "var": 0.10, "prev_mean": 1.20, "prev_var": 0.20},
        1: {"mean": 0.20, "var": 0.05, "prev_mean": 0.22, "prev_var": 0.04},
        3: {"mean": 0.50, "var": 0.10, "prev_mean": 2.50, "prev_var": 0.50}
    }
    model.driftAlert = 1
    model.alertNum = model.splitLen

    # Concept detection should reactivate preserved dormant Branch 1
    model.concept_detection()

    assert model.activeBranch == 1, "Preserved dormant branch 1 should be reactivated!"
    assert model.branchStatus[1] == "active", "Branch 1 status must be active!"
    assert model.branchList == [1, 3], "No new branch created on reactivation"

    print("TEST 5 PASSED: Preserved dormant branch successfully reactivated upon concept recurrence.\n")


if __name__ == "__main__":
    test_fifo_deletes_oldest_eligible_active_branch()
    test_dormant_branch_never_deleted_by_fifo()
    test_active_branch_and_base_branch_protected()
    test_all_candidates_dormant_preserves_all_reusable_branches()
    test_reactivation_of_fifo_preserved_dormant_branch()
    print("==================================================")
    print("ALL FIFO DELETION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

