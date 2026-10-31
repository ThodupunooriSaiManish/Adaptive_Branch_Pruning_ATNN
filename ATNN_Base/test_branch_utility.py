from ATNN import Model

model = Model(
    featureNum=10,
    hNeuronNum=16,
    cNeuronNum=3
)

# Create two test branches in the actual tree
model.add_child_node(model.model, 1)
model.add_child_node(model.model, 2)

model.branchList = [1, 2]
model.activeBranch = 2

model.branchUsage = {
    0: 500,
    1: 50,
    2: 400
}

model.branchLastUsed = {
    0: 1000,
    1: 100,
    2: 950
}

model.trainTimes = 1000

print("Branches before pruning:", model.branchList)

selected = model.select_branch_for_pruning()

print("Selected branch:", selected)

# Perform pruning
if selected is not None:
    model.prune_branch(selected)

print("Branches after pruning:", model.branchList)
print("Active branch:", model.activeBranch)
print("Remaining nodeList:", list(model.nodeList.keys()))