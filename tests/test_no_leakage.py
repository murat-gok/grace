"""Anti-leakage guard: train and test graph splits must never overlap.

This test exists because an earlier version generated train and test graphs with
the same seeds, leaking test graphs into GNN pretraining and inflating results.
Keep this test green.
"""
from grace_qaoa.quantum.graphs import make_dataset, SPLIT_OFFSET


def _edge_set(g):
    return frozenset(frozenset((u, v)) for u, v in g.edges())


def test_train_test_no_overlap():
    for fam in ["regular", "erdos_renyi", "watts_strogatz"]:
        train = make_dataset(fam, n_graphs=30, n_nodes=12, weighted=True,
                             split="train")
        test = make_dataset(fam, n_graphs=12, n_nodes=12, weighted=True,
                            split="test")
        train_sets = {_edge_set(g) for g in train}
        for tg in test:
            assert _edge_set(tg) not in train_sets, \
                f"LEAKAGE: a test graph appears in the {fam} training pool"


def test_splits_have_distinct_offsets():
    assert len(set(SPLIT_OFFSET.values())) == len(SPLIT_OFFSET)
    assert SPLIT_OFFSET["test"] != SPLIT_OFFSET["train"]


def test_test_split_is_reproducible():
    # Same split + seed must reproduce the same graphs (determinism).
    a = make_dataset("regular", n_graphs=5, n_nodes=10, weighted=True, split="test")
    b = make_dataset("regular", n_graphs=5, n_nodes=10, weighted=True, split="test")
    for ga, gb in zip(a, b):
        assert set(ga.edges()) == set(gb.edges())
