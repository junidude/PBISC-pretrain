"""Tiny CPU regression checks for all-cell membership and top-up block semantics."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from pbisc.training.allcell import batch_count, combine_recipe_frames, load_topup_block, pool_digest


def module_fixture():
    arrays = [np.array([[1, 7, 0], [2, 8, 0]], np.float32), np.array([[3, 9, 4]], np.float32)]
    pools = [{"shard_id": "a", "n_cells": 2}, {"shard_id": "b", "n_cells": 1}]
    base = SimpleNamespace(_load_sparse_refs=lambda refs, _: sparse.csr_matrix(np.stack([arrays[s][r] for s,r in refs])))
    module = SimpleNamespace(base=base, source_gene_indices=np.array([0, 2]),
                             _measurement_mask=lambda pool: np.array([1, 0 if pool["shard_id"] == "a" else 1], np.float32))
    return module, pools, arrays


def test_full_targets_and_masks_keep_explicit_identity():
    module,pools,arrays=module_fixture()
    ids=np.array([0,1,2]);recipe={"topup_start":0,"topup_stop":3,"k":3,"block_id":"topup"}
    block, emitted=load_topup_block(module,recipe,np.random.default_rng(9),ids,np.array([0,2]),pools)
    assert set(emitted)=={0,1,2}
    full=np.vstack(arrays)
    np.testing.assert_array_equal(block.counts,full[emitted][:,[0,2]])
    np.testing.assert_allclose(block.bulk_log1p,np.log1p(full.sum(axis=0)))
    np.testing.assert_allclose(block.bulk_measurement_fraction,[1,1/3])
    assert np.all(block.counts[block.availability_mask==0]==0)


def test_tail_recipe_uses_actual_k_without_padding():
    module,pools,_=module_fixture();recipe={"topup_start":0,"topup_stop":1,"k":1,"block_id":"tail"}
    block,emitted=load_topup_block(module,recipe,np.random.default_rng(9),np.array([2]),np.array([0,2]),pools)
    assert block.n_cells==1 and block.counts.shape==(1,2)
    assert emitted.tolist()==[2]


@pytest.mark.parametrize("ids",[[-1],[3],[0,0]])
def test_bad_explicit_coordinates_fail(ids):
    module,pools,_=module_fixture();recipe={"topup_start":0,"topup_stop":len(ids),"k":len(ids),"block_id":"bad"}
    with pytest.raises(ValueError):load_topup_block(module,recipe,np.random.default_rng(1),np.array(ids),np.array([0,2]),pools)


def test_worker_tail_batches_are_counted_separately():
    assert batch_count([257,257,257,257],256)==8
    assert batch_count([256,256,256,256],256)==4


def test_pool_digest_ignores_mapping_order_but_preserves_segment_order():
    a={"shard_id":"a","row_indices":np.array([0,2])};b={"shard_id":"b","row_indices":np.array([1])}
    assert pool_digest({"s":[a,b],"t":[b]},["t","s"])==pool_digest({"t":[b],"s":[a,b]},["s","t"])
    assert pool_digest({"s":[a,b]},["s"])!=pool_digest({"s":[b,a]},["s"])


def test_uint64_rng_seeds_remain_exact_when_topups_have_no_seed():
    import pandas as pd
    seeds = [2**53+1, 2**63+7, 2**64-1]
    main = pd.DataFrame({"recipe_id":["a","b","c"], "rng_seed":np.asarray(seeds,dtype=np.uint64), "k":[250,500,4000]})
    topup = pd.DataFrame({"recipe_id":["extra"],"schedule_slot":[0],"k":[1]})
    combined = combine_recipe_frames(main,topup,0)
    assert combined.rng_seed.tolist()==seeds+[0]
    assert [int(r.rng_seed) for _,r in combined.iloc[:3].iterrows()]==seeds
    for old,new in zip(seeds,combined.rng_seed):
        np.testing.assert_array_equal(np.random.default_rng(old).choice(10000,250,replace=False),
                                      np.random.default_rng(int(new)).choice(10000,250,replace=False))


def test_empty_topup_slot_also_preserves_seed_and_all_rows():
    import pandas as pd
    main=pd.DataFrame({"recipe_id":["a"],"rng_seed":np.asarray([2**64-3],dtype=np.uint64),"k":[250]})
    extras=pd.DataFrame({"recipe_id":["extra"],"schedule_slot":[1],"k":[1]})
    merged=combine_recipe_frames(main,extras,0)
    assert len(merged)==1 and int(merged.rng_seed.iloc[0])==2**64-3


def _contract_training(**overrides):
    training = {"seed": 20260627, "gradient_accumulation_steps": 1, "batch_size": 256,
                "num_workers": 4, "prefetch_factor": 1}
    training.update(overrides)
    return training


def test_stream_contract_allows_any_prefetch_depth_but_pins_the_stream_shape():
    from pbisc.training.allcell import validate_stream_contract

    data = {"train_cells_per_block": 4000}
    assert validate_stream_contract(_contract_training(), data) == 1
    assert validate_stream_contract(_contract_training(prefetch_factor=32), data) == 32
    for bad in (0, 65):
        with pytest.raises(ValueError, match="prefetch_factor must be in"):
            validate_stream_contract(_contract_training(prefetch_factor=bad), data)
    for overrides in ({"num_workers": 8}, {"batch_size": 512}, {"seed": 1},
                      {"gradient_accumulation_steps": 2}):
        with pytest.raises(ValueError, match="stream contract differs"):
            validate_stream_contract(_contract_training(**overrides), data)
    with pytest.raises(ValueError, match="stream contract differs"):
        validate_stream_contract(_contract_training(), {"train_cells_per_block": 2000})
