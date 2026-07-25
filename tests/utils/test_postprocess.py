import numpy as np
import pandas as pd
import pytest

from utils.postprocess import combine_result_dataframes, last_epoch_only, last_step_only
from utils.training import metrics_to_dataframe, per_epoch_metrics


def epoch_result(accuracies):
    """ A result frame shaped like the one a supervised run produces: one row per epoch. """
    return pd.DataFrame({"Step": range(len(accuracies)),
                         "Epoch": range(len(accuracies)),
                         "Test Accuracy": accuracies})


def step_result(steps, rewards):
    """ A result frame shaped like the one an RL run produces: step-indexed, with no Epoch column. """
    return pd.DataFrame({"Step": steps, "Eval/Reward": rewards})


def test_metrics_to_dataframe_keeps_every_step():
    metrics = {0: {"Loss": 1.0}, 16: {"Loss": 0.5}, 32: {"Loss": 0.25}}
    df = metrics_to_dataframe(metrics, seed=1)
    assert list(df["Step"]) == [0, 16, 32]
    assert list(df["Loss"]) == [1.0, 0.5, 0.25]
    assert list(df["seed"]) == [1, 1, 1], "Metadata should be prepended to every row."


def test_metrics_to_dataframe_leaves_gaps_as_nan():
    # Metrics recorded only periodically (evaluation, for instance) are absent from the rows in between.
    metrics = {0: {"Loss": 1.0, "Eval/Reward": 0.1}, 16: {"Loss": 0.5}, 32: {"Loss": 0.25, "Eval/Reward": 0.7}}
    df = metrics_to_dataframe(metrics)
    assert list(df["Eval/Reward"].notna()) == [True, False, True]


def test_per_epoch_metrics_still_groups_by_epoch():
    # per_epoch_metrics now delegates its first half to metrics_to_dataframe; it must behave exactly as before.
    metrics = {1: {"Epoch": 1, "Loss": 1.0}, 2: {"Epoch": 1, "Loss": 0.9},
               3: {"Epoch": 2, "Loss": 0.5}, 4: {"Epoch": 2, "Loss": 0.4}}
    df = per_epoch_metrics(metrics)
    assert list(df["Epoch"]) == [0, 1, 2], "Should keep one row per epoch, prepending a blank epoch 0."
    assert list(df["Step"]) == [0, 2, 4], "Should keep the last step of each epoch."


def test_combine_keeps_epoch_index_when_present():
    full = combine_result_dataframes([epoch_result([0.1, 0.2])], [{"model": "conv4"}])
    assert list(full.index.names) == ["model", "Step", "Epoch"]


def test_combine_omits_epoch_index_when_absent():
    # RL results have no Epoch column; indexing must not fail on them.
    full = combine_result_dataframes([step_result([0, 16], [0.1, 0.5])], [{"model": "conv4"}])
    assert list(full.index.names) == ["model", "Step"]
    assert "Eval/Reward" in full.columns


def test_combine_indexes_step_based_results_by_real_step():
    # For epoch-based results the "Step" level is historically the row number. Step-based results have no Epoch to
    # key off, so their "Step" level must carry the actual env step, not the row number.
    full = combine_result_dataframes([step_result([0, 16, 32], [0.1, 0.2, 0.3])], [{"model": "a"}])
    assert full.index.get_level_values("Step").tolist() == [0, 16, 32]


def test_combine_rejects_mixing_epoch_and_step_results():
    with pytest.raises(RuntimeError, match="epoch-based and step-based"):
        combine_result_dataframes([epoch_result([0.1, 0.2]), step_result([0, 16], [0.3, 0.4])],
                                  [{"model": "a"}, {"model": "b"}])


def test_last_epoch_only_picks_final_epoch_per_run():
    full = combine_result_dataframes([epoch_result([0.1, 0.2, 0.3]), epoch_result([0.4, 0.5])],
                                     [{"model": "a"}, {"model": "b"}])
    final = last_epoch_only(full)
    assert len(final) == 2
    assert sorted(final["Test Accuracy"]) == [0.3, 0.5]


def test_last_step_only_picks_final_step_per_run():
    full = combine_result_dataframes([step_result([0, 16, 32], [0.1, 0.2, 0.3]), step_result([0, 16], [0.4, 0.5])],
                                     [{"model": "a"}, {"model": "b"}])
    final = last_step_only(full)
    assert len(final) == 2
    assert sorted(final["Eval/Reward"]) == [0.3, 0.5]


def test_last_step_only_can_require_a_populated_column():
    # Evaluation runs only periodically, so the final step of a run usually has no eval result at all.
    full = combine_result_dataframes([step_result([0, 16, 32], [0.1, np.nan, np.nan])], [{"model": "a"}])

    plain = last_step_only(full)
    assert plain.index.get_level_values("Step").tolist() == [32]
    assert plain["Eval/Reward"].isna().all(), "Without `require`, we land on the last step, which has no eval result."

    final = last_step_only(full, require="Eval/Reward")
    assert final.index.get_level_values("Step").tolist() == [0], "Should fall back to the last step that has a value."
    assert final["Eval/Reward"].tolist() == [0.1]


def test_last_step_only_rejects_unknown_require_column():
    full = combine_result_dataframes([step_result([0, 16], [0.1, 0.2])], [{"model": "a"}])
    with pytest.raises(ValueError, match="Nonexistent"):
        last_step_only(full, require="Nonexistent")
