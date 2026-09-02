# Reproducing the results

Run commands from the repository root after completing [setup](../README.md).
The workflow uses the online experiment (`experiment=0`, labeled H2).

## Run

```sh
python analysis/reproduce_figures.py --list
mkdir -p reproduction
python -u analysis/reproduce_figures.py --output reproduction/figures > reproduction/analysis.log 2>&1
```

For a subset, add `--panels 8A 8B` (or any IDs below). Use a fresh output directory.
The runner produces manuscript-named PNGs and descriptive copies. Statistical
summaries are printed to the log; inspect it for incomplete analyses or missing
participants. The runner does not refit models or replace manuscript files.

Required inputs:

- `data/experiment_0/*.json`: raw participant records.
- `analysis/cache/all_subjects_model_parameters_online.csv`: processed trials.
- `analysis/results_online/<model>_0/<subject_id>.pkl`: full-block fits.
- `analysis/results_online/<model>_by_block_half_0/<subject_id>.pkl`: half-block fits.

Figure 8 uses IDs 0–29 for nine models: `momentum_learn_alt_goal`, `momentum`,
`momentum_with_pros_subgoal`, `momentum_with_pers_subgoal`, `prospective`,
`td_persistence`, `retrospective`, `resources`, and `progress`.

## Figure map

Numbering follows the manuscript figure ids.
Methods below are in `plot_measures.py` unless another module is named.

| Panels | Method(s) |
| --- | --- |
| 2A/B/C | `plot_goal_selection_per_block`, `plot_subgoal_selection_per_block`, `plot_viable_goal_selection_high_low` |
| 3A/B/C | `plot_goal_transition_heatmaps`, `plot_subgoal_transition_heatmaps(condition_on_goal_stay=True)`, `plot_goal_transition_heatmaps_by_block_half` |
| 4A/B, 5, 6B | `plot_switching_characteristics(goal_type='goal')` |
| 6A/C/D | `plot_rt_measures.py`: `plot_goal_rt_by_switching_pattern`, `plot_goal_rt_by_repetition`, `plot_subgoal_rt_by_repetition` |
| 7A/B | `plot_stay_proportion_with_progress_combined`, `plot_stay_by_block_type_combined` |
| 7C/D | `plot_goal_selection_related_goal_progress(simulate=False)`, `plot_subgoal_selection_related_subgoal_progress(simulate=False)` |
| 8A/B | `model_comparisons.py`: `plot_comprehensive_model_comparison`, `plot_comprehensive_model_comparison_by_block_half` |
| 9A/B | `plot_measures_mixed_effects.py`: `plot_figure9a`, `plot_forest_plot_goal_rt_switch_vs_stay` |

Figure 1 is supplied artwork. The runner renames the native 7C/D outputs
(`Figure9C.png`/`Figure9D.png`) to match the manuscript. Shared methods may generate
additional panels. Use the runner rather than the exploratory script entry points.

## Rebuild data or refit models

**First verify the original participant-ID-to-JSON mapping with the maintainers.**
The loader assigns IDs using unsorted directory listings, which can change across
machines. Numbered fits must match the correct raw participants. This also matters
for cached-data analyses that read raw participant summaries.

Start a Python session from the repository root and configure paths before
importing analysis modules:

```python
import os
import sys
from pathlib import Path

analysis = Path('analysis').resolve()
sys.path.insert(0, str(analysis))
from reproduce_figures import configure_paths
out = Path('reproduction/rebuilt').resolve()
out.mkdir(parents=True, exist_ok=True)
configure_paths(out)
```

To rebuild the cache from observed choices and saved momentum parameters:

```python
from src.create_model_parameter_df import get_all_subjects_concatenated_model_dataframe
os.chdir(out)  # The builder writes its CSV into the working directory.
get_all_subjects_concatenated_model_dataframe(
    data_type='online', model_name='momentum_learn_alt_goal',
    optimal=False, simulate=False, read_from_csv=False,
)
```

Check the rebuilt CSV before replacing
`analysis/cache/all_subjects_model_parameters_online.csv`; preserve the original.

To refit one participant/model, use a separate working copy because existing
online fits can be overwritten:

```python
from fit_behavior import BehaviorFits
os.chdir(analysis)
fit = BehaviorFits(experiment=0, subject_id=0,
                   model_name='momentum_learn_alt_goal', data_type='online',
                   num_sims=100, normalize=False)
fit.fit_behavior(write_results=True)
fit.fit_behavior_by_block_half(write_results=True)
```

Repeat for all participants/models to rebuild comparisons. Each restart uses 100
Optuna trials; fitting all models is expensive and has no fixed sampler seed.
Rebuild the processed cache after changing momentum fits.


Shared data/output paths resolve from the checkout. Task-generation code and its
JSON designs live in `analysis/src/generate/`; run that generator from its own
directory. Processed caches and saved fits are retained as reproduction inputs.
