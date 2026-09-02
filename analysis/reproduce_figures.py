"""Regenerate the final-final manuscript panels; see README.md for provenance.

This adapter configures the legacy imports/paths without changing the analysis
calculations. Figure 1 is supplied artwork and is intentionally not regenerated.
"""
import argparse
import importlib
from pathlib import Path
import shutil
import sys

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
# panel: (module, class, method, keyword arguments, original output filename)
PANELS = {
    '2A': ('plot_measures', 'PlotMeasures', 'plot_goal_selection_per_block', {}, 'goal_selection_per_block_H2.png'),
    '2B': ('plot_measures', 'PlotMeasures', 'plot_subgoal_selection_per_block', {}, 'subgoal_selection_per_block_H2.png'),
    '2C': ('plot_measures', 'PlotMeasures', 'plot_viable_goal_selection_high_low', {}, 'viable_goal_subgoal_selection_high_low_H2.png'),
    '3A': ('plot_measures', 'PlotMeasures', 'plot_goal_transition_heatmaps', {}, 'goal_transition_heatmaps_experiment_H2.png'),
    '3B': ('plot_measures', 'PlotMeasures', 'plot_subgoal_transition_heatmaps', {'condition_on_goal_stay': True}, 'subgoal_transition_heatmaps_experiment_H2_goal_stay.png'),
    '3C': ('plot_measures', 'PlotMeasures', 'plot_goal_transition_heatmaps_by_block_half', {}, 'goal_transition_heatmaps_by_block_half_experiment_H2.png'),
    '4A': ('plot_measures', 'PlotMeasures', 'plot_switching_characteristics', {'goal_type': 'goal'}, 'switching_characteristics_goal_combined_experiment_H2_lineplot.png'),
    '4B': ('plot_measures', 'PlotMeasures', 'plot_switching_characteristics', {'goal_type': 'goal'}, 'switching_characteristics_goal_by_outcome_experiment_H2_barplot.png'),
    '5': ('plot_measures', 'PlotMeasures', 'plot_switching_characteristics', {'goal_type': 'goal'}, 'depth_first_analysis_goal_experiment_H2.png'),
    '6A': ('plot_rt_measures', 'PlotRTMeasures', 'plot_goal_rt_by_switching_pattern', {}, 'goal_and_subgoal_rt_by_switching_pattern_experiment_H2.png'),
    '6B': ('plot_measures', 'PlotMeasures', 'plot_switching_characteristics', {'goal_type': 'goal'}, 'depth_first_vs_rt_increase_experiment_H2.png'),
    '6C': ('plot_rt_measures', 'PlotRTMeasures', 'plot_goal_rt_by_repetition', {}, 'goal_rt_by_repetition_experiment_H2.png'),
    '6D': ('plot_rt_measures', 'PlotRTMeasures', 'plot_subgoal_rt_by_repetition', {}, 'subgoal_rt_by_repetition_experiment_H2.png'),
    '7A': ('plot_measures', 'PlotMeasures', 'plot_stay_proportion_with_progress_combined', {}, 'Figure7A.png'),
    '7B': ('plot_measures', 'PlotMeasures', 'plot_stay_by_block_type_combined', {}, 'Figure7B.png'),
    '7C': ('plot_measures', 'PlotMeasures', 'plot_goal_selection_related_goal_progress', {'simulate': False}, 'Figure9C.png'),
    '7D': ('plot_measures', 'PlotMeasures', 'plot_subgoal_selection_related_subgoal_progress', {'simulate': False}, 'Figure9D.png'),
    '8A': ('model_comparisons', 'ModelComparisons', 'plot_comprehensive_model_comparison', {}, 'all_models_comparison_H2_total_only.png'),
    '8B': ('model_comparisons', 'ModelComparisons', 'plot_comprehensive_model_comparison_by_block_half', {}, 'all_models_comparison_by_block_half_H2_total.png'),
    '9A': ('plot_measures_mixed_effects', 'PlotMeasuresMixedEffects', 'plot_figure9a', {}, 'Figure9A.png'),
    '9B': ('plot_measures_mixed_effects', 'PlotMeasuresMixedEffects', 'plot_forest_plot_goal_rt_switch_vs_stay', {}, 'forest_plot_goal_rt_switch_vs_stay.png'),
}


def configure_paths(output):
    """Set paths before wildcard imports copy constants into plotting modules."""
    sys.path.insert(0, str(ANALYSIS / 'src'))
    sys.path.insert(0, str(ANALYSIS))
    datautils = importlib.import_module('src.datautils')
    # The legacy code imports this module under both names. Share one configured
    # object so cache and raw-data readers resolve to the same checkout.
    sys.modules['datautils'] = datautils
    for name, path in {
        'ONLINE_DATA': ROOT / 'data',
        'CACHE_DIR': ANALYSIS / 'cache',
        'MODEL_RESULTS_ONLINE': ANALYSIS / 'results_online',
        'MODEL_RESULTS': ANALYSIS / 'results',
        'FIGURES': output,
        'FIGURES_TOPICS': output,
    }.items():
        # Some callers concatenate filenames without inserting a separator.
        setattr(datautils, name, str(path.resolve()) + '/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', nargs='+', choices=list(PANELS), default=list(PANELS))
    parser.add_argument('--list', action='store_true', help='List sources without importing scientific packages')
    parser.add_argument('--output', type=Path, default=ANALYSIS / 'figures' / 'reproduced')
    args = parser.parse_args()
    if args.list:
        for panel, (module, cls, method, kwargs, filename) in PANELS.items():
            print(f'Figure{panel}.png <- {module}.{cls}.{method}({kwargs}) -> {filename}')
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Refuse stale output: a method can return early without writing a figure.
    expected = {output / PANELS[p][-1] for p in args.panels}
    expected.update(output / f'Figure{p}.png' for p in args.panels)
    if any(p.exists() for p in expected):
        parser.error('Output files already exist; choose a fresh --output directory.')
    cache = ANALYSIS / 'cache' / 'all_subjects_model_parameters_online.csv'
    if any(not p.startswith('8') for p in args.panels) and not cache.is_file():
        parser.error(f'Missing {cache}; see README.md for explicit cache rebuilding.')

    configure_paths(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    objects, completed = {}, set()
    for panel in args.panels:
        module, cls, method, kwargs, filename = PANELS[panel]
        key = (module, method, tuple(kwargs.items()))
        if key not in completed:
            if module not in objects:
                klass = getattr(importlib.import_module(module), cls)
                if module == 'model_comparisons':
                    objects[module] = klass(experiment=0, data_type='online', save_dict=False, likelihood_type='total')
                elif module == 'plot_measures_mixed_effects':
                    objects[module] = klass  # These forest-plot methods are static.
                else:
                    objects[module] = klass(data_type='online', read_from_csv=True)
            print(f'Regenerating Figure {panel}: {method}', flush=True)
            getattr(objects[module], method)(**kwargs)
            plt.close('all')
            completed.add(key)
        source = output / filename
        if not source.is_file():
            raise RuntimeError(f'{method} did not produce {source}; inspect its diagnostic output.')
        destination = output / f'Figure{panel}.png'
        if source != destination:
            shutil.copy2(source, destination)
        print(f'Created {destination}', flush=True)


if __name__ == '__main__':
    main()
