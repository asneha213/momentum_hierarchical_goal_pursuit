import sys
sys.path.append("..")
from src.models import *
from src.datautils import *
from src.measures import SubjectMeasure
from src.run_model import get_model

import numpy as np
import optuna
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy import stats

from fit_behavior import get_optuna_params


class ModelOptimizer:
    def __init__(self, experiment, model_name, data_type='fmri'):
        self.experiment = experiment
        self.model_name = model_name
        self.data_type = data_type

    def get_mean_performance(self, params):
        seeds = np.random.randint(0, 1000, 30)
        model_results = []
        if self.data_type == 'fmri':
            active_subjects = ACTIVE_SUBJECT_IDS
        else:
            active_subjects = ACTIVE_SUBJECT_IDS_ONLINE
        for subject_id in active_subjects:
            model = get_model(self.model_name, params)
            data = get_subject_data_from_id(self.experiment, subject_id, data_type=self.data_type)
            model_res = model.run_model(data)
            measures = SubjectMeasure(subject_id=subject_id, experiment=self.experiment, model_res=model_res)
            sub_measure = measures.get_task_performance()
            model_results.append(sub_measure)
        return model_results

    def get_mean_measure(self, params, measure_name):
        model_results = []
        for subject_id in ACTIVE_SUBJECT_IDS:
            model = get_model(self.model_name, params)
            data = get_subject_data_from_id(self.experiment, subject_id, data_type=self.data_type)
            model_res = model.run_model(data)
            measures = SubjectMeasure(subject_id=subject_id, experiment=self.experiment, model_res=model_res)
            if measure_name == "optimal_goal":
                sub_measure = measures.get_optimal_goal_condition()
            elif measure_name == "goal_switches":
                sub_measure = measures.get_goal_switches_condition()
            elif measure_name == "goal_selects":
                sub_measure = measures.get_goal_selections_condition()
            model_results.append(sub_measure)
        return model_results

    def get_optuna_objective(self, trial):

        params = get_optuna_params(trial, self.model_name)
        model_measures = self.get_mean_performance(params)
        err = -1 * np.mean(model_measures)
        return err

    def fit_optuna(self):
        best_params = []
        best_vals = []
        for i in range(1):
            study = optuna.create_study()
            study.optimize(self.get_optuna_objective, n_trials=50)
            best_params.append(study.best_params)
            best_vals.append(study.best_value)

        best_params = np.array(best_params)
        best_vals = np.array(best_vals)
        best_params = best_params[np.argmin(best_vals)]
        best_vals = np.min(best_vals)
        print(best_vals, best_params)
        return best_params, best_vals

    def get_model_optimal_params(self):
        if self.data_type == 'online':
            results_dir = MODEL_RESULTS_ONLINE
        else:
            results_dir = MODEL_RESULTS
        
        os.makedirs(results_dir + "sims/", exist_ok=True)
        file_name = results_dir + "sims/" + self.model_name + "_" + str(self.experiment) + "_" + self.data_type + "_optimal_params.pkl"
        res = self.fit_optuna()
        with open(file_name, 'wb') as f:
            pickle.dump(res, f)
        return res

    def load_optimal_params(self):
        """Load optimal parameters from saved file, or return None if not found.
        File naming: {model_name}_{experiment}_{data_type}_optimal_params.pkl in MODEL_RESULTS/sims/.
        """
        if self.data_type == 'online':
            results_dir = MODEL_RESULTS_ONLINE
        else:
            results_dir = MODEL_RESULTS
        
        file_name = results_dir + "sims/" + self.model_name + "_" + str(self.experiment) + "_" + self.data_type + "_optimal_params.pkl"
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                res = pickle.load(f)
                return res[0]  # Return best_params (first element)
        return None

    @staticmethod
    def optimize_all_models(experiment=0, data_type='online', force_reoptimize=False):
        """
        Optimize all models in the list and return their optimal parameters.
        
        Parameters
        ----------
        experiment : int
            Experiment number
        data_type : str
            Type of data ('online' or 'fmri')
        force_reoptimize : bool
            If True, reoptimize even if optimal params exist
        
        Returns
        -------
        dict
            Dictionary mapping model names to their optimal parameters
        """
        models = [
            'momentum_learn_alt_goal',
            'momentum',
            'momentum_with_pros_subgoal',
            'momentum_with_pers_subgoal',
            'prospective',
            'td_persistence',
            'retrospective',
            'resources',
            'progress',
        ]
        
        optimal_params_dict = {}
        
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"Optimizing model: {model_name}")
            print(f"{'='*60}")
            
            optimizer = ModelOptimizer(experiment, model_name, data_type=data_type)
            
            # Check if optimal params already exist
            if not force_reoptimize:
                existing_params = optimizer.load_optimal_params()
                if existing_params is not None:
                    print(f"Found existing optimal parameters for {model_name}")
                    optimal_params_dict[model_name] = existing_params
                    continue
            
            # Optimize the model
            best_params, best_val = optimizer.get_model_optimal_params()
            optimal_params_dict[model_name] = best_params
            print(f"Optimization complete for {model_name}. Best value: {best_val}")
        
        return optimal_params_dict

    @staticmethod
    def plot_task_performance_comparison(experiment=0, data_type='online', 
                                         optimal_params_dict=None, 
                                         save_figure=True,
                                         use_cache=True):
        """
        Plot task performance comparison for all models using their optimal parameters.
        
        Parameters
        ----------
        experiment : int
            Experiment number
        data_type : str
            Type of data ('online' or 'fmri')
        optimal_params_dict : dict, optional
            Dictionary mapping model names to optimal parameters. If None, will load or optimize.
        save_figure : bool
            Whether to save the figure
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        performance_dict : dict
            Dictionary mapping model names to their task performance scores
        """
        models = [
            'momentum_learn_alt_goal',
            'momentum', 
            'momentum_with_pros_subgoal',
            'momentum_with_pers_subgoal',
            'prospective',
            'td_persistence',
            'retrospective',
            'resources',
            'progress',
        ]
        
        model_display_names = {
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal',
            'momentum': 'TD-momentum',
            'momentum_with_pros_subgoal': 'TD-momentum(G)-Prospective(SG)',
            'momentum_with_pers_subgoal': 'TD-momentum(G)-Persistence(SG)', 
            'prospective': 'Prospective',
            'td_persistence': 'TD-persistence',
            'retrospective': 'Retrospective',
            'resources': 'Resources',
            'progress': 'Progress',
        }
        
        # Get optimal parameters if not provided (load from MODEL_RESULTS/sims/ or optimize)
        if optimal_params_dict is None:
            optimal_params_dict = ModelOptimizer.optimize_all_models(experiment, data_type)
        
        # Try to load cached task performance to avoid re-running models.
        cache_file = os.path.join(
            CACHE_DIR,
            f"task_performance_optimal_models_experiment_{experiment}_{data_type}.pkl"
        )
        performance_dict = {}
        all_performances = []

        if use_cache and os.path.exists(cache_file):
            print(f"\nLoading cached task performance from: {cache_file}")
            with open(cache_file, "rb") as f:
                performance_dict = pickle.load(f)
            # Migrate old cache: resources_depth was renamed to progress
            if 'resources_depth' in performance_dict:
                performance_dict['progress'] = performance_dict.pop('resources_depth')
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(performance_dict, f)
        else:
            # Get active subjects based on data type
            if data_type == 'fmri':
                active_subjects = ACTIVE_SUBJECT_IDS
            else:
                active_subjects = ACTIVE_SUBJECT_IDS_ONLINE
            
            # Collect task performance for each model
            for model_name in models:
                if model_name not in optimal_params_dict:
                    print(f"Warning: No optimal parameters found for {model_name}, skipping...")
                    continue
                
                params = optimal_params_dict[model_name]
                model_performances = []
                
                print(f"\nEvaluating {model_name}...")
                for subject_id in active_subjects:
                    try:
                        model = get_model(model_name, params)
                        data = get_subject_data_from_id(experiment, subject_id, data_type=data_type)
                        model_res = model.run_model(data)
                        measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)
                        task_perf = measures.get_task_performance()
                        model_performances.append(task_perf)
                    except Exception as e:
                        print(f"  Error processing subject {subject_id}: {e}")
                        continue
                
                if len(model_performances) > 0:
                    performance_dict[model_name] = model_performances
                    all_performances.extend(model_performances)
                    print(f"  Mean task performance: {np.mean(model_performances):.2f} ± {np.std(model_performances):.2f}")

            # Save to cache if requested
            if use_cache:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(performance_dict, f)
        
        # Sort models by mean performance (descending order)
        model_means = {m: np.mean(performance_dict[m]) for m in performance_dict.keys()}
        sorted_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)
        sorted_model_names = [m[0] for m in sorted_models]
        
        # Create the plot
        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for plotting (sorted by performance)
        plot_data = []
        for model_name in sorted_model_names:
            display_name = model_display_names.get(model_name, model_name)
            perfs = performance_dict[model_name]
            for perf in perfs:
                plot_data.append({'Model': display_name, 'Task Performance': perf})
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(plot_data)
        
        # Create bar plot with error bars (sorted by performance)
        model_order = [model_display_names.get(m, m) for m in sorted_model_names]
        means = [np.mean(performance_dict[m]) for m in sorted_model_names]
        stds = [np.std(performance_dict[m]) for m in sorted_model_names]
        
        x_pos = np.arange(len(model_order))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Color bars based on performance (darker = better)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add individual data points
        for i, model_name in enumerate(sorted_model_names):
            display_name = model_display_names.get(model_name, model_name)
            perfs = performance_dict[model_name]
            x_jitter = np.random.normal(i, 0.05, len(perfs))
            ax.scatter(x_jitter, perfs, alpha=0.4, s=15, color='gray', zorder=3)
        
        # Perform statistical tests between successive models
        p_values = []
        for i in range(len(sorted_model_names) - 1):
            model1_perfs = np.array(performance_dict[sorted_model_names[i]])
            model2_perfs = np.array(performance_dict[sorted_model_names[i + 1]])
            
            # Use independent samples t-test (different models)
            t_stat, p_val = stats.ttest_ind(model1_perfs, model2_perfs)
            p_values.append(p_val)
            print(f"  {model_order[i]} vs {model_order[i+1]}: p = {p_val:.4f}")
        
        # Add significance bars between successive models
        y_offset = max(means) * 0.08  # Spacing between significance bars
        
        # Calculate heights for significance bars (stack them to avoid overlap)
        sig_bar_heights = []
        for i in range(len(sorted_model_names) - 1):
            p_val = p_values[i]
            if p_val < 0.05:
                # Calculate base height for this specific comparison (above the two bars being compared)
                local_max_height = max(means[i] + stds[i], means[i+1] + stds[i+1])
                # Stack bars vertically if there are multiple significant comparisons
                bar_height = local_max_height + y_offset * (len(sig_bar_heights) + 1)
                sig_bar_heights.append((i, bar_height, p_val))
        
        # Draw significance bars
        for i, bar_height, p_val in sig_bar_heights:
            # Determine significance level
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            # Draw horizontal line
            ax.plot([i, i+1], [bar_height, bar_height], 'k-', linewidth=1.5, zorder=2)
            
            # Draw vertical lines at ends
            ax.plot([i, i], [means[i] + stds[i], bar_height], 'k-', linewidth=1.5, zorder=2)
            ax.plot([i+1, i+1], [means[i+1] + stds[i+1], bar_height], 'k-', linewidth=1.5, zorder=2)
            
            # Add significance text
            ax.text((i + i+1) / 2, bar_height + max(means) * 0.01, sig_text,
                   ha='center', va='bottom', fontsize=12, fontweight='bold', zorder=3)
        
        # Calculate max y for plot limits
        if sig_bar_heights:
            max_y = max([h for _, h, _ in sig_bar_heights]) + max(means) * 0.1
        else:
            max_y = max(means) + max(stds) + max(means) * 0.1
        
        # Customize plot
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Task Performance (Total Rewards)', fontsize=14, fontweight='bold')
        ax.set_title(f'Task Performance Comparison Across Models', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis limits to accommodate significance bars
        ax.set_ylim([0, max_y * 1.1])
        
        # Add mean values on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + max(means) * 0.02, f'{mean:.1f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_figure:
            filename = FIGURES_TOPICS + f"task_performance_comparison_experiment_{experiment}_{data_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {filename}")
        
        plt.show()
        
        return fig, performance_dict


if __name__ == "__main__":

    experiment = 0
    data_type = "online"

    # Example 1: Optimize all models
    print("Optimizing all models...")
    optimal_params = ModelOptimizer.optimize_all_models(experiment=experiment, data_type=data_type, force_reoptimize=False)
    
    # Example 2: Plot task performance comparison
    print("\nCreating task performance comparison plot...")
    fig, performance_dict = ModelOptimizer.plot_task_performance_comparison(
        experiment=experiment, 
        data_type=data_type,
        optimal_params_dict=optimal_params,
        save_figure=True,
        use_cache=True
    )
    
    # Example 3: Optimize a single model (original functionality)
    # model_name = "prospective"
    # optimizer = ModelOptimizer(experiment, model_name, data_type=data_type)
    # res = optimizer.get_model_optimal_params()
    # print(res)

