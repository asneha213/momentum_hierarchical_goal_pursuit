import pickle
import os

from src.datautils import *
from reports import *

from fit_behavior import *

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pingouin as pg


def log_likelihood_ratio_test(ll_null, ll_hypothesis, df):
    from scipy.stats import chi2
    llr = 2 * (-ll_hypothesis - (-ll_null))
    p = [1 - chi2.cdf(x, df) for x in llr]
    return np.array(p)


class ModelComparisons:

    def __init__(self, experiment, data_type, save_dict=False, likelihood_type=None):
        self.experiment = experiment
        self.data_type = data_type
        self.save_dict = save_dict
        self.likelihood_type = likelihood_type
        if self.data_type == 'online':
            self.experiment_name = 'H2'
        elif self.data_type == 'fmri':
            self.experiment_name = 'H1'

    def get_model_BIC(self, model_name, AIC=False):
        print(model_name)
        bics = []
        lls = []

        subjects = []

        details = get_experiment_trial_details(experiment=0)
        num_samples = details['num_samples']
        #print("Model name:", model_name)


        if self.data_type == 'online':
            subject_ids = list(set(ACTIVE_SUBJECT_IDS_ONLINE) )
            results_path = MODEL_RESULTS_ONLINE
        elif self.data_type == 'fmri':
            subject_ids = ACTIVE_SUBJECT_IDS
            results_path = MODEL_RESULTS

        for subject_id in subject_ids:
            #print("Processing subject:", subject_id)
            subject_path = results_path + model_name + "_" + str(self.experiment) + "/" + str(subject_id) + ".pkl"
            #print(subject_id, model_name)
            try:
                with open(subject_path, "rb") as f:
                    model_fits = pickle.load(f)
            except:
                continue
            # if self.save_dict:
            #     likelihood_dict = BehaviorFits(experiment=0, subject_id=subject_id, \
            #                                 model_name=model_name, data_type=self.data_type).get_choice_likelihood(model_fits['params'], likelihood_split=True)
            #     with open(results_path + model_name + "_" + str(self.experiment) + "/likelihoods_" + str(subject_id) + "_" + self.data_type + ".pkl", "wb") as f:
            #         pickle.dump(likelihood_dict, f)
            # else:
            #     with open(results_path + model_name + "_" + str(self.experiment) + "/likelihoods_" + str(subject_id) + "_" + self.data_type + ".pkl", "rb") as f:
            #         likelihood_dict = pickle.load(f)
            likelihood_dict = model_fits['likelihood_breakdown']
            params = list(model_fits['params'].values())
            if self.likelihood_type == 'goal':
                ll = likelihood_dict['goal_loglikelihood']
            elif self.likelihood_type == 'subgoal':
                ll = likelihood_dict['subgoal_loglikelihood']
            elif self.likelihood_type == 'total':
                ll = likelihood_dict['total_loglikelihood']
            else:
                ll = model_fits['fits']

            if 'normalized' in model_name:
                if self.likelihood_type == 'goal':
                    ll = likelihood_dict['stay_loglikelihoods'] + likelihood_dict['switch_loglikelihoods']
                elif self.likelihood_type == 'subgoal':
                    ll = likelihood_dict['subgoal_stay_loglikelihoods'] + likelihood_dict['subgoal_switch_loglikelihoods']
                elif self.likelihood_type == 'total':
                    ll = likelihood_dict['stay_loglikelihoods'] + likelihood_dict['switch_loglikelihoods'] + \
                    likelihood_dict['subgoal_stay_loglikelihoods'] + likelihood_dict['subgoal_switch_loglikelihoods']

            if not AIC:
                bic = 2 * ll + (len(params))  * np.log(num_samples)
            else:
                bic = 2 * ll + 2 * (len(params))

            
            bics.append(bic)
            lls.append(ll)
            subjects.append(subject_id)
                
        return np.array(bics), np.array(lls), np.array(subjects)


    def plot_all_models_comparison(self, ax=None, likelihood_type='total'):
        """
        Compare all 9 models using total likelihoods and display both AIC and BIC scores.
        
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates a new figure with subplots.
        likelihood_type : str, optional
            Type of likelihood to use ('total', 'goal', 'subgoal')
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            show = True
        else:
            ax1, ax2 = ax if isinstance(ax, (list, tuple)) else (ax, None)
            show = False
            
        # Store original likelihood_type
        original_likelihood_type = self.likelihood_type
        self.likelihood_type = likelihood_type
        
        # Get all model scores
        model_names = [
            'momentum_learn_alt_goal',
            'momentum', 
            'momentum_with_pros_subgoal',
            'momentum_with_pers_subgoal',
            'prospective',
            'td_persistence',
            'retrospective',
            'resources',
            'progress',
            #'prospective_exp',
            #'prospective_k1',
            #'momentum_learn_alt_goal_normalized',
            #'prospective_normalized',
            #'td_persistence_normalized'
        ]
        
        model_display_names = {
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal(G,SG)',
            'momentum': 'TD-momentum(G,SG)',
            'momentum_with_pros_subgoal': 'TD-momentum(G)-Prospective(SG)',
            'momentum_with_pers_subgoal': 'TD-momentum(G)-Persistence(SG)', 
            'prospective': 'Prospective(G,SG)',
            'td_persistence': 'TD-persistence(G,SG)',
            'retrospective': 'Retrospective(G,SG)',
            'resources': 'Resources(G,SG)',
            'progress': 'Progress(G,SG)',
            'prospective_exp': 'ProspectiveExp(G,SG)',
            'prospective_k1': 'Prospective_k1(G,SG)',
            'momentum_learn_alt_goal_normalized': 'TD-momentum-alt-goal-normalized(G,SG)',
            'prospective_normalized': 'Prospective-normalized(G,SG)',
            'td_persistence_normalized': 'TD-persistence-normalized(G,SG)'
        }
        
        # Collect AIC and BIC scores
        aic_scores = {}
        bic_scores = {}
        all_subjects = None
        
        for model_name in model_names:
            try:
                aic_vals, _, subjects = self.get_model_BIC(model_name, AIC=True)
                bic_vals, _, _ = self.get_model_BIC(model_name, AIC=False)
                
                display_name = model_display_names[model_name]
                aic_scores[display_name] = aic_vals
                bic_scores[display_name] = bic_vals
                
                if all_subjects is None:
                    all_subjects = subjects;
                
                print(model_name, np.mean(aic_vals), np.mean(bic_vals), len(aic_vals))
                    
            except Exception as e:
                print(f"Warning: Could not load model {model_name}: {e}")
                continue
        
        # Restore original likelihood_type
        self.likelihood_type = original_likelihood_type

        if likelihood_type == 'total':
            likelihood_type = 'aggregate'
        
        # Assign colors based on AIC scores
        # Calculate mean AIC for each model
        model_mean_aic = {model: np.mean(scores) for model, scores in aic_scores.items()}
        
        # Fixed colors
        model_colors = {
            'TD-momentum-alt-goal(G,SG)': '#FFFFFF',  # White
            'Prospective(G,SG)': '#000000',  # Black
        }
        
        # Get other models (excluding the two fixed ones)
        other_models = [m for m in model_mean_aic.keys() 
                       if m not in ['TD-momentum-alt-goal(G,SG)', 'Prospective(G,SG)']]
        
        if len(other_models) > 0:
            # Get AIC scores for other models
            other_aics = [model_mean_aic[m] for m in other_models]
            min_aic = min(other_aics)
            max_aic = max(other_aics)
            
            # Normalize AIC scores to 0-1 range (inverse: lower AIC = better = lighter gray)
            # Map: lowest AIC -> lightest gray, highest AIC -> darkest gray
            for model in other_models:
                if max_aic > min_aic:
                    # Normalize: lower AIC gets lower normalized value (lighter gray)
                    normalized = (model_mean_aic[model] - min_aic) / (max_aic - min_aic)
                else:
                    normalized = 0.5  # If all same, use middle gray
                
                # Convert to grayscale: 0 = light gray (#E0E0E0), 1 = dark gray (#404040)
                # Avoid pure white and pure black since those are reserved
                gray_value = int(224 - (normalized * 184))  # Range from 224 (light) to 40 (dark)
                hex_color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
                model_colors[model] = hex_color
        
        # Plot AIC scores
        if ax1 is not None:
            self._plot_sorted_model_scores(ax1, aic_scores, model_colors, 
                                         f"AIC Metric ({likelihood_type} likelihood)", 
                                         "AIC")
        
        # Plot BIC scores
        if ax2 is not None:
            self._plot_sorted_model_scores(ax2, bic_scores, model_colors, 
                                         f"BIC Metric ({likelihood_type} likelihood)", 
                                         "BIC")
        
        if show:
            plt.tight_layout()
            plt.savefig(FIGURES_TOPICS + f"all_models_comparison_{self.experiment_name}_{likelihood_type}.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        return aic_scores, bic_scores
    
    def _plot_sorted_model_scores(self, ax, scores_dict, model_colors, title, score_type):
        """
        Helper method to plot sorted model scores with significance bars.
        
        Parameters
        ----------
        ax : matplotlib axes
            Axes to plot on
        scores_dict : dict
            Dictionary with model names as keys and score arrays as values
        model_colors : dict
            Dictionary mapping model names to colors
        title : str
            Plot title
        score_type : str
            Type of score ('AIC' or 'BIC')
        """
        # Calculate means and sort from worst (highest) to best (lowest)
        model_means = {model: np.sum(scores) for model, scores in scores_dict.items()}
        sorted_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)
        
        models = [item[0] for item in sorted_models]
        mean_scores = [item[1] for item in sorted_models]
        
        # Calculate standard errors
        std_scores = [np.std(scores_dict[model]) / np.sqrt(len(scores_dict[model])) 
                     for model in models]
        
        # Get colors in sorted order
        colors = [model_colors.get(model, '#cccccc') for model in models]
        
        # Calculate p-values for consecutive comparisons
        pvals = []
        for i in range(len(models)-1):
            model1_data = scores_dict[models[i]]
            model2_data = scores_dict[models[i+1]]
            # Skip if either array is empty
            if len(model1_data) == 0 or len(model2_data) == 0:
                pvals.append(np.nan)
                continue
            try:
                pval = pg.ttest(model1_data, model2_data, paired=True)['p-val'].iloc[0]
                pvals.append(pval)
            except Exception as e:
                print(f"Warning: Could not compute p-value between {models[i]} and {models[i+1]}: {e}")
                pvals.append(np.nan)
        
        # Prepare data for plotting function
        ordered_data = {model.lower().replace('-', '_'): scores_dict[model] for model in models}
        
        print(f"\n{score_type} Model Rankings (worst to best):")
        for i, (model, score) in enumerate(zip(models, mean_scores)):
            print(f"{i+1}. {model}: {score:.2f} ± {std_scores[i]:.2f}")
        
        print(f"\nConsecutive p-values: {pvals}")
        
        # Use existing plotting function
        draw_significant_bar_plots(ax, ordered_data, models, mean_scores, std_scores, 
                                 pvals, score_type, title, colors=colors)

    def _plot_sorted_model_scores_vertical(self, ax, scores_dict, model_colors, title, score_type):
        """
        Plot sorted model scores as vertical bars (used only by plot_momentum_models_comparison).
        Same computation as _plot_sorted_model_scores but with ax.bar instead of horizontal bars.
        """
        model_means = {model: np.sum(scores) for model, scores in scores_dict.items()}
        sorted_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)

        models = [item[0] for item in sorted_models]
        mean_scores = np.array([item[1] for item in sorted_models])
        std_scores = np.array([np.std(scores_dict[model]) / np.sqrt(len(scores_dict[model]))
                               for model in models])
        colors = [model_colors.get(model, '#cccccc') for model in models]

        x = np.arange(len(models))
        width = 0.5
        bars = ax.bar(x, mean_scores, width, yerr=std_scores, color=colors, edgecolor='#000000',
                     linewidth=2, alpha=0.6, capsize=5)

        ax.set_ylabel(score_type, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=17, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12, fontweight='bold', rotation=25, ha='right')
        ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if len(mean_scores) > 0:
            y_min = min(mean_scores) * 0.96
            y_max = max(mean_scores) * 1.04
            ax.set_ylim(y_min, y_max)

        # Optional: print rankings and p-values (same as horizontal version)
        pvals = []
        for i in range(len(models) - 1):
            try:
                pval = pg.ttest(scores_dict[models[i]], scores_dict[models[i + 1]], paired=True)['p-val'].iloc[0]
                pvals.append(pval)
            except Exception:
                pvals.append(np.nan)
        print(f"\n{score_type} Model Rankings (worst to best):")
        for i, (model, score) in enumerate(zip(models, mean_scores)):
            print(f"{i+1}. {model}: {score:.2f} ± {std_scores[i]:.2f}")
        print(f"Consecutive p-values: {pvals}")

    def plot_comprehensive_model_comparison(self):
        """
        Create a comprehensive comparison plot showing all 9 models using total likelihood only.
        Shows both AIC and BIC scores side by side.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        print(f"\nProcessing total likelihood comparison...")
        self.plot_all_models_comparison(ax=(ax1, ax2), likelihood_type=self.likelihood_type)
        
        # Add overall title
        fig.suptitle(f"Model Comparisons",
                    fontsize=17, y=0.95, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save the plot
        filename = f"all_models_comparison_{self.experiment_name}_total_only.png"
        plt.savefig(FIGURES_TOPICS + filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved model comparison plot: {filename}")
        plt.show()
        
        return fig

    def plot_momentum_models_comparison(self, ax=None, likelihood_type='total'):
        """
        Compare only momentum, momentum_learn_alt_goal, and momentum_with_softmax
        using total likelihoods and display both AIC and BIC scores.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates a new figure with subplots.
        likelihood_type : str, optional
            Type of likelihood to use ('total', 'goal', 'subgoal')

        Returns
        -------
        aic_scores : dict
            Model display name -> array of AIC scores per subject
        bic_scores : dict
            Model display name -> array of BIC scores per subject
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            show = True
        else:
            ax1, ax2 = ax if isinstance(ax, (list, tuple)) else (ax, None)
            show = False

        original_likelihood_type = self.likelihood_type
        self.likelihood_type = likelihood_type

        model_names = [
            'momentum_learn_alt_goal',
            'momentum',
            'momentum_with_softmax',
        ]

        model_display_names = {
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal(G,SG)',
            'momentum': 'TD-momentum(G,SG)',
            'momentum_with_softmax': 'TD-momentum-softmax(G,SG)',
        }

        aic_scores = {}
        bic_scores = {}

        for model_name in model_names:
            try:
                aic_vals, _, subjects = self.get_model_BIC(model_name, AIC=True)
                bic_vals, _, _ = self.get_model_BIC(model_name, AIC=False)

                display_name = model_display_names[model_name]
                aic_scores[display_name] = aic_vals
                bic_scores[display_name] = bic_vals

                print(model_name, np.mean(aic_vals), np.mean(bic_vals), len(aic_vals))

            except Exception as e:
                print(f"Warning: Could not load model {model_name}: {e}")
                continue

        self.likelihood_type = original_likelihood_type

        if likelihood_type == 'total':
            likelihood_type = 'aggregate'

        # Colors for the three models
        model_colors = {
            'TD-momentum-alt-goal(G,SG)': '#FFFFFF',
            'TD-momentum(G,SG)': '#808080',
            'TD-momentum-softmax(G,SG)': '#404040',
        }

        if ax1 is not None and len(aic_scores) > 0:
            self._plot_sorted_model_scores_vertical(ax1, aic_scores, model_colors,
                                                    f"AIC Metric ({likelihood_type} likelihood)",
                                                    "AIC")

        if ax2 is not None and len(bic_scores) > 0:
            self._plot_sorted_model_scores_vertical(ax2, bic_scores, model_colors,
                                                    f"BIC Metric ({likelihood_type} likelihood)",
                                                    "BIC")

        if show and (len(aic_scores) > 0 or len(bic_scores) > 0):
            plt.tight_layout()
            plt.savefig(FIGURES_TOPICS + f"momentum_models_comparison_{self.experiment_name}_{likelihood_type}.png",
                       dpi=300, bbox_inches='tight')
            plt.show()

        return aic_scores, bic_scores

    def get_model_BIC_by_block_half(self, model_name, block_half='first_half', AIC=False):
        """
        Get BIC/AIC scores for models fit to a specific block half.
        
        Args:
            model_name: Name of the model
            block_half: 'first_half' or 'second_half'
            AIC: If True, return AIC; if False, return BIC
            
        Returns:
            Tuple of (bics/aics, loglikelihoods, subjects)
        """
        print(f"{model_name} - {block_half}")
        bics = []
        lls = []
        subjects = []

        details = get_experiment_trial_details(experiment=0)
        # Use approximately half the samples for block half comparisons
        num_samples = details['num_samples'] // 2

        if self.data_type == 'online':
            subject_ids = list(set(ACTIVE_SUBJECT_IDS_ONLINE))
            results_path = MODEL_RESULTS_ONLINE
        elif self.data_type == 'fmri':
            subject_ids = ACTIVE_SUBJECT_IDS
            results_path = MODEL_RESULTS

        for subject_id in subject_ids:
            # Load from by_block_half directory
            if self.data_type == 'fmri':
                if not hasattr(self, 'normalize') or not getattr(self, 'normalize', False):
                    subject_path = results_path + model_name + "_by_block_half_" + str(self.experiment) + "/" + str(subject_id) + ".pkl"
                else:
                    subject_path = results_path + model_name + "_by_block_half_normalized_" + str(self.experiment) + "/" + str(subject_id) + ".pkl"
            elif self.data_type == 'online':
                # Try absolute path first (MODEL_RESULTS_ONLINE), then fallback to relative path
                # to match where fit_behavior.py saves files
                subject_path_abs = results_path + model_name + "_by_block_half_" + str(self.experiment) + "/" + str(subject_id) + ".pkl"
                subject_path_rel = 'results_online/' + model_name + "_by_block_half_" + str(self.experiment) + "/" + str(subject_id) + ".pkl"
                # Try absolute path first, then relative
                if os.path.exists(subject_path_abs):
                    subject_path = subject_path_abs
                elif os.path.exists(subject_path_rel):
                    subject_path = subject_path_rel
                else:
                    # Neither exists, skip this subject
                    continue
            
            try:
                with open(subject_path, "rb") as f:
                    model_fits_by_half = pickle.load(f)
            except FileNotFoundError:
                # File doesn't exist - this is expected for some subjects
                continue
            except Exception as e:
                # Other errors - print for debugging (only print once per model to avoid spam)
                if subject_id == subject_ids[0]:
                    print(f"Warning: Error loading {subject_path}: {e}")
                continue
            
            # Extract data for the specified block half
            if block_half not in model_fits_by_half:
                continue
                
            half_data = model_fits_by_half[block_half]
            likelihood_dict = half_data['likelihood_breakdown']
            params = list(half_data['params'].values())
            
            if self.likelihood_type == 'goal':
                ll = likelihood_dict['goal_loglikelihood']
            elif self.likelihood_type == 'subgoal':
                ll = likelihood_dict['subgoal_loglikelihood']
            elif self.likelihood_type == 'total':
                ll = likelihood_dict['total_loglikelihood']
            else:
                ll = half_data['fits']

            if not AIC:
                bic = 2 * ll + (len(params)) * np.log(num_samples)
            else:
                bic = 2 * ll + 2 * (len(params))

            bics.append(bic)
            lls.append(ll)
            subjects.append(subject_id)
                
        return np.array(bics), np.array(lls), np.array(subjects)

    def plot_all_models_comparison_by_block_half(self, ax=None, likelihood_type='total', block_half='first_half'):
        """
        Compare all models for a specific block half using total likelihoods and display both AIC and BIC scores.
        
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates a new figure with subplots.
        likelihood_type : str, optional
            Type of likelihood to use ('total', 'goal', 'subgoal')
        block_half : str, optional
            'first_half' or 'second_half'
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            show = True
        else:
            ax1, ax2 = ax if isinstance(ax, (list, tuple)) else (ax, None)
            show = False
            
        # Store original likelihood_type
        original_likelihood_type = self.likelihood_type
        self.likelihood_type = likelihood_type
        
        # Get all model scores
        model_names = [
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
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal(G,SG)',
            'momentum': 'TD-momentum(G,SG)',
            'momentum_with_pros_subgoal': 'TD-momentum(G)-Prospective(SG)',
            'momentum_with_pers_subgoal': 'TD-momentum(G)-Persistence(SG)', 
            'prospective': 'Prospective(G,SG)',
            'td_persistence': 'TD-persistence(G,SG)',
            'retrospective': 'Retrospective(G,SG)',
            'resources': 'Resources(G,SG)',
            'progress': 'Progress(G,SG)',
        }
        
        # Collect AIC and BIC scores
        aic_scores = {}
        bic_scores = {}
        all_subjects = None
        
        for model_name in model_names:
            try:
                aic_vals, _, subjects = self.get_model_BIC_by_block_half(model_name, block_half=block_half, AIC=True)
                bic_vals, _, _ = self.get_model_BIC_by_block_half(model_name, block_half=block_half, AIC=False)
                
                # Skip if no data was found
                if len(aic_vals) == 0:
                    print(f"Warning: No data found for model {model_name} ({block_half})")
                    continue
                
                display_name = model_display_names[model_name]
                aic_scores[display_name] = aic_vals
                bic_scores[display_name] = bic_vals
                
                if all_subjects is None:
                    all_subjects = subjects
                
                print(f"{model_name} ({block_half}): AIC={np.mean(aic_vals):.2f}, BIC={np.mean(bic_vals):.2f}, n={len(aic_vals)}")
                    
            except Exception as e:
                print(f"Warning: Could not load model {model_name} for {block_half}: {e}")
                continue
        
        # Check if we have any data to plot
        if len(aic_scores) == 0:
            print(f"Error: No data found for any models in {block_half}. Make sure models have been fit using fit_behavior_by_block_half().")
            return {}, {}
        
        # Restore original likelihood_type
        self.likelihood_type = original_likelihood_type

        if likelihood_type == 'total':
            likelihood_type = 'aggregate'
        
        # Assign colors based on AIC scores
        model_mean_aic = {model: np.mean(scores) for model, scores in aic_scores.items()}
        
        # Fixed colors
        model_colors = {
            'TD-momentum-alt-goal(G,SG)': '#FFFFFF',  # White
            'Prospective(G,SG)': '#000000',  # Black
        }
        
        # Get other models
        other_models = [m for m in model_mean_aic.keys() 
                       if m not in ['TD-momentum-alt-goal(G,SG)', 'Prospective(G,SG)']]
        
        if len(other_models) > 0:
            other_aics = [model_mean_aic[m] for m in other_models]
            min_aic = min(other_aics)
            max_aic = max(other_aics)
            
            for model in other_models:
                if max_aic > min_aic:
                    normalized = (model_mean_aic[model] - min_aic) / (max_aic - min_aic)
                else:
                    normalized = 0.5
                
                gray_value = int(224 - (normalized * 184))
                hex_color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
                model_colors[model] = hex_color
        
        # Plot AIC scores
        if ax1 is not None:
            self._plot_sorted_model_scores(ax1, aic_scores, model_colors, 
                                         f"AIC Metric ({likelihood_type} likelihood)", 
                                         "AIC")
        
        # Plot BIC scores
        if ax2 is not None:
            self._plot_sorted_model_scores(ax2, bic_scores, model_colors, 
                                         f"BIC Metric ({likelihood_type} likelihood)", 
                                         "BIC")
        
        if show:
            plt.tight_layout()
            plt.savefig(FIGURES_TOPICS + f"all_models_comparison_{self.experiment_name}_{likelihood_type}.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        return aic_scores, bic_scores

    def plot_comprehensive_model_comparison_by_block_half(self):
        """
        Create a comprehensive comparison plot showing all models for first half and second half of blocks.
        Shows only BIC scores side by side.
        """
        fig, (ax_first, ax_second) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Store original likelihood_type
        original_likelihood_type = self.likelihood_type
        likelihood_type = self.likelihood_type
        
        # Get all model scores
        model_names = [
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
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal(G,SG)',
            'momentum': 'TD-momentum(G,SG)',
            'momentum_with_pros_subgoal': 'TD-momentum(G)-Prospective(SG)',
            'momentum_with_pers_subgoal': 'TD-momentum(G)-Persistence(SG)', 
            'prospective': 'Prospective(G,SG)',
            'td_persistence': 'TD-persistence(G,SG)',
            'retrospective': 'Retrospective(G,SG)',
            'resources': 'Resources(G,SG)',
            'progress': 'Progress(G,SG)',
        }
        
        # Collect BIC scores for first half
        print(f"\n{'='*60}")
        print(f"Processing FIRST HALF comparison...")
        print(f"{'='*60}")
        bic_scores_first = {}
        
        for model_name in model_names:
            try:
                bic_vals, _, _ = self.get_model_BIC_by_block_half(model_name, block_half='first_half', AIC=False)
                
                if len(bic_vals) == 0:
                    print(f"Warning: No data found for model {model_name} (first_half)")
                    continue
                
                display_name = model_display_names[model_name]
                bic_scores_first[display_name] = bic_vals
                print(f"{model_name} (first_half): BIC={np.mean(bic_vals):.2f}, n={len(bic_vals)}")
                    
            except Exception as e:
                print(f"Warning: Could not load model {model_name} for first_half: {e}")
                continue
        
        # Collect BIC scores for second half
        print(f"\n{'='*60}")
        print(f"Processing SECOND HALF comparison...")
        print(f"{'='*60}")
        bic_scores_second = {}
        
        for model_name in model_names:
            try:
                bic_vals, _, _ = self.get_model_BIC_by_block_half(model_name, block_half='second_half', AIC=False)
                
                if len(bic_vals) == 0:
                    print(f"Warning: No data found for model {model_name} (second_half)")
                    continue
                
                display_name = model_display_names[model_name]
                bic_scores_second[display_name] = bic_vals
                print(f"{model_name} (second_half): BIC={np.mean(bic_vals):.2f}, n={len(bic_vals)}")
                    
            except Exception as e:
                print(f"Warning: Could not load model {model_name} for second_half: {e}")
                continue
        
        # Check if we have any data to plot
        if len(bic_scores_first) == 0 and len(bic_scores_second) == 0:
            print(f"Error: No data found for any models. Make sure models have been fit using fit_behavior_by_block_half().")
            return fig
        
        if likelihood_type == 'total':
            likelihood_type = 'aggregate'
        
        # Assign colors based on BIC scores (using first half for color scheme)
        if len(bic_scores_first) > 0:
            model_mean_bic = {model: np.mean(scores) for model, scores in bic_scores_first.items()}
        elif len(bic_scores_second) > 0:
            model_mean_bic = {model: np.mean(scores) for model, scores in bic_scores_second.items()}
        else:
            model_mean_bic = {}
        
        # Fixed colors
        model_colors = {
            'TD-momentum-alt-goal(G,SG)': '#FFFFFF',  # White
            'Prospective(G,SG)': '#000000',  # Black
        }
        
        # Get other models
        other_models = [m for m in model_mean_bic.keys() 
                       if m not in ['TD-momentum-alt-goal(G,SG)', 'Prospective(G,SG)']]
        
        if len(other_models) > 0:
            other_bics = [model_mean_bic[m] for m in other_models]
            min_bic = min(other_bics)
            max_bic = max(other_bics)
            
            for model in other_models:
                if max_bic > min_bic:
                    normalized = (model_mean_bic[model] - min_bic) / (max_bic - min_bic)
                else:
                    normalized = 0.5
                
                gray_value = int(224 - (normalized * 184))
                hex_color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
                model_colors[model] = hex_color
        
        # Plot BIC scores for first half
        if len(bic_scores_first) > 0:
            self._plot_sorted_model_scores(ax_first, bic_scores_first, model_colors, 
                                         f"First Half of Block - BIC ({likelihood_type} likelihood)", 
                                         "BIC")
        else:
            ax_first.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                         transform=ax_first.transAxes, fontsize=14)
            ax_first.set_title('First Half of Block - BIC', fontsize=14, fontweight='bold')
        
        # Plot BIC scores for second half
        if len(bic_scores_second) > 0:
            self._plot_sorted_model_scores(ax_second, bic_scores_second, model_colors, 
                                         f"Second Half of Block - BIC ({likelihood_type} likelihood)", 
                                         "BIC")
        else:
            ax_second.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                          transform=ax_second.transAxes, fontsize=14)
            ax_second.set_title('Second Half of Block - BIC', fontsize=14, fontweight='bold')
        
        # Add overall title
        fig.suptitle(f"Model Comparisons by Block Half (BIC)",
                    fontsize=17, y=1.05, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the plot
        filename = f"all_models_comparison_by_block_half_{self.experiment_name}_{self.likelihood_type}.png"
        plt.savefig(FIGURES_TOPICS + filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved model comparison plot: {filename}")
        plt.show()
        
        return fig


if __name__ == "__main__":
    # Example usage of the comprehensive model comparison
    model_comparisons = ModelComparisons(experiment=0, data_type='online', save_dict=False, likelihood_type='total')
    
    # Compare all 9 models across different likelihood types
    #model_comparisons.plot_comprehensive_model_comparison()
    
    # Compare models by block half (first half vs second half)
    #model_comparisons.plot_comprehensive_model_comparison_by_block_half()


    model_comparisons.plot_momentum_models_comparison()
    
    # Or compare just for total likelihood only
    # aic_scores, bic_scores = model_comparisons.plot_all_models_comparison(likelihood_type='total')
    
    # Original comparison methods still available
    # model_comparisons.plot_model_fits_experiment()
    # model_comparisons.plot_model_fits_experiment_momentum()

