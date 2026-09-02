"""Goal-choice and RT regressions using model-derived trial predictors."""

from plot_measures import *

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class PlotMeasuresMixedEffects(PlotMeasures):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_type == "online":
            self.experiment = "1"
        elif self.data_type == "fmri":
            self.experiment = "2"


    @staticmethod
    def _draw_forest_panel(ax, result, title, xlabel, coef_fmt='.1f', show_ylabels=True):
        """Draw a forest plot panel onto the given axes."""
        y_positions = np.array([0.5, 1.5])

        coefs = result['coefs']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        p_values = result['p_values']
        regressor_corr = result['regressor_corr']

        y_min = min(ci_lower) - (max(ci_upper) - min(ci_lower)) * 0.1
        y_max = max(ci_upper) + (max(ci_upper) - min(ci_lower)) * 0.1

        for j, (low, high) in enumerate(zip(ci_lower, ci_upper)):
            ax.plot([low, high], [y_positions[j], y_positions[j]], 'k-', linewidth=3, alpha=0.8)

        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        sizes = [100 if p < 0.05 else 60 for p in p_values]

        ax.scatter(coefs, y_positions, c=colors, s=sizes,
                   zorder=5, alpha=0.8, edgecolors='black', linewidth=1)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_yticks(y_positions)
        wrapped_labels = ['Switching\nCosts', 'Momentum\nAdvantage']
        if show_ylabels:
            ax.set_yticklabels(wrapped_labels, fontsize=20, fontweight='bold')
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlim(y_min, y_max)
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=16)
        ax.grid(True, alpha=0.3, axis='x')

        for j, (coef, p_val, ci_low, ci_high) in enumerate(zip(coefs, p_values, ci_lower, ci_upper)):
            if p_val < 0.001:
                sig_star = '***'
            elif p_val < 0.01:
                sig_star = '**'
            elif p_val < 0.05:
                sig_star = '*'
            else:
                sig_star = 'ns'

            x_pos = y_max - (y_max - y_min) * 0.15

            ax.text(x_pos, y_positions[j] + 0.1, f'β = {coef:{coef_fmt}}',
                    fontsize=18, va='center', fontweight='bold')
            ax.text(x_pos, y_positions[j] - 0.1, f'[{ci_low:{coef_fmt}}, {ci_high:{coef_fmt}}]',
                    fontsize=16, va='center', style='italic')

            if p_val < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_val:.3f}'

            ax.text(x_pos, y_positions[j] - 0.3, f'{p_text} {sig_star}',
                    fontsize=17, va='center',
                    color='red' if p_val < 0.05 else 'black',
                    fontweight='bold' if p_val < 0.05 else 'normal')

        ax.text(0.02, 0.1, f'n = {result["n_subjects"]}, r = {regressor_corr:.3f}',
                transform=ax.transAxes, fontsize=18, weight='bold',
                verticalalignment='bottom', horizontalalignment='left')

        ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')

    @staticmethod
    def _fit_rt_mixed_effects_h2(df):
        """Fit mixed effects model for goal RT in experiment H2."""
        if 'probe_rt' in df.columns:
            rt_column = 'probe_rt'
        elif 'goal_rt' in df.columns:
            rt_column = 'goal_rt'
        else:
            print(f"Warning: No RT column found for experiment H2")
            return None

        df_model = df.dropna(subset=[
            rt_column, 'chosen_goal_perseveration', 'chosen_goal_value', 'subject_id',
            'chosen_subgoal_value', 'chosen_subgoal_perseveration', 'switching_costs', 'momentum_advantage'
        ]).copy()

        if len(df_model) == 0:
            print(f"Warning: No valid data for experiment H2")
            return None

        df_model['goal_rt'] = df_model[rt_column]
        df_model['subgoal_momentum_z'] = (df_model['chosen_subgoal_value'] - df_model['chosen_subgoal_value'].mean()) / df_model['chosen_subgoal_value'].std()
        df_model['subgoal_perseveration_z'] = (df_model['chosen_subgoal_perseveration'] - df_model['chosen_subgoal_perseveration'].mean()) / df_model['chosen_subgoal_perseveration'].std()
        df_model['goal_perseveration_z'] = (df_model['switching_costs'] - df_model['switching_costs'].mean()) / df_model['switching_costs'].std()
        df_model['goal_momentum_z'] = (df_model['momentum_advantage'] - df_model['momentum_advantage'].mean()) / df_model['momentum_advantage'].std()
        regressor_corr = df_model['goal_perseveration_z'].corr(df_model['goal_momentum_z'])

        try:
            model = smf.mixedlm("goal_rt ~ goal_perseveration_z + goal_momentum_z ", df_model, groups=df_model["subject_id"])
            result = model.fit()

            predictors = ['goal_perseveration_z', 'goal_momentum_z']
            predictor_labels = ['Goal Perseveration', 'Goal Momentum']

            coefs = [result.params[pred] for pred in predictors]
            ci_lower = [result.conf_int().loc[pred, 0] for pred in predictors]
            ci_upper = [result.conf_int().loc[pred, 1] for pred in predictors]
            t_stats = [result.params[pred] / result.bse[pred] for pred in predictors]
            p_values = [2 * (1 - stats.t.cdf(abs(t), result.df_resid)) for t in t_stats]

            return {
                'coefs': coefs,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_values': p_values,
                'regressor_corr': regressor_corr,
                'predictor_labels': predictor_labels,
                'n_subjects': df_model['subject_id'].nunique(),
                'n_trials': len(df_model)
            }
        except Exception as e:
            print(f"Error fitting model for H2: {e}")
            return None

    @staticmethod
    def _fit_goal_choice_mixed_effects_h2(df):
        """Fit logistic regression for goal choice in experiment H2."""
        df_model = df.dropna(subset=[
            'goal_rt', 'chosen_goal_perseveration', 'chosen_goal_value', 'subject_id',
            'chosen_subgoal_value', 'chosen_subgoal_perseveration', 'switching_costs', 'momentum_advantage'
        ]).copy()

        if len(df_model) == 0:
            print(f"Warning: No valid data for experiment H2 goal choice")
            return None

        df_model['goal_switch_binary'] = (df_model['goal_switch'] == 1).astype(int)
        df_model['goal_perseveration_z'] = (df_model['switching_costs'] - df_model['switching_costs'].mean()) / df_model['switching_costs'].std()
        df_model['goal_momentum_z'] = (df_model['momentum_advantage'] - df_model['momentum_advantage'].mean()) / df_model['momentum_advantage'].std()
        regressor_corr = df_model['goal_perseveration_z'].corr(df_model['goal_momentum_z'])

        try:
            # Figure 9A choice panel uses pooled logistic regression; despite
            # this module name, only the RT panel fits subject random intercepts.
            model = smf.logit("goal_switch_binary ~ goal_perseveration_z + goal_momentum_z", df_model)
            result = model.fit()

            predictors = ['goal_perseveration_z', 'goal_momentum_z']
            predictor_labels = ['Goal Perseveration', 'Goal Momentum']

            coefs = [result.params[pred] for pred in predictors]
            ci_lower = [result.conf_int().loc[pred, 0] for pred in predictors]
            ci_upper = [result.conf_int().loc[pred, 1] for pred in predictors]
            z_stats = [result.params[pred] / result.bse[pred] for pred in predictors]
            p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_stats]

            return {
                'coefs': coefs,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_values': p_values,
                'regressor_corr': regressor_corr,
                'predictor_labels': predictor_labels,
                'n_subjects': df_model['subject_id'].nunique(),
                'n_trials': len(df_model)
            }
        except Exception as e:
            print(f"Error fitting goal choice model for H2: {e}")
            return None

    @staticmethod
    def plot_forest_plot_experiment_h2_only(ax=None, save=True, show=True):
        """
        Create forest plot for experiment H2 only.
        If ax is provided, draw onto that axes (for combined figures).
        """
        import os
        import pandas as pd
        
        # Define experiment H2 CSV file
        file_path = CACHE_DIR + 'all_subjects_model_parameters_online.csv'
        
        # Load dataset
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print(f"Loaded H2: {len(df)} rows")
        
        result = PlotMeasuresMixedEffects._fit_rt_mixed_effects_h2(df)
        
        if result is None:
            print("No valid results for experiment H2")
            return None
        
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        PlotMeasuresMixedEffects._draw_forest_panel(
            ax, result, title='Goal Reaction Time', xlabel='Effect Size (ms)', coef_fmt='.1f'
        )

        if fig is not None:
            plt.tight_layout()
            if save:
                plt.savefig(FIGURES_TOPICS + "/forest_plot_experiment_h2_only.png", dpi=300, bbox_inches='tight')
            if show:
                plt.show()
        
        # Print summary statistics
        coefs = result['coefs']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        p_values = result['p_values']
        regressor_corr = result['regressor_corr']
        predictor_labels = result['predictor_labels']

        print(f"\n{'='*80}")
        print(f"EXPERIMENT H2 MIXED EFFECTS MODEL RESULTS")
        print(f"{'='*80}")
        print(f"{'N Subjects':<12} {'N Trials':<12} {'Regressor r':<15}")
        print(f"{result['n_subjects']:<12} {result['n_trials']:<12} {regressor_corr:<15.3f}")
        print(f"{'-'*80}")
        print(f"{'Predictor':<20} {'Coef (ms)':<12} {'95% CI':<20} {'p-value':<12} {'Sig':<6}")
        print(f"{'-'*80}")
        
        for pred, coef, p_val, ci_low, ci_high in zip(predictor_labels, coefs, p_values, ci_lower, ci_upper):
            if p_val < 0.001:
                sig_star = '***'
                p_display = '< 0.001'
            elif p_val < 0.01:
                sig_star = '**'
                p_display = f'{p_val:.3f}'
            elif p_val < 0.05:
                sig_star = '*'
                p_display = f'{p_val:.3f}'
            else:
                sig_star = 'ns'
                p_display = f'{p_val:.3f}'
            
            print(f"{pred:<20} {coef:<12.1f} [{ci_low:.1f}, {ci_high:.1f}] {p_display:<12} {sig_star:<6}")
        
        return fig if fig is not None else result

    # final-final manuscript: Figure 9B.
    # 9B: forest_plot_goal_rt_switch_vs_stay.png
    # See analysis/README.md and reproduce_figures.py for inputs and panel aliases.
    @staticmethod
    def plot_forest_plot_goal_rt_switch_vs_stay():
        """
        Create forest plot for goal RT effects separated by switch vs stay trials
        Two subplots: one for goal switches, one for goal stays
        """
        import os
        import pandas as pd
        
        # Define experiment H2 CSV file
        file_path = CACHE_DIR + 'all_subjects_model_parameters_online.csv'

        # Load dataset
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print(f"Loaded H2: {len(df)} rows")
        
        def fit_mixed_effects_model_switch_stay(df, switch_trials=True):
            """Fit mixed effects model for switch or stay trials"""
            # Check which RT column to use
            if 'probe_rt' in df.columns:
                rt_column = 'probe_rt'
            elif 'goal_rt' in df.columns:
                rt_column = 'goal_rt'
            else:
                print(f"Warning: No RT column found for experiment H2")
                return None

            # Create goal switch indicator
            df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
            df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
            df['is_goal_switch'] = (df['goal_selected'] != df['prev_goal'])
            
            # Filter for switch or stay trials
            if switch_trials:
                df_filtered = df[df['is_goal_switch'] == True].copy()
                trial_type = 'Switch'
            else:
                df_filtered = df[df['is_goal_switch'] == False].copy()
                trial_type = 'Stay'
                
            df_model = df_filtered.dropna(subset=[
                rt_column, 'chosen_goal_perseveration', 'chosen_goal_value', 'subject_id', 
                'chosen_subgoal_value', 'chosen_subgoal_perseveration', 'switching_costs', 'momentum_advantage'
            ]).copy()
            
            if len(df_model) == 0:
                print(f"Warning: No valid data for {trial_type} trials")
                return None
                
            # Rename RT column to 'goal_rt' for consistency
            df_model['goal_rt'] = df_model[rt_column]
                
            # Standardize variables
            df_model['subgoal_momentum_z'] = (df_model['chosen_subgoal_value'] - df_model['chosen_subgoal_value'].mean()) / df_model['chosen_subgoal_value'].std()
            df_model['subgoal_perseveration_z'] = (df_model['chosen_subgoal_perseveration'] - df_model['chosen_subgoal_perseveration'].mean()) / df_model['chosen_subgoal_perseveration'].std()
            df_model['goal_perseveration_z'] = (df_model['switching_costs'] - df_model['switching_costs'].mean()) / df_model['switching_costs'].std()
            df_model['goal_momentum_z'] = (df_model['momentum_advantage'] - df_model['momentum_advantage'].mean()) / df_model['momentum_advantage'].std()
            
            # Calculate regressor correlation
            regressor_corr = df_model['goal_perseveration_z'].corr(df_model['goal_momentum_z'])
            
            # Fit mixed-effects model
            try:
                model = smf.mixedlm("goal_rt ~ goal_perseveration_z + goal_momentum_z", df_model, groups=df_model["subject_id"])
                result = model.fit()
                
                # Extract results
                predictors = ['goal_perseveration_z', 'goal_momentum_z']
                predictor_labels = ['Goal Perseveration', 'Goal Momentum']
                
                coefs = [result.params[pred] for pred in predictors]
                ci_lower = [result.conf_int().loc[pred, 0] for pred in predictors]
                ci_upper = [result.conf_int().loc[pred, 1] for pred in predictors]
                std_errors = [result.bse[pred] for pred in predictors]
                t_stats = [result.params[pred] / result.bse[pred] for pred in predictors]
                p_values = [2 * (1 - stats.t.cdf(abs(t), result.df_resid)) for t in t_stats]
                
                return {
                    'coefs': coefs,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_values': p_values,
                    'regressor_corr': regressor_corr,
                    'predictor_labels': predictor_labels,
                    'n_subjects': df_model['subject_id'].nunique(),
                    'n_trials': len(df_model),
                    'trial_type': trial_type
                }
            except Exception as e:
                print(f"Error fitting model for {trial_type} trials: {e}")
                return None
        
        # Fit models for both switch and stay trials
        switch_result = fit_mixed_effects_model_switch_stay(df, switch_trials=True)
        stay_result = fit_mixed_effects_model_switch_stay(df, switch_trials=False)
        
        if switch_result is None or stay_result is None:
            print("No valid results for switch/stay analysis")
            return None
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Set up y positions for the two variables
        y_positions = np.array([0.5, 1.5])
        
        # Plot for switch trials
        coefs_switch = switch_result['coefs']
        ci_lower_switch = switch_result['ci_lower']
        ci_upper_switch = switch_result['ci_upper']
        p_values_switch = switch_result['p_values']
        regressor_corr_switch = switch_result['regressor_corr']
        predictor_labels = switch_result['predictor_labels']
        
        # Find y-axis limits for switch plot
        y_min_switch = min(ci_lower_switch) - (max(ci_upper_switch) - min(ci_lower_switch)) * 0.1
        y_max_switch = max(ci_upper_switch) + (max(ci_upper_switch) - min(ci_lower_switch)) * 0.1
        
        # Plot confidence intervals as horizontal lines for switch trials
        for j, (low, high) in enumerate(zip(ci_lower_switch, ci_upper_switch)):
            ax1.plot([low, high], [y_positions[j], y_positions[j]], 'k-', linewidth=3, alpha=0.8)
        
        # Plot point estimates as circles, colored by significance for switch trials
        colors_switch = ['red' if p < 0.05 else 'gray' for p in p_values_switch]
        sizes_switch = [100 if p < 0.05 else 60 for p in p_values_switch]
        
        ax1.scatter(coefs_switch, y_positions, c=colors_switch, s=sizes_switch, 
                   zorder=5, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add vertical line at zero for switch trials
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Customize axes for switch trials
        ax1.set_yticks(y_positions)
        wrapped_labels = ['Switching\nCosts', 'Momentum\nAdvantage']
        ax1.set_yticklabels(wrapped_labels, fontsize=20, fontweight='bold')
        ax1.set_ylim(-0.5, 2.5)
        ax1.set_xlim(y_min_switch, y_max_switch)
        ax1.set_title('Goal RT: Switch Trials', fontsize=20, fontweight='bold')
        ax1.tick_params(axis='x', labelsize=16)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add effect size and p-value annotations for switch trials
        for j, (coef, p_val, ci_low, ci_high) in enumerate(zip(coefs_switch, p_values_switch, ci_lower_switch, ci_upper_switch)):
            # Determine significance stars
            if p_val < 0.001:
                sig_star = '***'
            elif p_val < 0.01:
                sig_star = '**'
            elif p_val < 0.05:
                sig_star = '*'
            else:
                sig_star = 'ns'
            
            # Position text to the right of the plot
            x_pos = y_max_switch - (y_max_switch - y_min_switch) * 0.15
            
            # Add coefficient and CI text
            ax1.text(x_pos, y_positions[j] + 0.1, f'β = {coef:.1f}', 
                    fontsize=18, va='center', fontweight='bold')
            ax1.text(x_pos, y_positions[j] - 0.1, f'[{ci_low:.1f}, {ci_high:.1f}]', 
                    fontsize=16, va='center', style='italic')
            
            # Format p-value properly
            if p_val < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_val:.3f}'
            
            ax1.text(x_pos, y_positions[j] - 0.3, f'{p_text} {sig_star}', 
                    fontsize=17, va='center', 
                    color='red' if p_val < 0.05 else 'black',
                    fontweight='bold' if p_val < 0.05 else 'normal')
        
        # Add sample size and correlation information for switch trials
        ax1.text(0.02, 0.1, f'n = {switch_result["n_subjects"]}, r = {regressor_corr_switch:.3f}', 
                transform=ax1.transAxes, fontsize=18, weight='bold',
                verticalalignment='bottom', horizontalalignment='left')
        
        # Plot for stay trials
        coefs_stay = stay_result['coefs']
        ci_lower_stay = stay_result['ci_lower']
        ci_upper_stay = stay_result['ci_upper']
        p_values_stay = stay_result['p_values']
        regressor_corr_stay = stay_result['regressor_corr']
        
        # Find y-axis limits for stay plot
        y_min_stay = min(ci_lower_stay) - (max(ci_upper_stay) - min(ci_lower_stay)) * 0.1
        y_max_stay = max(ci_upper_stay) + (max(ci_upper_stay) - min(ci_lower_stay)) * 0.1
        
        # Plot confidence intervals as horizontal lines for stay trials
        for j, (low, high) in enumerate(zip(ci_lower_stay, ci_upper_stay)):
            ax2.plot([low, high], [y_positions[j], y_positions[j]], 'k-', linewidth=3, alpha=0.8)
        
        # Plot point estimates as circles, colored by significance for stay trials
        colors_stay = ['red' if p < 0.05 else 'gray' for p in p_values_stay]
        sizes_stay = [100 if p < 0.05 else 60 for p in p_values_stay]
        
        ax2.scatter(coefs_stay, y_positions, c=colors_stay, s=sizes_stay, 
                   zorder=5, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add vertical line at zero for stay trials
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Customize axes for stay trials
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(wrapped_labels, fontsize=20, fontweight='bold')
        ax2.set_ylim(-0.5, 2.5)
        ax2.set_xlim(y_min_stay, y_max_stay)
        ax2.set_title('Goal RT: Stay Trials', fontsize=20, fontweight='bold')
        ax2.tick_params(axis='x', labelsize=16)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add effect size and p-value annotations for stay trials
        for j, (coef, p_val, ci_low, ci_high) in enumerate(zip(coefs_stay, p_values_stay, ci_lower_stay, ci_upper_stay)):
            # Determine significance stars
            if p_val < 0.001:
                sig_star = '***'
            elif p_val < 0.01:
                sig_star = '**'
            elif p_val < 0.05:
                sig_star = '*'
            else:
                sig_star = 'ns'
            
            # Position text to the right of the plot
            x_pos = y_max_stay - (y_max_stay - y_min_stay) * 0.15
            
            # Add coefficient and CI text
            ax2.text(x_pos, y_positions[j] + 0.1, f'β = {coef:.1f}', 
                    fontsize=18, va='center', fontweight='bold')
            ax2.text(x_pos, y_positions[j] - 0.1, f'[{ci_low:.1f}, {ci_high:.1f}]', 
                    fontsize=16, va='center', style='italic')
            
            # Format p-value properly
            if p_val < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_val:.3f}'
            
            ax2.text(x_pos, y_positions[j] - 0.3, f'{p_text} {sig_star}', 
                    fontsize=17, va='center', 
                    color='red' if p_val < 0.05 else 'black',
                    fontweight='bold' if p_val < 0.05 else 'normal')
        
        # Add sample size and correlation information for stay trials
        ax2.text(0.02, 0.1, f'n = {stay_result["n_subjects"]}, r = {regressor_corr_stay:.3f}', 
                transform=ax2.transAxes, fontsize=18, weight='bold',
                verticalalignment='bottom', horizontalalignment='left')
        
        # Add overall x-axis label
        fig.text(0.5, 0.02, 'Effect Size (ms)', ha='center', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(FIGURES_TOPICS + "/forest_plot_goal_rt_switch_vs_stay.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\n{'='*100}")
        print(f"GOAL RT MIXED EFFECTS MODEL RESULTS: SWITCH VS STAY TRIALS")
        print(f"{'='*100}")
        print(f"{'Trial Type':<15} {'N Subjects':<12} {'N Trials':<12} {'Regressor r':<15}")
        print(f"{'Switch':<15} {switch_result['n_subjects']:<12} {switch_result['n_trials']:<12} {switch_result['regressor_corr']:<15.3f}")
        print(f"{'Stay':<15} {stay_result['n_subjects']:<12} {stay_result['n_trials']:<12} {stay_result['regressor_corr']:<15.3f}")
        print(f"{'-'*100}")
        print(f"{'Trial Type':<15} {'Predictor':<20} {'Coef (ms)':<12} {'95% CI':<20} {'p-value':<12} {'Sig':<6}")
        print(f"{'-'*100}")
        
        # Print switch results
        for pred, coef, p_val, ci_low, ci_high in zip(predictor_labels, coefs_switch, p_values_switch, ci_lower_switch, ci_upper_switch):
            if p_val < 0.001:
                sig_star = '***'
                p_display = '< 0.001'
            elif p_val < 0.01:
                sig_star = '**'
                p_display = f'{p_val:.3f}'
            elif p_val < 0.05:
                sig_star = '*'
                p_display = f'{p_val:.3f}'
            else:
                sig_star = 'ns'
                p_display = f'{p_val:.3f}'
            
            print(f"{'Switch':<15} {pred:<20} {coef:<12.1f} [{ci_low:.1f}, {ci_high:.1f}] {p_display:<12} {sig_star:<6}")
        
        # Print stay results
        for pred, coef, p_val, ci_low, ci_high in zip(predictor_labels, coefs_stay, p_values_stay, ci_lower_stay, ci_upper_stay):
            if p_val < 0.001:
                sig_star = '***'
                p_display = '< 0.001'
            elif p_val < 0.01:
                sig_star = '**'
                p_display = f'{p_val:.3f}'
            elif p_val < 0.05:
                sig_star = '*'
                p_display = f'{p_val:.3f}'
            else:
                sig_star = 'ns'
                p_display = f'{p_val:.3f}'
            
            print(f"{'Stay':<15} {pred:<20} {coef:<12.1f} [{ci_low:.1f}, {ci_high:.1f}] {p_display:<12} {sig_star:<6}")
        
        return fig


    @staticmethod
    def plot_forest_plot_goal_choice_experiment_h2_only(ax=None, save=True, show=True):
        """
        Create forest plot for goal choice analysis in experiment H2 only.
        If ax is provided, draw onto that axes (for combined figures).
        """
        import os
        import pandas as pd
        
        # Define experiment H2 CSV file
        file_path = CACHE_DIR + 'all_subjects_model_parameters_online.csv'

        # Load dataset
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print(f"Loaded H2: {len(df)} rows")
        
        result = PlotMeasuresMixedEffects._fit_goal_choice_mixed_effects_h2(df)
        
        if result is None:
            print("No valid results for experiment H2 goal choice")
            return None
        
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        PlotMeasuresMixedEffects._draw_forest_panel(
            ax, result, title='Goal Switch Choice', xlabel='Log Odds Ratio', coef_fmt='.2f'
        )

        if fig is not None:
            plt.tight_layout()
            if save:
                plt.savefig(FIGURES_TOPICS + "/forest_plot_goal_choice_experiment_h2_only.png", dpi=300, bbox_inches='tight')
            if show:
                plt.show()
        
        # Print summary statistics
        coefs = result['coefs']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        p_values = result['p_values']
        regressor_corr = result['regressor_corr']
        predictor_labels = result['predictor_labels']

        print(f"\n{'='*80}")
        print(f"EXPERIMENT H2 GOAL CHOICE MIXED EFFECTS MODEL RESULTS")
        print(f"{'='*80}")
        print(f"{'N Subjects':<12} {'N Trials':<12} {'Regressor r':<15}")
        print(f"{result['n_subjects']:<12} {result['n_trials']:<12} {regressor_corr:<15.3f}")
        print(f"{'-'*80}")
        print(f"{'Predictor':<20} {'Log OR':<12} {'95% CI':<20} {'p-value':<12} {'Sig':<6}")
        print(f"{'-'*80}")
        
        for pred, coef, p_val, ci_low, ci_high in zip(predictor_labels, coefs, p_values, ci_lower, ci_upper):
            if p_val < 0.001:
                sig_star = '***'
                p_display = '< 0.001'
            elif p_val < 0.01:
                sig_star = '**'
                p_display = f'{p_val:.3f}'
            elif p_val < 0.05:
                sig_star = '*'
                p_display = f'{p_val:.3f}'
            else:
                sig_star = 'ns'
                p_display = f'{p_val:.3f}'
            
            print(f"{pred:<20} {coef:<12.3f} [{ci_low:.3f}, {ci_high:.3f}] {p_display:<12} {sig_star:<6}")
        
        print(f"{'='*80}")
        
        return result

    # final-final manuscript: Figure 9A.
    # 9A: Figure9A.png
    # See analysis/README.md and reproduce_figures.py for inputs and panel aliases.
    @staticmethod
    def plot_figure9a():
        """
        Combine goal RT and goal choice forest plots as subplots of Figure 9A.
        Left: Goal Switch Choice; Right: Goal Reaction Time.
        Saves to Figure9A.png.
        """
        import os
        import pandas as pd

        file_path = CACHE_DIR + 'all_subjects_model_parameters_online.csv'
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return None

        df = pd.read_csv(file_path)
        print(f"Loaded H2: {len(df)} rows")

        rt_result = PlotMeasuresMixedEffects._fit_rt_mixed_effects_h2(df)
        choice_result = PlotMeasuresMixedEffects._fit_goal_choice_mixed_effects_h2(df)

        if rt_result is None or choice_result is None:
            print("No valid results for Figure 9A")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        PlotMeasuresMixedEffects._draw_forest_panel(
            axes[0], choice_result,
            title='Goal Switch Choice', xlabel='Log Odds Ratio', coef_fmt='.2f'
        )
        PlotMeasuresMixedEffects._draw_forest_panel(
            axes[1], rt_result,
            title='Goal Reaction Time', xlabel='Effect Size (ms)', coef_fmt='.1f',
            show_ylabels=False
        )

        plt.tight_layout()
        plt.savefig(FIGURES_TOPICS + "/Figure9A.png", dpi=300, bbox_inches='tight')
        plt.show()

        return fig


if __name__ == "__main__":
    # For all manuscript panels, use reproduce_figures.py; this is a local example.
    print("\nCreating forest plot for experiment H2 only...")

    print("\nCreating forest plot for goal choice in experiment H2...")

    print("\nCreating combined Figure 9A...")
    PlotMeasuresMixedEffects.plot_figure9a()

    print("\nCreating forest plot for goal RT switch vs stay trials...")
    PlotMeasuresMixedEffects.plot_forest_plot_goal_rt_switch_vs_stay()

    print("\nAll plots completed!")
