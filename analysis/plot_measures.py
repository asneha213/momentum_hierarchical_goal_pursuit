import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "src/")
from src.create_behavior_dataframe import *
from src.create_model_parameter_df import *
from src.datautils import *

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.gridspec as gridspec
import numpy as np
import textwrap

import pandas as pd
import pingouin as pg
from scipy.stats import chi2_contingency


class PlotMeasures:
    def __init__(self, data_type, read_from_csv=True):
        self.data_type = data_type
        if not read_from_csv:
            self.df = get_all_subjects_concatenated_model_dataframe(data_type=self.data_type)
            pass
        else:
            behavior_dataframe_csv = CACHE_DIR + "all_subjects_model_parameters_" + self.data_type + ".csv"
            # if not os.path.exists(behavior_dataframe_csv):
            #     get_all_subjects_concatenated_model_dataframe(data_type=self.data_type)
            self.df = pd.read_csv(behavior_dataframe_csv)

        # Set figure directory based on data_type
        if data_type == "online":
            data = get_subject_data_from_id(experiment=0, subject_id=0, data_type=self.data_type)
            print(data['block_type_data'])
            print(data['block_counts'])
            self.experiment = 'H2'
            self.figure_dir = FIGURES_TOPICS
        elif data_type == "fmri":
            self.experiment = 'H1'
            self.figure_dir = FIGURES_TOPICS


    def plot_goal_selection_per_block(self):
        df = self.df
        # Aesthetic setup - black, white, and gray palette
        palette = ['#000000', '#FFFFFF', '#808080']  # Black, white, gray
        sns.set(style="white", context="talk")

        # Compute selection probability per subject, per block_type
        long_df = df.groupby(['subject_id', 'block_type', 'goal_selected']).size().reset_index(name='count')

        # Normalize within subject/block_type
        long_df['total'] = long_df.groupby(['subject_id', 'block_type'])['count'].transform('sum')
        long_df['prob'] = long_df['count'] / long_df['total']
        long_df['goal_selected'] = long_df['goal_selected'].replace('BR', 'OB')

        # Plot
        plt.figure(figsize=(9, 6))
        ax = sns.barplot(
            data=long_df,
            x='block_type',
            y='prob',
            hue='goal_selected',
            errorbar="se",  # use this instead of ci
            palette=palette,
            edgecolor='black',
            width=0.7,
        )

        # Aesthetic enhancements
        ax.set_xlabel("Block Type", fontsize=15, weight='bold')
        ax.set_ylabel("Probability of Goal Selection", fontsize=15, weight='bold')
        ax.set_xticklabels([
            'SH-high', 
            'OB-high', 
            'HO-high', 
            'SH-low', 
            'OB-low', 
            'HO-low'
        ])

        experiment = "H2" if self.data_type == "online" else "H1"

        ax.set_ylim(0, 1.0)
        ax.set_title("Goal Selection Probability by Block Type: Experiment " + experiment, fontsize=16, weight='bold')
        plt.legend(title="Goal selected")
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figure_dir + "/goal_selection_per_block_" + self.experiment + ".png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_viable_goal_selection_high_low(self):
        """
        Plot overall percentage of viable goal and subgoal selection in high vs low conditions.
        Collapses over individual goals (SH, OB, HO) within each condition type.
        """
        df = self.df.copy()
        
        # Replace BR with OB for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Define viable goals for each block_type
        # 0: SH, 1:OB, 2:HO, 3:SH, 4:OB, 5:HO
        # 0,1,2 are high; 3,4,5 are low
        viable_goals = {
            0: 'SH',  # SH-high
            1: 'OB',  # OB-high  
            2: 'HO',  # HO-high
            3: 'SH',  # SH-low
            4: 'OB',  # OB-low
            5: 'HO'   # HO-low
        }
        
        # Define viable subgoals for each goal
        viable_subgoals = {
            'SH': ['G', 'M'],
            'HO': ['M', 'C', 'S'],
            'OB': ['G', 'C', 'S']
        }
        
        # Create high/low condition mapping
        df['condition_type'] = df['block_type'].map({0: 'high', 1: 'high', 2: 'high', 
                                                    3: 'low', 4: 'low', 5: 'low'})
        
        # Determine if selected goal is viable for each trial
        df['is_viable_goal'] = df.apply(lambda row: row['goal_selected'] == viable_goals[row['block_type']], axis=1)
        
        # Determine if selected subgoal is viable for each trial
        def is_viable_subgoal(row):
            if pd.isna(row['subgoal_selected']) or row['subgoal_selected'] is None:
                return False
            
            # Check if subgoal is viable for the selected goal
            if row['goal_selected'] not in viable_subgoals:
                return False
            
            if row['subgoal_selected'] not in viable_subgoals[row['goal_selected']]:
                return False
            
            # Check if subgoal progress is less than 1
            progress_col = f"{row['subgoal_selected']}_progress_{row['goal_selected']}"
            if progress_col in row and not pd.isna(row[progress_col]):
                return row[progress_col] < 1
            else:
                return False
        
        df['is_viable_subgoal'] = df.apply(is_viable_subgoal, axis=1)
        
        # Calculate viable goal selection percentage per subject per condition type
        viable_goal_selection = df.groupby(['subject_id', 'condition_type'])['is_viable_goal'].mean().reset_index()
        viable_goal_selection.columns = ['subject_id', 'condition_type', 'viable_percentage']
        
        # Calculate viable subgoal selection percentage per subject per condition type
        viable_subgoal_selection = df.groupby(['subject_id', 'condition_type'])['is_viable_subgoal'].mean().reset_index()
        viable_subgoal_selection.columns = ['subject_id', 'condition_type', 'viable_percentage']
        
        # Aesthetic setup - grayscale color variations
        goal_palette = ['#D3D3D3', '#696969']  # Light gray for high, dim gray for low
        subgoal_palette = ['#A9A9A9', '#505050']  # Dark gray for high, very dark gray for low
        sns.set(style="white", context="talk")
        
        # Create the plot with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Viable Goal Selection
        sns.barplot(
            data=viable_goal_selection,
            x='condition_type',
            y='viable_percentage',
            errorbar="se",
            palette=goal_palette,
            edgecolor='black',
            width=0.3,
            ax=ax1
        )
        
        # Plot 2: Viable Subgoal Selection
        sns.barplot(
            data=viable_subgoal_selection,
            x='condition_type',
            y='viable_percentage',
            errorbar="se",
            palette=subgoal_palette,
            edgecolor='black',
            width=0.3,
            ax=ax2
        )
        
        # Format both subplots
        for ax, title, ylabel in [(ax1, "Dominant Goal Selection", "Dominant Goal Selection Percentage"), 
                                 (ax2, "Dominant Subgoal Selection", "Dominant Subgoal Selection Percentage")]:
            
            # Aesthetic enhancements
            ax.set_xlabel("Condition Type", fontsize=16, weight='bold')
            ax.set_ylabel(ylabel, fontsize=16, weight='bold')
            ax.set_xticklabels(['High', 'Low'], fontsize=14, weight='bold')
            
            # Set y-axis to percentage format
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            # Center the bars by adjusting x-axis limits
            ax.set_xlim(-0.5, 1.5)
            
            # Add title
            experiment = "H2" if self.data_type == "online" else "H1"
            ax.set_title(f"{title}: High vs Low Conditions", 
                        fontsize=16, weight='bold')
        
        # Statistical testing for goals
        from scipy import stats
        high_goal_data = viable_goal_selection[viable_goal_selection['condition_type'] == 'high']['viable_percentage']
        low_goal_data = viable_goal_selection[viable_goal_selection['condition_type'] == 'low']['viable_percentage']
        
        # Paired t-test for goals
        try:
            t_stat_goal, p_value_goal = stats.ttest_rel(high_goal_data, low_goal_data)
        except:
            t_stat_goal, p_value_goal = stats.ttest_ind(high_goal_data, low_goal_data)

        df_goal = len(high_goal_data) - 1
        print(f"Goal selection: t-statistic: {t_stat_goal:.3f}, p-value: {p_value_goal:.3f}, df: {df_goal}")
        
        # Statistical testing for subgoals
        high_subgoal_data = viable_subgoal_selection[viable_subgoal_selection['condition_type'] == 'high']['viable_percentage']
        low_subgoal_data = viable_subgoal_selection[viable_subgoal_selection['condition_type'] == 'low']['viable_percentage']
        
        # Paired t-test for subgoals
        try:
            t_stat_subgoal, p_value_subgoal = stats.ttest_rel(high_subgoal_data, low_subgoal_data)
        except:
            t_stat_subgoal, p_value_subgoal = stats.ttest_ind(high_subgoal_data, low_subgoal_data)

        print(f"Subgoal selection: t-statistic: {t_stat_subgoal:.3f}, p-value: {p_value_subgoal:.3f}")
        
        # Add significance markers for both plots
        for ax, high_data, low_data, p_value in [(ax1, high_goal_data, low_goal_data, p_value_goal),
                                                 (ax2, high_subgoal_data, low_subgoal_data, p_value_subgoal)]:
            
            y_max = max(high_data.mean(), low_data.mean())
            y_pos = y_max + 0.05
            
            if p_value < 0.001:
                marker = '***'
            elif p_value < 0.01:
                marker = '**'
            elif p_value < 0.05:
                marker = '*'
            else:
                marker = 'ns'
            
            # Add significance line and marker
            ax.plot([0, 1], [y_pos, y_pos], 'k-', linewidth=1)
            ax.text(0.5, y_pos + 0.02, marker, ha='center', va='bottom', fontsize=20, fontweight='bold', color='blue')
            
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figure_dir + f"/viable_goal_subgoal_selection_high_low_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistical results
        print(f"\nStatistical Results - Goals:")
        print(f"High condition: {high_goal_data.mean():.3f} ± {high_goal_data.std():.3f}")
        print(f"Low condition: {low_goal_data.mean():.3f} ± {low_goal_data.std():.3f}")
        print(f"t-statistic: {t_stat_goal:.3f}, p-value: {p_value_goal:.3f}")
        
        print(f"\nStatistical Results - Subgoals:")
        print(f"High condition: {high_subgoal_data.mean():.3f} ± {high_subgoal_data.std():.3f}")
        print(f"Low condition: {low_subgoal_data.mean():.3f} ± {low_subgoal_data.std():.3f}")
        print(f"t-statistic: {t_stat_subgoal:.3f}, p-value: {p_value_subgoal:.3f}")


    def plot_subgoal_selection_per_block(self):
        df = self.df
        # Aesthetic setup
        sns.set(style="white", context="talk")

        # Compute selection probability per subject, per block_type
        long_df = df.groupby(['subject_id', 'block_type', 'subgoal_selected']).size().reset_index(name='count')

        # Normalize within subject/block_type
        long_df['total'] = long_df.groupby(['subject_id', 'block_type'])['count'].transform('sum')
        long_df['prob'] = long_df['count'] / long_df['total']

        # Plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=long_df,
            x='block_type',
            y='prob',
            hue='subgoal_selected',
            errorbar="se",  # use this instead of ci
            palette=['#FFFFFF', '#000000', '#D3D3D3', '#696969'],  # White, black, light gray, dark gray
            edgecolor='black'
        )

        # Aesthetic enhancements
        ax.set_xlabel("Condition", fontsize=15, weight='bold')
        ax.set_xticklabels([
            'SH-high', 
            'OB-high', 
            'HO-high', 
            'SH-low', 
            'OB-low', 
            'HO-low'
        ])
        ax.set_ylabel("P(Subgoal Selected)", fontsize=15, weight='bold')
        ax.set_ylim(0, 0.8)
        experiment = "H2" if self.data_type == "online" else "H1"
        ax.set_title("Subgoal Selection Probability by Block Type: Experiment " + experiment, fontsize=16, weight='bold')
        plt.legend(title="Subgoal selected")
        sns.despine()
        plt.tight_layout()

        plt.savefig(self.figure_dir + "/subgoal_selection_per_block_" + self.experiment + ".png", dpi=300, bbox_inches='tight')
        plt.show()



    def plot_goal_switching_transitions_per_block(self):

        df = self.df
        # Sort by subject, block, and trial to maintain temporal order
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])

        # Shift goal selections to get transitions
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df = df.dropna(subset=['prev_goal'])

        # Compute transition probabilities by condition
        transition_matrices = {}
        conditions = df['block_type'].unique()

        for cond in conditions:
            cond_df = df[df['block_type'] == cond]
            transition_counts = pd.crosstab(cond_df['prev_goal'], cond_df['goal_selected'], normalize='index')
            transition_matrices[cond] = transition_counts
            
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()

        for i, cond in enumerate(sorted(conditions)):
            sns.heatmap(
                transition_matrices[cond],
                annot=True,
                fmt=".2f",
                cmap="crest",
                cbar=False,
                linewidths=0.5,
                linecolor='gray',
                square=True,
                ax=axs[i],
                annot_kws={"size": 12}
            )
            axs[i].set_title(f'Transitions - {cond}', fontsize=14, weight='bold')
            axs[i].set_xlabel('To Goal', fontsize=12)
            axs[i].set_ylabel('From Goal', fontsize=12)
            axs[i].tick_params(labelsize=10)

        plt.suptitle('Goal Transition Matrices by Condition', fontsize=18, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(FIGURES + "/goal_switching_transitions_per_block_" + self.experiment + ".png", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_subgoal_switching_transitions_per_block(self):
        df = self.df
        # Sort by subject, block, and trial to maintain temporal order
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])


        # Compute transition probabilities by condition
        transition_matrices = {}
        conditions = df['block_type'].unique()
        df['prev_subgoal'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        df_subgoal = df.dropna(subset=['prev_subgoal'])

        # Compute subgoal transition probabilities by condition
        subgoal_transition_matrices = {}
        for cond in conditions:
            cond_df = df_subgoal[df_subgoal['block_type'] == cond]
            transition_counts = pd.crosstab(cond_df['prev_subgoal'], cond_df['subgoal_selected'], normalize='index')
            subgoal_transition_matrices[cond] = transition_counts

        # Plotting subgoal transitions
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()

        vmin = min(matrix.values.min() for matrix in subgoal_transition_matrices.values())
        vmax = max(matrix.values.max() for matrix in subgoal_transition_matrices.values())


        for i, cond in enumerate(sorted(conditions)):
            sns.heatmap(
                subgoal_transition_matrices[cond],
                annot=False,
                fmt=".2f",
                cmap="gray",
                cbar=True,
                linewidths=0.5,
                linecolor='gray',
                square=True,
                ax=axs[i],
                annot_kws={"size": 12},
                vmin=vmin,
                vmax=vmax
            )
            axs[i].set_title(f'Subgoal Transitions - {cond}', fontsize=14, weight='bold')
            axs[i].set_xlabel('To Subgoal', fontsize=12)
            axs[i].set_ylabel('From Subgoal', fontsize=12)
            axs[i].tick_params(labelsize=10)

        plt.suptitle('Subgoal Transition Matrices by Condition', fontsize=18, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.figure_dir + "subgoal_switching_transitions_per_block_" + self.experiment + ".png", dpi=300, bbox_inches='tight')
        plt.show()



    def plot_goal_selection_related_goal_progress(self, simulate=False, model_name="momentum_learn_alt_goal", collapse_over_goals=False):
        """
        Plot the goal selection related goal progress.
        
        Args:
            simulate (bool): Whether to use simulated data
            model_name (str): Name of the model to use for simulation (e.g., 'momentum_learn_alt_goal', 'prospective', 'td_persistence')
            collapse_over_goals (bool): If True, creates a single plot collapsing over goals and conditions.
                                      If False, creates separate subplots for each goal.
        """
        if simulate:
            behavior_dataframe_csv = f"all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
            self.df = pd.read_csv(behavior_dataframe_csv)

        goals = ['SH', 'BR', 'HO']
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high",
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }


        # Apply the replacement
        self.df['block_type'] = self.df['block_type'].replace(block_labels)

        melted = pd.melt(
            self.df,
            id_vars=['goal_selected', 'block_type'],
            value_vars=[f'{g}_progress' for g in goals],
            var_name='goal_progressed',
            value_name='progress'
        )
        # Clean goal_progressed to just "SH", "BR", "HO"
        melted['goal'] = melted['goal_progressed'].str.extract(r'(\w+)_progress')\


        # Step 4: Compute counts per (goal, progress_bin, block_type)
        counts = melted.groupby(['goal', 'progress', 'block_type']).size().reset_index(name='total')
        selected = melted[melted['goal'] == melted['goal_selected']]
        selected_counts = selected.groupby(['goal', 'progress', 'block_type']).size().reset_index(name='selected')
        # Step 6: Merge and compute probability
        merged = pd.merge(counts, selected_counts, on=['goal', 'progress', 'block_type'], how='left')
        merged['selected'] = merged['selected'].fillna(0)
        merged['prob'] = merged['selected'] / merged['total']
        merged = merged[merged['progress'].isin([0, 1, 2, 3, 4, 5])]


        sns.set_theme(style="white", font_scale=1.2)

        # Filter and order progress bins
        merged['progress'] = pd.Categorical(merged['progress'], categories=[0, 1, 2, 3, 4, 5], ordered=True)

        # Define custom color palette
        palette = sns.color_palette("Dark2")  # or "Set2", "Dark2", etc.

        if collapse_over_goals:
            # Create a single plot collapsing over both goals and block types
            # Aggregate data across goals and block types
            collapsed_data = merged.groupby('progress').agg({
                'prob': ['mean', 'std', 'count']
            }).reset_index()
            
            # Flatten column names
            collapsed_data.columns = ['progress', 'prob_mean', 'prob_std', 'prob_count']
            
            # Calculate standard error
            collapsed_data['prob_se'] = collapsed_data['prob_std'] / np.sqrt(collapsed_data['prob_count'])
            
            plt.figure(figsize=(6, 6))
            plt.errorbar(
                collapsed_data['progress'],
                collapsed_data['prob_mean'],
                yerr=collapsed_data['prob_se'],
                marker='o',
                linewidth=2,
                capsize=5,
                capthick=2,
                color='black',
                markerfacecolor='black',
                markeredgecolor='black'
            )
            plt.xlabel("Goal Progress", fontsize=14, weight='bold')
            plt.ylabel("Goal Selection Probability", fontsize=14, weight='bold')
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            
            # Perform repeated measures ANOVA
            print("\n" + "="*80)
            print("REPEATED MEASURES ANOVA: Progress Effect on Goal Selection")
            print("="*80)
            
            # Prepare data for repeated measures ANOVA
            # We need to restructure the data to have subject-level observations
            # First, let's get the original data with subject information
            if simulate:
                # For simulated data, we need to reconstruct subject-level data
                # Get the original dataframe with subject_id
                if self.data_type == "online":
                    behavior_dataframe_csv = f"all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
                else:
                    behavior_dataframe_csv = f"all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
                df_original = pd.read_csv(behavior_dataframe_csv)
            else:
                df_original = self.df
            
            # Apply block type replacement
            df_original['block_type'] = df_original['block_type'].replace(block_labels)
            
            # Create melted dataframe with subject information
            melted_with_subjects = pd.melt(
                df_original,
                id_vars=['subject_id', 'goal_selected', 'block_type'],
                value_vars=[f'{g}_progress' for g in goals],
                var_name='goal_progressed',
                value_name='progress'
            )
            melted_with_subjects['goal'] = melted_with_subjects['goal_progressed'].str.extract(r'(\w+)_progress')
            
            # Filter to valid progress values
            melted_with_subjects = melted_with_subjects[melted_with_subjects['progress'].isin([0, 1, 2, 3, 4, 5])]
            
            # Create binary selection variable (1 if goal was selected, 0 otherwise)
            melted_with_subjects['goal_selected_binary'] = (melted_with_subjects['goal'] == melted_with_subjects['goal_selected']).astype(int)
            
            # Calculate subject-level selection probabilities for each progress level
            subject_progress_data = []
            for subject_id in melted_with_subjects['subject_id'].unique():
                subject_data = melted_with_subjects[melted_with_subjects['subject_id'] == subject_id]
                
                for progress_level in [0, 1, 2, 3, 4, 5]:
                    progress_trials = subject_data[subject_data['progress'] == progress_level]
                    if len(progress_trials) > 0:
                        selection_prob = progress_trials['goal_selected_binary'].mean()
                        subject_progress_data.append({
                            'subject_id': subject_id,
                            'progress': progress_level,
                            'selection_prob': selection_prob
                        })
            
            # Convert to DataFrame for ANOVA
            anova_df = pd.DataFrame(subject_progress_data)
            
            if len(anova_df) > 0:
                # Perform repeated measures ANOVA using pingouin
                try:
                    # Use pingouin's rm_anova function
                    rm_anova_result = pg.rm_anova(data=anova_df, dv='selection_prob', within='progress', subject='subject_id')
                    
                    print(f"Repeated Measures ANOVA Results:")
                    print(f"F({rm_anova_result['ddof1'].iloc[0]:.0f}, {rm_anova_result['ddof2'].iloc[0]:.0f}) = {rm_anova_result['F'].iloc[0]:.3f}")
                    print(f"p-value = {rm_anova_result['p-unc'].iloc[0]:.6f}")
                    print(f"Effect size (η²p) = {rm_anova_result['np2'].iloc[0]:.3f}")
                    
                    # Interpret significance
                    if rm_anova_result['p-unc'].iloc[0] < 0.001:
                        significance = "***"
                    elif rm_anova_result['p-unc'].iloc[0] < 0.01:
                        significance = "**"
                    elif rm_anova_result['p-unc'].iloc[0] < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"
                    
                    print(f"Significance: {significance}")
                    
                    # Post-hoc pairwise comparisons if significant
                    if rm_anova_result['p-unc'].iloc[0] < 0.05:
                        print(f"\nPost-hoc pairwise comparisons (Bonferroni corrected):")
                        pairwise_results = pg.pairwise_ttests(data=anova_df, dv='selection_prob', within='progress', subject='subject_id', padjust='bonf')
                        
                        # Show only significant comparisons
                        significant_pairs = pairwise_results[pairwise_results['p-corr'] < 0.05]
                        if len(significant_pairs) > 0:
                            for _, row in significant_pairs.iterrows():
                                print(f"  Progress {row['A']} vs {row['B']}: t = {row['T']:.3f}, p = {row['p-corr']:.3f}")
                        else:
                            print("  No significant pairwise differences found.")
                    
                    # Descriptive statistics
                    print(f"\nDescriptive Statistics:")
                    desc_stats = anova_df.groupby('progress')['selection_prob'].agg(['mean', 'std', 'count']).round(3)
                    for progress_level in [0, 1, 2, 3, 4, 5]:
                        if progress_level in desc_stats.index:
                            mean_val = desc_stats.loc[progress_level, 'mean']
                            std_val = desc_stats.loc[progress_level, 'std']
                            count_val = desc_stats.loc[progress_level, 'count']
                            print(f"  Progress {progress_level}: M = {mean_val:.3f}, SD = {std_val:.3f}, n = {count_val:.0f}")
                    
                except Exception as e:
                    print(f"Error performing repeated measures ANOVA: {e}")
                    print("This might be due to insufficient data or missing values.")
            else:
                print("Insufficient data for repeated measures ANOVA.")
            
            print("="*80)
        else:
            # Original behavior with separate subplots for each goal
            # Create FacetGrid first (it will create its own figure)
            g = sns.FacetGrid(
                merged,
                col='goal',
                sharey=True,
                height=4,
                aspect=0.7
            )
            
            # Adjust figure size to make room for legend
            fig = g.fig
            fig.set_size_inches(14, 4)
            
            # Reposition existing axes to make room for legend and reduce spacing
            # Calculate positions for all subplots with minimal spacing
            n_subplots = len(g.axes.flat)
            total_width = 0.7  # Use 70% of figure width for all subplots
            spacing = 0.02  # Small gap between subplots
            subplot_width = (total_width - (n_subplots - 1) * spacing) / n_subplots
            start_x = 0.1  # Start position
            
            for i, ax in enumerate(g.axes.flat):
                current_pos = ax.get_position()
                new_x0 = start_x + i * (subplot_width + spacing)
                ax.set_position([new_x0, current_pos.y0, subplot_width, current_pos.height])

            # Use constant color with different markers for each condition
            # Define markers for different block types
            markers = ['o', 's', '^', 'D', 'v', 'p']  # circle, square, triangle, diamond, triangle_down, pentagon
            block_types = sorted(merged['block_type'].unique())
            marker_dict = {bt: markers[i % len(markers)] for i, bt in enumerate(block_types)}
            
            # Plot each block type separately with same color but different markers on each subplot
            for i, ax in enumerate(g.axes.flat):
                goal_name = g.col_names[i]
                goal_data = merged[merged['goal'] == goal_name]
                
                for block_type in block_types:
                    block_data = goal_data[goal_data['block_type'] == block_type]
                    if len(block_data) > 0:
                        # Sort by progress for proper line plotting
                        block_data = block_data.sort_values('progress')
                        ax.plot(
                            block_data['progress'],
                            block_data['prob'],
                            marker=marker_dict[block_type],
                            linestyle=':',
                            linewidth=2,
                            color='black',
                            markersize=8,
                            label=block_type
                        )

            # Clean titles and axis labels
            # Custom title mapping with BR replaced by OB
            custom_titles = {
                'SH': 'Goal: SH',
                'BR': 'Goal: OB',  # Replace 'BR' with 'OB'
                'HO': 'Goal: HO'
            }

            # Apply the custom titles
            g.set_axis_labels(None, "Goal Selection Probability", size=14, weight='bold')
            g.set(ylim=(0, 1.05))

            for i, ax in enumerate(g.axes.flat):
                ax.set_title(
                    custom_titles.get(g.col_names[i], g.col_names[i]),
                    fontsize=14,
                    weight='bold',
                    pad=0  # move title further down (smaller pad moves it closer to the plot)
                )
                if i == 0:
                    ax.set_xlabel("Goal Progress", fontsize=14, weight='bold')
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])  # Optional: also hide the tick labels

            # Add legend in a separate subplot
            # Create custom legend with markers
            legend_handles = []
            for block_type in block_types:
                if block_type in marker_dict:
                    legend_handles.append(plt.Line2D([0], [0], 
                                                      marker=marker_dict[block_type], 
                                                      linestyle=':', 
                                                      color='black', 
                                                      linewidth=2, 
                                                      markersize=8, 
                                                      label=block_type))
            
            # Create legend subplot in the rightmost position
            # Get position from the rightmost plot and place legend to its right
            rightmost_ax = g.axes.flat[-1]
            rightmost_pos = rightmost_ax.get_position()
            legend_x0 = rightmost_pos.x0 + rightmost_pos.width + 0.02
            legend_ax = fig.add_axes([legend_x0, rightmost_pos.y0, 0.15, rightmost_pos.height])
            legend_ax.axis('off')
            legend_ax.legend(handles=legend_handles, loc='center left', title='Block Type', frameon=False)
        if self.data_type == "online":
            experiment = "H2"
        elif self.data_type == "fmri":
            experiment = "H1"
        if collapse_over_goals:
            plt.suptitle('Goal selection with progress', fontsize=15, weight='bold', y=0.95)
        if not simulate:
            if collapse_over_goals:
                plt.savefig(self.figure_dir + "/goal_selection_related_goal_progress_" + self.experiment + "_collapsed.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.figure_dir + "/goal_selection_related_goal_progress_" + self.experiment + ".png", dpi=300, bbox_inches='tight')
        else:
            if collapse_over_goals:
                plt.savefig(self.figure_dir + f"/goal_selection_related_goal_progress_{self.experiment}_{model_name}_simulated_collapsed.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.figure_dir + f"/goal_selection_related_goal_progress_{self.experiment}_{model_name}_simulated.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_goal_selection_related_goal_progress_simulated(self, models=['momentum_learn_alt_goal', 'prospective', 'td_persistence'], 
                                                           collapse_over_goals=True, collapse_over_conditions=True, include_behavior=True):
        """
        Plot goal selection related goal progress for different simulated models.
        
        Args:
            models (list): List of model names to simulate
            collapse_over_goals (bool): If True, collapses over goals
            collapse_over_conditions (bool): If True, collapses over conditions
            include_behavior (bool): If True, includes actual behavior data as 'Behavior' model
        """
        goals = ['SH', 'BR', 'HO']
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high",
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }

        # Model display names and colors
        model_display_names = {
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal',
            'momentum': 'TD-momentum',
            'prospective': 'Prospective',
            'td_persistence': 'TD-persistence',
            'behavior': 'Behavior'
        }
        
        model_colors = {
            'momentum_learn_alt_goal': '#1f77b4',  # Dark blue
            'momentum': '#1f77b4',  # Dark blue
            'prospective': '#ff7f0e',  # Dark orange
            'td_persistence': '#2ca02c',  # Dark green
            'behavior': '#d62728'  # Dark red
        }

        all_results = []
        
        for model_name in models:
            # Load simulated data for this model
            behavior_dataframe_csv = CACHE_DIR + f"all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
            df = pd.read_csv(behavior_dataframe_csv)
            
            # Apply the replacement
            df['block_type'] = df['block_type'].replace(block_labels)

            melted = pd.melt(
                df,
                id_vars=['goal_selected', 'block_type'],
                value_vars=[f'{g}_progress' for g in goals],
                var_name='goal_progressed',
                value_name='progress'
            )
            # Clean goal_progressed to just "SH", "BR", "HO"
            melted['goal'] = melted['goal_progressed'].str.extract(r'(\w+)_progress')

            # Step 4: Compute counts per (goal, progress_bin, block_type)
            counts = melted.groupby(['goal', 'progress', 'block_type']).size().reset_index(name='total')
            selected = melted[melted['goal'] == melted['goal_selected']]
            selected_counts = selected.groupby(['goal', 'progress', 'block_type']).size().reset_index(name='selected')
            # Step 6: Merge and compute probability
            merged = pd.merge(counts, selected_counts, on=['goal', 'progress', 'block_type'], how='left')
            merged['selected'] = merged['selected'].fillna(0)
            merged['prob'] = merged['selected'] / merged['total']
            merged = merged[merged['progress'].isin([0, 1, 2, 3, 4, 5])]
            
            # Add model information
            merged['model'] = model_name
            merged['model_display'] = model_display_names[model_name]
            
            all_results.append(merged)

        # Add behavior data if requested
        if include_behavior:
            # Use the original dataframe (self.df) for behavior data
            df_behavior = self.df.copy()
            
            # Apply the replacement
            df_behavior['block_type'] = df_behavior['block_type'].replace(block_labels)

            melted_behavior = pd.melt(
                df_behavior,
                id_vars=['goal_selected', 'block_type'],
                value_vars=[f'{g}_progress' for g in goals],
                var_name='goal_progressed',
                value_name='progress'
            )
            # Clean goal_progressed to just "SH", "BR", "HO"
            melted_behavior['goal'] = melted_behavior['goal_progressed'].str.extract(r'(\w+)_progress')

            # Step 4: Compute counts per (goal, progress_bin, block_type)
            counts_behavior = melted_behavior.groupby(['goal', 'progress', 'block_type']).size().reset_index(name='total')
            selected_behavior = melted_behavior[melted_behavior['goal'] == melted_behavior['goal_selected']]
            selected_counts_behavior = selected_behavior.groupby(['goal', 'progress', 'block_type']).size().reset_index(name='selected')
            # Step 6: Merge and compute probability
            merged_behavior = pd.merge(counts_behavior, selected_counts_behavior, on=['goal', 'progress', 'block_type'], how='left')
            merged_behavior['selected'] = merged_behavior['selected'].fillna(0)
            merged_behavior['prob'] = merged_behavior['selected'] / merged_behavior['total']
            merged_behavior = merged_behavior[merged_behavior['progress'].isin([0, 1, 2, 3, 4, 5])]
            
            # Add model information
            merged_behavior['model'] = 'behavior'
            merged_behavior['model_display'] = 'Behavior'
            
            all_results.append(merged_behavior)

        # Combine all results
        final_data = pd.concat(all_results, ignore_index=True)
        
        sns.set_theme(style="white", font_scale=1.2)
        
        # Filter and order progress bins
        final_data['progress'] = pd.Categorical(final_data['progress'], categories=[0, 1, 2, 3, 4, 5], ordered=True)

        if collapse_over_goals and collapse_over_conditions:
            # Create a single plot collapsing over both goals and conditions
            # Aggregate data across goals, conditions, and models
            collapsed_data = final_data.groupby(['progress', 'model']).agg({
                'prob': ['mean', 'std', 'count']
            }).reset_index()
            
            # Flatten column names
            collapsed_data.columns = ['progress', 'model', 'prob_mean', 'prob_std', 'prob_count']
            
            # Calculate standard error
            collapsed_data['prob_se'] = collapsed_data['prob_std'] / np.sqrt(collapsed_data['prob_count'])
            
            plt.figure(figsize=(10, 6))
            
            # Plot all models including behavior if requested
            all_models = models.copy()
            if include_behavior:
                all_models.append('behavior')
                
            for model_name in all_models:
                model_data = collapsed_data[collapsed_data['model'] == model_name]
                plt.errorbar(
                    model_data['progress'],
                    model_data['prob_mean'],
                    yerr=model_data['prob_se'],
                    marker='o',
                    linewidth=2,
                    capsize=5,
                    capthick=2,
                    label=model_display_names[model_name],
                    color=model_colors[model_name]
                )
            
            plt.xlabel("Goal Progress", fontsize=14, weight='bold')
            plt.ylabel("Goal Selection Probability", fontsize=14, weight='bold')
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Model', fontsize=12)
            plt.title('Goal Selection Probability by Progress (Simulated Models)', fontsize=16, weight='bold')
            
        elif collapse_over_goals and not collapse_over_conditions:
            # Collapse over goals but show different conditions
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            conditions = ['SH-high', 'OB-high', 'HO-high', 'SH-low', 'OB-low', 'HO-low']
            
            for i, condition in enumerate(conditions):
                ax = axes[i]
                condition_data = final_data[final_data['block_type'] == condition]
                
                # Aggregate across goals for this condition
                collapsed_condition = condition_data.groupby(['progress', 'model']).agg({
                    'prob': ['mean', 'std', 'count']
                }).reset_index()
                
                collapsed_condition.columns = ['progress', 'model', 'prob_mean', 'prob_std', 'prob_count']
                collapsed_condition['prob_se'] = collapsed_condition['prob_std'] / np.sqrt(collapsed_condition['prob_count'])
                
                # Plot all models including behavior if requested
                all_models = models.copy()
                if include_behavior:
                    all_models.append('behavior')
                    
                for model_name in all_models:
                    model_data = collapsed_condition[collapsed_condition['model'] == model_name]
                    ax.errorbar(
                        model_data['progress'],
                        model_data['prob_mean'],
                        yerr=model_data['prob_se'],
                        marker='o',
                        linewidth=2,
                        capsize=3,
                        capthick=1,
                        label=model_display_names[model_name],
                        color=model_colors[model_name]
                    )
                
                ax.set_title(f'{condition}', fontsize=12, weight='bold')
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=10)
                if i >= 3:
                    ax.set_xlabel("Goal Progress", fontsize=12, weight='bold')
                if i % 3 == 0:
                    ax.set_ylabel("Goal Selection Probability", fontsize=12, weight='bold')
            
            plt.suptitle('Goal Selection Probability by Progress and Condition (Simulated Models)', fontsize=16, weight='bold')
            plt.tight_layout()
            
        else:
            # Show separate subplots for each goal
            g = sns.FacetGrid(
                final_data,
                col='goal',
                sharey=True,
                height=4,
                aspect=0.7
            )

            g.map_dataframe(
                sns.lineplot,
                x='progress',
                y='prob',
                hue='model',
                style='block_type',
                marker='o',
                linewidth=2,
                palette=model_colors
            )

            # Clean titles and axis labels
            custom_titles = {
                'SH': 'Goal: SH',
                'BR': 'Goal: OB',
                'HO': 'Goal: HO'
            }

            # Apply the custom titles
            g.set_axis_labels(None, "Goal Selection Probability", size=14, weight='bold')
            g.set(ylim=(0, 1.05))

            for i, ax in enumerate(g.axes.flat):
                ax.set_title(
                    custom_titles.get(g.col_names[i], g.col_names[i]),
                    fontsize=14,
                    weight='bold',
                    pad=0
                )
                if i == 0:
                    ax.set_xlabel("Goal Progress", fontsize=14, weight='bold')
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

            # Add legend
            g.add_legend(title='Model')

        # Save the plot
        if self.data_type == "online":
            experiment = "H2"
        elif self.data_type == "fmri":
            experiment = "H1"
            
        # Determine filename suffix based on whether behavior is included
        behavior_suffix = "_with_behavior" if include_behavior else ""
        
        if collapse_over_goals and collapse_over_conditions:
            plt.savefig(self.figure_dir + f"/goal_selection_related_goal_progress_simulated_{self.experiment}_collapsed_all{behavior_suffix}.png", dpi=300, bbox_inches='tight')
        elif collapse_over_goals and not collapse_over_conditions:
            plt.savefig(self.figure_dir + f"/goal_selection_related_goal_progress_simulated_{self.experiment}_collapsed_goals{behavior_suffix}.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.figure_dir + f"/goal_selection_related_goal_progress_simulated_{self.experiment}{behavior_suffix}.png", dpi=300, bbox_inches='tight')
        
        plt.show()


    def plot_subgoal_selection_related_subgoal_progress(self, simulate=False, collapse_conditions=False, collapse_over_subgoals=False):
        """
        Plot subgoal selection probability as a function of subgoal progress.
        Plots all possible subgoal progress combinations (G, M, C, S for each goal SH, BR, HO)
        and shows probability of choosing each subgoal over others.
        Excludes progress = 1 from the analysis.
        
        Args:
            collapse_conditions (bool): If True, collapse across all block types.
                                      If False, show separate lines for each block type.
            collapse_over_subgoals (bool): If True, collapse across all subgoals into a single plot.
                                         If False, show separate subplots for each subgoal.
        """
        if simulate:
            behavior_dataframe_csv = CACHE_DIR + "all_subjects_behavior_data_" + self.data_type + "_momentum_learn_alt_goal_simulated.csv"
            self.df = pd.read_csv(behavior_dataframe_csv)

        # Define all possible subgoal progress variables to plot
        progress_configs = [
            # SH goal subgoals
            ('G_progress_SH', 'G', 'SH'),
            ('M_progress_SH', 'M', 'SH'),
            ('C_progress_SH', 'C', 'SH'),
            ('S_progress_SH', 'S', 'SH'),
            
            # BR goal subgoals  
            ('G_progress_BR', 'G', 'BR'),
            ('S_progress_BR', 'S', 'BR'),
            ('C_progress_BR', 'C', 'BR'),
            ('M_progress_BR', 'M', 'BR'),
            
            # HO goal subgoals
            ('G_progress_HO', 'G', 'HO'),
            ('S_progress_HO', 'S', 'HO'),
            ('C_progress_HO', 'C', 'HO'),
            ('M_progress_HO', 'M', 'HO')
        ]
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }

        # Apply the replacement
        df = self.df.copy()
        df['block_type'] = df['block_type'].replace(block_labels)
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')

        # Process each progress variable separately
        all_results = []
        
        for progress_var, subgoal_type, goal_type in progress_configs:
            # Convert goal_type for consistency
            goal_display = 'OB' if goal_type == 'BR' else goal_type
            
            # Filter data where this progress variable exists and goal matches
            mask = (df[progress_var].notna()) & (df['goal_selected'] == goal_display)
            subset = df[mask].copy()
            
            # Remove progress = 1 and 0.5 as requested
            subset = subset[subset[progress_var] != 1.0]
            subset = subset[subset[progress_var] != 0.5]
            
            # Round progress values to 2 decimal places for exact matching
            subset[progress_var] = subset[progress_var].round(2)
            
            # Keep all progress values (remove the restriction)
            # subset = subset[subset[progress_var].isin([0.0, 0.33, 0.67])]
            
            if subset.empty:
                continue
            
            if collapse_conditions:
                # Compute counts per progress only (collapsed across block types)
                counts = subset.groupby([progress_var]).size().reset_index(name='total')
                
                # Compute selected counts (when subgoal_selected matches the subgoal_type)
                selected_subset = subset[subset['subgoal_selected'] == subgoal_type]
                selected_counts = selected_subset.groupby([progress_var]).size().reset_index(name='selected')
                
                # Merge and compute probability
                merged = pd.merge(counts, selected_counts, on=[progress_var], how='left')
                merged['selected'] = merged['selected'].fillna(0)
                merged['prob'] = merged['selected'] / merged['total']
                
                # Add metadata
                merged['subgoal_type'] = subgoal_type
                merged['goal_type'] = goal_display
                merged['progress'] = merged[progress_var]
                merged['block_type'] = 'All Conditions'  # Collapsed label
                
                all_results.append(merged[['subgoal_type', 'goal_type', 'progress', 'block_type', 'prob']])
            else:
                # Compute counts per (progress, block_type)
                counts = subset.groupby([progress_var, 'block_type']).size().reset_index(name='total')
                
                # Compute selected counts (when subgoal_selected matches the subgoal_type)
                selected_subset = subset[subset['subgoal_selected'] == subgoal_type]
                selected_counts = selected_subset.groupby([progress_var, 'block_type']).size().reset_index(name='selected')
                
                # Merge and compute probability
                merged = pd.merge(counts, selected_counts, on=[progress_var, 'block_type'], how='left')
                merged['selected'] = merged['selected'].fillna(0)
                merged['prob'] = merged['selected'] / merged['total']
                
                # Add metadata
                merged['subgoal_type'] = subgoal_type
                merged['goal_type'] = goal_display
                merged['progress'] = merged[progress_var]
                
                all_results.append(merged[['subgoal_type', 'goal_type', 'progress', 'block_type', 'prob']])
        
        if not all_results:
            print("No data found for the specified progress variables")
            return
            
        # Combine all results
        final_data = pd.concat(all_results, ignore_index=True)

        sns.set_theme(style="white", font_scale=1.2)

        # Round progress values to ensure exact matching
        final_data['progress'] = final_data['progress'].round(2)
        
        # Filter and order progress bins (excluding 1) - use all available progress values
        unique_progress = sorted(final_data['progress'].dropna().unique())
        final_data['progress'] = pd.Categorical(final_data['progress'], categories=unique_progress, ordered=True)

        if collapse_over_subgoals:
            # Create a single plot collapsing over all subgoals
            # Aggregate data across all subgoals and conditions
            collapsed_data = final_data.groupby('progress').agg({
                'prob': ['mean', 'std', 'count']
            }).reset_index()
            
            # Flatten column names
            collapsed_data.columns = ['progress', 'prob_mean', 'prob_std', 'prob_count']
            
            # Calculate standard error
            collapsed_data['prob_se'] = collapsed_data['prob_std'] / np.sqrt(collapsed_data['prob_count'])
            
            plt.figure(figsize=(6, 6))
            plt.errorbar(
                collapsed_data['progress'],
                collapsed_data['prob_mean'],
                yerr=collapsed_data['prob_se'],
                marker='o',
                linewidth=2,
                capsize=5,
                capthick=2,
                color='black',
                markerfacecolor='black',
                markeredgecolor='black'
            )
            plt.xlabel("Subgoal Progress", fontsize=14, weight='bold')
            plt.ylabel("Subgoal Selection Probability", fontsize=14, weight='bold')
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            
            # Perform repeated measures ANOVA
            print("\n" + "="*80)
            print("REPEATED MEASURES ANOVA: Progress Effect on Subgoal Selection")
            print("="*80)
            
            # Prepare data for repeated measures ANOVA
            # We need to restructure the data to have subject-level observations
            # Get the original dataframe with subject information
            if simulate:
                # For simulated data, we need to reconstruct subject-level data
                behavior_dataframe_csv = CACHE_DIR + "all_subjects_behavior_data_" + self.data_type + "_momentum_learn_alt_goal_simulated.csv"
                df_original = pd.read_csv(behavior_dataframe_csv)
            else:
                df_original = self.df.copy()
            
            # Apply block type replacement
            df_original['block_type'] = df_original['block_type'].replace(block_labels)
            df_original['goal_selected'] = df_original['goal_selected'].replace('BR', 'OB')
            
            # Create subject-level data for ANOVA
            subject_progress_data = []
            
            # Process each progress configuration to get subject-level data
            for progress_var, subgoal_type, goal_type in progress_configs:
                goal_display = 'OB' if goal_type == 'BR' else goal_type
                
                # Filter data where this progress variable exists and goal matches
                mask = (df_original[progress_var].notna()) & (df_original['goal_selected'] == goal_display)
                subset = df_original[mask].copy()
                
                # Remove progress = 1 and 0.5 as in the original method
                subset = subset[subset[progress_var] != 1.0]
                subset = subset[subset[progress_var] != 0.5]
                
                # Round progress values
                subset[progress_var] = subset[progress_var].round(2)
                
                if subset.empty:
                    continue
                
                # For each subject and progress level, calculate selection probability
                for subject_id in subset['subject_id'].unique():
                    subject_data = subset[subset['subject_id'] == subject_id]
                    
                    # Get unique progress values for this subject
                    unique_progress = subject_data[progress_var].unique()
                    
                    for progress_level in unique_progress:
                        progress_trials = subject_data[subject_data[progress_var] == progress_level]
                        if len(progress_trials) > 0:
                            # Create binary selection variable (1 if subgoal was selected, 0 otherwise)
                            selection_binary = (progress_trials['subgoal_selected'] == subgoal_type).astype(int)
                            selection_prob = selection_binary.mean()
                            
                            subject_progress_data.append({
                                'subject_id': subject_id,
                                'progress': progress_level,
                                'selection_prob': selection_prob,
                                'subgoal_type': subgoal_type,
                                'goal_type': goal_display
                            })
            
            # Convert to DataFrame for ANOVA
            anova_df = pd.DataFrame(subject_progress_data)
            
            if len(anova_df) > 0:
                # Perform repeated measures ANOVA using pingouin
                try:
                    # Use pingouin's rm_anova function
                    rm_anova_result = pg.rm_anova(data=anova_df, dv='selection_prob', within='progress', subject='subject_id')
                    
                    print(f"Repeated Measures ANOVA Results:")
                    print(f"F({rm_anova_result['ddof1'].iloc[0]:.0f}, {rm_anova_result['ddof2'].iloc[0]:.0f}) = {rm_anova_result['F'].iloc[0]:.3f}")
                    print(f"p-value = {rm_anova_result['p-unc'].iloc[0]:.6f}")
                    print(f"Effect size (η²p) = {rm_anova_result['np2'].iloc[0]:.3f}")
                    
                    # Interpret significance
                    if rm_anova_result['p-unc'].iloc[0] < 0.001:
                        significance = "***"
                    elif rm_anova_result['p-unc'].iloc[0] < 0.01:
                        significance = "**"
                    elif rm_anova_result['p-unc'].iloc[0] < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"
                    
                    print(f"Significance: {significance}")
                    
                    # Post-hoc pairwise comparisons if significant
                    if rm_anova_result['p-unc'].iloc[0] < 0.05:
                        print(f"\nPost-hoc pairwise comparisons (Bonferroni corrected):")
                        pairwise_results = pg.pairwise_ttests(data=anova_df, dv='selection_prob', within='progress', subject='subject_id', padjust='bonf')
                        
                        # Show only significant comparisons
                        significant_pairs = pairwise_results[pairwise_results['p-corr'] < 0.05]
                        if len(significant_pairs) > 0:
                            for _, row in significant_pairs.iterrows():
                                print(f"  Progress {row['A']} vs {row['B']}: t = {row['T']:.3f}, p = {row['p-corr']:.3f}")
                        else:
                            print("  No significant pairwise differences found.")
                    
                    # Descriptive statistics
                    print(f"\nDescriptive Statistics:")
                    desc_stats = anova_df.groupby('progress')['selection_prob'].agg(['mean', 'std', 'count']).round(3)
                    unique_progress_levels = sorted(anova_df['progress'].unique())
                    for progress_level in unique_progress_levels:
                        if progress_level in desc_stats.index:
                            mean_val = desc_stats.loc[progress_level, 'mean']
                            std_val = desc_stats.loc[progress_level, 'std']
                            count_val = desc_stats.loc[progress_level, 'count']
                            print(f"  Progress {progress_level}: M = {mean_val:.3f}, SD = {std_val:.3f}, n = {count_val:.0f}")
                    
                    # Additional analysis: breakdown by subgoal type
                    print(f"\nAnalysis by Subgoal Type:")
                    for subgoal in anova_df['subgoal_type'].unique():
                        subgoal_data = anova_df[anova_df['subgoal_type'] == subgoal]
                        if len(subgoal_data) > 0:
                            try:
                                subgoal_anova = pg.rm_anova(data=subgoal_data, dv='selection_prob', within='progress', subject='subject_id')
                                print(f"  {subgoal}: F({subgoal_anova['ddof1'].iloc[0]:.0f}, {subgoal_anova['ddof2'].iloc[0]:.0f}) = {subgoal_anova['F'].iloc[0]:.3f}, p = {subgoal_anova['p-unc'].iloc[0]:.3f}")
                            except:
                                print(f"  {subgoal}: Insufficient data for ANOVA")
                    
                except Exception as e:
                    print(f"Error performing repeated measures ANOVA: {e}")
                    print("This might be due to insufficient data or missing values.")
            else:
                print("Insufficient data for repeated measures ANOVA.")
            
            print("="*80)
        else:
            # Original behavior with separate subplots for each subgoal
            # Define custom color palette
            if collapse_conditions:
                palette = ['black']  # Single color for collapsed data
                g = sns.FacetGrid(
                    final_data,
                    col='subgoal_type',
                    sharey=True,
                    height=4,
                    aspect=0.7
                )
            else:
                palette = sns.color_palette("Dark2")
                # Create FacetGrid first (it will create its own figure)
                g = sns.FacetGrid(
                    final_data,
                    col='subgoal_type',
                    sharey=True,
                    height=4,
                    aspect=0.7
                )
                
                # Adjust figure size to make room for legend
                fig = g.fig
                fig.set_size_inches(16, 4)
                
                # Reposition existing axes to make room for legend and reduce spacing
                # Calculate positions for all subplots with minimal spacing
                n_subplots = len(g.axes.flat)
                total_width = 0.75  # Use 75% of figure width for all subplots
                spacing = 0.02  # Small gap between subplots
                subplot_width = (total_width - (n_subplots - 1) * spacing) / n_subplots
                start_x = 0.1  # Start position
                
                for i, ax in enumerate(g.axes.flat):
                    current_pos = ax.get_position()
                    new_x0 = start_x + i * (subplot_width + spacing)
                    ax.set_position([new_x0, current_pos.y0, subplot_width, current_pos.height])

            if collapse_conditions:
                g.map_dataframe(
                    sns.lineplot,
                    x='progress',
                    y='prob',
                    marker='o',
                    linewidth=3,
                    color='black',
                    markersize=8
                )
            else:
                # Use constant color with different markers for each condition
                # Define markers for different block types
                markers = ['o', 's', '^', 'D', 'v', 'p']  # circle, square, triangle, diamond, triangle_down, pentagon
                block_types = sorted(final_data['block_type'].unique())
                marker_dict = {bt: markers[i % len(markers)] for i, bt in enumerate(block_types)}
                
                # Plot each block type separately with same color but different markers on each subplot
                for i, ax in enumerate(g.axes.flat):
                    subgoal_name = g.col_names[i]
                    subgoal_data = final_data[final_data['subgoal_type'] == subgoal_name]
                    
                    for block_type in block_types:
                        block_data = subgoal_data[subgoal_data['block_type'] == block_type]
                        if len(block_data) > 0:
                            # Sort by progress for proper line plotting
                            block_data = block_data.sort_values('progress')
                            ax.plot(
                                block_data['progress'],
                                block_data['prob'],
                                marker=marker_dict[block_type],
                                linestyle=':',
                                linewidth=2,
                                color='black',
                                markersize=8,
                                label=block_type
                            )

        if not collapse_over_subgoals:
            # Clean titles and axis labels
            custom_titles = {
                'G': 'P(Select G)',
                'C': 'P(Select C)', 
                'S': 'P(Select S)',
                'M': 'P(Select M)'
            }

            # Apply the custom titles
            for ax, title_key in zip(g.axes.flat, g.col_names):
                ax.set_title(custom_titles.get(title_key, title_key), fontsize=13, weight='bold', pad=0)
            g.set_axis_labels("Subgoal Progress", "Subgoal Selection Probability", size=14, weight='bold')

            # Remove y-axis labels for subplots 2-4 (index 1, 2, 3)
            for i, ax in enumerate(g.axes.flat):
                if i > 0:
                    ax.set_xlabel("")
            g.set(ylim=(0, 1.05))

            # Add legend in a separate subplot
            if not collapse_conditions:
                # Create custom legend with markers
                legend_handles = []
                for block_type in block_types:
                    if block_type in marker_dict:
                        legend_handles.append(plt.Line2D([0], [0], 
                                                          marker=marker_dict[block_type], 
                                                          linestyle=':', 
                                                          color='black', 
                                                          linewidth=2, 
                                                          markersize=8, 
                                                          label=block_type))
                
                # Create legend subplot in the rightmost position
                # Get position from the rightmost plot and place legend to its right
                rightmost_ax = g.axes.flat[-1]
                rightmost_pos = rightmost_ax.get_position()
                legend_x0 = rightmost_pos.x0 + rightmost_pos.width + 0.02
                legend_ax = fig.add_axes([legend_x0, rightmost_pos.y0, 0.15, rightmost_pos.height])
                legend_ax.axis('off')
                legend_ax.legend(handles=legend_handles, loc='center left', title='Block Type', frameon=False)

        if self.data_type == "online":
            experiment = "H2"
        elif self.data_type == "fmri":
            experiment = "H1"
        
        if collapse_over_subgoals:
            plt.suptitle('Subgoal selection with progress', fontsize=16, weight='bold', y=0.95)
            if simulate:
                plt.savefig(self.figure_dir + "/subgoal_selection_related_subgoal_progress_" + self.experiment + "_simulated_collapsed_subgoals.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.figure_dir + "/subgoal_selection_related_subgoal_progress_" + self.experiment + "_collapsed_subgoals.png", dpi=300, bbox_inches='tight')
        elif collapse_conditions:
            plt.suptitle('Subgoal Selection Probability by Own Progress (Collapsed Across Conditions)', fontsize=16, weight='bold', y=1.02)
            plt.savefig(FIGURES + "/subgoal_selection_related_subgoal_progress_collapsed.png", dpi=300, bbox_inches='tight')
        elif simulate:
            plt.savefig(self.figure_dir + "/subgoal_selection_related_subgoal_progress_" + self.experiment + "_simulated.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.figure_dir + "/subgoal_selection_related_subgoal_progress_" + self.experiment + ".png", dpi=300, bbox_inches='tight')

        print(self.figure_dir)
        plt.show()

    def plot_subgoal_selection_related_subgoal_progress_simulated(self, models=['momentum_learn_alt_goal', 'prospective', 'td_persistence'], 
                                                                 collapse_conditions=True, collapse_over_subgoals=True, include_behavior=True):
        """
        Plot subgoal selection related subgoal progress for different simulated models.
        
        Args:
            models (list): List of model names to simulate
            collapse_conditions (bool): If True, collapses over conditions
            collapse_over_subgoals (bool): If True, collapses over subgoals
            include_behavior (bool): If True, includes actual behavior data as 'Behavior' model
        """
        # Define all possible subgoal progress variables to plot
        progress_configs = [
            # SH goal subgoals
            ('G_progress_SH', 'G', 'SH'),
            ('M_progress_SH', 'M', 'SH'),
            ('C_progress_SH', 'C', 'SH'),
            ('S_progress_SH', 'S', 'SH'),
            
            # BR goal subgoals  
            ('G_progress_BR', 'G', 'BR'),
            ('S_progress_BR', 'S', 'BR'),
            ('C_progress_BR', 'C', 'BR'),
            ('M_progress_BR', 'M', 'BR'),
            
            # HO goal subgoals
            ('G_progress_HO', 'G', 'HO'),
            ('S_progress_HO', 'S', 'HO'),
            ('C_progress_HO', 'C', 'HO'),
            ('M_progress_HO', 'M', 'HO')
        ]
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }

        # Model display names and colors
        model_display_names = {
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal',
            'prospective': 'Prospective',
            'td_persistence': 'TD-persistence',
            'behavior': 'Behavior'
        }
        
        model_colors = {
            'momentum_learn_alt_goal': '#1f77b4',  # Dark blue
            'prospective': '#ff7f0e',  # Dark orange
            'td_persistence': '#2ca02c',  # Dark green
            'behavior': '#d62728'  # Dark red
        }

        all_results = []
        
        for model_name in models:
            # Load simulated data for this model
            behavior_dataframe_csv = f"{CACHE_DIR}all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
            df = pd.read_csv(behavior_dataframe_csv)
            
            # Apply the replacement
            df['block_type'] = df['block_type'].replace(block_labels)
            
            # Replace 'BR' with 'OB' for consistency
            df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')

            # Process each progress variable separately
            model_results = []
            
            for progress_var, subgoal_type, goal_type in progress_configs:
                # Convert goal_type for consistency
                goal_display = 'OB' if goal_type == 'BR' else goal_type
                
                # Filter data where this progress variable exists and goal matches
                mask = (df[progress_var].notna()) & (df['goal_selected'] == goal_display)
                subset = df[mask].copy()
                
                # Remove progress = 1 and 0.5 as requested
                subset = subset[subset[progress_var] != 1.0]
                subset = subset[subset[progress_var] != 0.5]
                
                # Round progress values to 2 decimal places for exact matching
                subset[progress_var] = subset[progress_var].round(2)
                
                if subset.empty:
                    continue
                
                if collapse_conditions:
                    # Compute counts per progress only (collapsed across block types)
                    counts = subset.groupby([progress_var]).size().reset_index(name='total')
                    
                    # Compute selected counts (when subgoal_selected matches the subgoal_type)
                    selected_subset = subset[subset['subgoal_selected'] == subgoal_type]
                    selected_counts = selected_subset.groupby([progress_var]).size().reset_index(name='selected')
                    
                    # Merge and compute probability
                    merged = pd.merge(counts, selected_counts, on=[progress_var], how='left')
                    merged['selected'] = merged['selected'].fillna(0)
                    merged['prob'] = merged['selected'] / merged['total']
                    
                    # Add metadata
                    merged['subgoal_type'] = subgoal_type
                    merged['goal_type'] = goal_display
                    merged['progress'] = merged[progress_var]
                    merged['block_type'] = 'All Conditions'  # Collapsed label
                    merged['model'] = model_name
                    merged['model_display'] = model_display_names[model_name]
                    
                    model_results.append(merged[['subgoal_type', 'goal_type', 'progress', 'block_type', 'prob', 'model', 'model_display']])
                else:
                    # Compute counts per (progress, block_type)
                    counts = subset.groupby([progress_var, 'block_type']).size().reset_index(name='total')
                    
                    # Compute selected counts (when subgoal_selected matches the subgoal_type)
                    selected_subset = subset[subset['subgoal_selected'] == subgoal_type]
                    selected_counts = selected_subset.groupby([progress_var, 'block_type']).size().reset_index(name='selected')
                    
                    # Merge and compute probability
                    merged = pd.merge(counts, selected_counts, on=[progress_var, 'block_type'], how='left')
                    merged['selected'] = merged['selected'].fillna(0)
                    merged['prob'] = merged['selected'] / merged['total']
                    
                    # Add metadata
                    merged['subgoal_type'] = subgoal_type
                    merged['goal_type'] = goal_display
                    merged['progress'] = merged[progress_var]
                    merged['model'] = model_name
                    merged['model_display'] = model_display_names[model_name]
                    
                    model_results.append(merged[['subgoal_type', 'goal_type', 'progress', 'block_type', 'prob', 'model', 'model_display']])
            
            if model_results:
                all_results.extend(model_results)

        # Add behavior data if requested
        if include_behavior:
            # Use the original dataframe (self.df) for behavior data
            df_behavior = self.df.copy()
            
            # Apply the replacement
            df_behavior['block_type'] = df_behavior['block_type'].replace(block_labels)
            
            # Replace 'BR' with 'OB' for consistency
            df_behavior['goal_selected'] = df_behavior['goal_selected'].replace('BR', 'OB')

            # Process each progress variable separately
            behavior_results = []
            
            for progress_var, subgoal_type, goal_type in progress_configs:
                # Convert goal_type for consistency
                goal_display = 'OB' if goal_type == 'BR' else goal_type
                
                # Filter data where this progress variable exists and goal matches
                mask = (df_behavior[progress_var].notna()) & (df_behavior['goal_selected'] == goal_display)
                subset = df_behavior[mask].copy()
                
                # Remove progress = 1 and 0.5 as requested
                subset = subset[subset[progress_var] != 1.0]
                subset = subset[subset[progress_var] != 0.5]
                
                # Round progress values to 2 decimal places for exact matching
                subset[progress_var] = subset[progress_var].round(2)
                
                if subset.empty:
                    continue
                
                if collapse_conditions:
                    # Compute counts per progress only (collapsed across block types)
                    counts = subset.groupby([progress_var]).size().reset_index(name='total')
                    
                    # Compute selected counts (when subgoal_selected matches the subgoal_type)
                    selected_subset = subset[subset['subgoal_selected'] == subgoal_type]
                    selected_counts = selected_subset.groupby([progress_var]).size().reset_index(name='selected')
                    
                    # Merge and compute probability
                    merged = pd.merge(counts, selected_counts, on=[progress_var], how='left')
                    merged['selected'] = merged['selected'].fillna(0)
                    merged['prob'] = merged['selected'] / merged['total']
                    
                    # Add metadata
                    merged['subgoal_type'] = subgoal_type
                    merged['goal_type'] = goal_display
                    merged['progress'] = merged[progress_var]
                    merged['block_type'] = 'All Conditions'  # Collapsed label
                    merged['model'] = 'behavior'
                    merged['model_display'] = 'Behavior'
                    
                    behavior_results.append(merged[['subgoal_type', 'goal_type', 'progress', 'block_type', 'prob', 'model', 'model_display']])
                else:
                    # Compute counts per (progress, block_type)
                    counts = subset.groupby([progress_var, 'block_type']).size().reset_index(name='total')
                    
                    # Compute selected counts (when subgoal_selected matches the subgoal_type)
                    selected_subset = subset[subset['subgoal_selected'] == subgoal_type]
                    selected_counts = selected_subset.groupby([progress_var, 'block_type']).size().reset_index(name='selected')
                    
                    # Merge and compute probability
                    merged = pd.merge(counts, selected_counts, on=[progress_var, 'block_type'], how='left')
                    merged['selected'] = merged['selected'].fillna(0)
                    merged['prob'] = merged['selected'] / merged['total']
                    
                    # Add metadata
                    merged['subgoal_type'] = subgoal_type
                    merged['goal_type'] = goal_display
                    merged['progress'] = merged[progress_var]
                    merged['model'] = 'behavior'
                    merged['model_display'] = 'Behavior'
                    
                    behavior_results.append(merged[['subgoal_type', 'goal_type', 'progress', 'block_type', 'prob', 'model', 'model_display']])
            
            if behavior_results:
                all_results.extend(behavior_results)

        if not all_results:
            print("No data found for the specified progress variables")
            return
            
        # Combine all results
        final_data = pd.concat(all_results, ignore_index=True)

        sns.set_theme(style="white", font_scale=1.2)

        # Round progress values to ensure exact matching
        final_data['progress'] = final_data['progress'].round(2)
        
        # Filter and order progress bins (excluding 1) - use all available progress values
        unique_progress = sorted(final_data['progress'].dropna().unique())
        final_data['progress'] = pd.Categorical(final_data['progress'], categories=unique_progress, ordered=True)

        if collapse_over_subgoals:
            # Create a single plot collapsing over all subgoals
            # Aggregate data across all subgoals and conditions
            collapsed_data = final_data.groupby(['progress', 'model']).agg({
                'prob': ['mean', 'std', 'count']
            }).reset_index()
            
            # Flatten column names
            collapsed_data.columns = ['progress', 'model', 'prob_mean', 'prob_std', 'prob_count']
            
            # Calculate standard error
            collapsed_data['prob_se'] = collapsed_data['prob_std'] / np.sqrt(collapsed_data['prob_count'])
            
            plt.figure(figsize=(10, 6))
            
            # Plot all models including behavior if requested
            all_models = models.copy()
            if include_behavior:
                all_models.append('behavior')
                
            for model_name in all_models:
                model_data = collapsed_data[collapsed_data['model'] == model_name]
                plt.errorbar(
                    model_data['progress'],
                    model_data['prob_mean'],
                    yerr=model_data['prob_se'],
                    marker='o',
                    linewidth=2,
                    capsize=5,
                    capthick=2,
                    label=model_display_names[model_name],
                    color=model_colors[model_name]
                )
            
            plt.xlabel("Subgoal Progress", fontsize=14, weight='bold')
            plt.ylabel("Subgoal Selection Probability", fontsize=14, weight='bold')
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Model', fontsize=12)
            plt.title('Subgoal Selection Probability by Progress (Simulated Models)', fontsize=16, weight='bold')
            
        else:
            # Show separate subplots for each subgoal
            g = sns.FacetGrid(
                final_data,
                col='subgoal_type',
                sharey=True,
                height=4,
                aspect=0.7
            )

            g.map_dataframe(
                sns.lineplot,
                x='progress',
                y='prob',
                hue='model',
                style='block_type' if not collapse_conditions else None,
                marker='o',
                linewidth=2,
                palette=model_colors
            )

            # Clean titles and axis labels
            custom_titles = {
                'G': 'P(Select G)',
                'C': 'P(Select C)', 
                'S': 'P(Select S)',
                'M': 'P(Select M)'
            }

            # Apply the custom titles
            for ax, title_key in zip(g.axes.flat, g.col_names):
                ax.set_title(custom_titles.get(title_key, title_key), fontsize=13, weight='bold', pad=0)
            g.set_axis_labels("Subgoal Progress", "Subgoal Selection Probability", size=14, weight='bold')

            # Remove y-axis labels for subplots 2-4 (index 1, 2, 3)
            for i, ax in enumerate(g.axes.flat):
                if i > 0:
                    ax.set_xlabel("")
            g.set(ylim=(0, 1.05))

            # Add legend
            g.add_legend(title='Model')

        # Save the plot
        if self.data_type == "online":
            experiment = "H2"
        elif self.data_type == "fmri":
            experiment = "H1"
            
        # Determine filename suffix based on whether behavior is included
        behavior_suffix = "_with_behavior" if include_behavior else ""
        
        if collapse_over_subgoals:
            plt.savefig(self.figure_dir + f"/subgoal_selection_related_subgoal_progress_simulated_{self.experiment}_collapsed{behavior_suffix}.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.figure_dir + f"/subgoal_selection_related_subgoal_progress_simulated_{self.experiment}{behavior_suffix}.png", dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_goal_selection_related_goal_progress_four_models(self, models=['momentum', 'prospective', 'td_persistence'], 
                                                           include_behavior=True, collapse_over_goals=False):
        """
        Plot goal selection related goal progress for different models in four separate rows.
        
        Args:
            models (list): List of model names to simulate
            include_behavior (bool): If True, includes actual behavior data as 'Behavior' model
            collapse_over_goals (bool): If True, collapses over goals and overlays all models in one figure
        """
        goals = ['SH', 'BR', 'HO']
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high",
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }

        # Model display names and colors
        model_display_names = {
            'momentum': 'TD-momentum',
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal',
            'prospective': 'Prospective',
            'td_persistence': 'TD-persistence',
            'behavior': 'Behavior'
        }
        
        model_colors = {
            'momentum': '#1f77b4',  # Dark blue
            'momentum_learn_alt_goal': '#1f77b4',  # Dark blue
            'prospective': '#ff7f0e',  # Dark orange
            'td_persistence': '#2ca02c',  # Dark green
            'behavior': '#d62728'  # Dark red
        }

        # Create list of all models to plot
        all_models = models.copy()
        if include_behavior:
            all_models.append('behavior')

        # Define markers for different block types (constant black color, different markers)
        # We'll define this before the loop to use across all models
        markers = ['o', 's', '^', 'D', 'v', 'p']  # circle, square, triangle, diamond, triangle_down, pentagon
        marker_dict = None  # Will be set in first iteration
        gs_main = None  # Will be set for non-collapsed mode
        
        # Create figure - single plot when collapsing, multiple rows otherwise
        if collapse_over_goals:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            axes = None  # Not used in collapsed mode
        else:
            # Create figure with extra space for legend subplot
            # Increased height and reduced width for subplots
            fig = plt.figure(figsize=(12, 5 * len(all_models)))
            gs_main = gridspec.GridSpec(len(all_models), 4, width_ratios=[1, 1, 1, 0.3], figure=fig, wspace=0.02)
            # Create subplots for data (first 3 columns)
            axes = np.empty((len(all_models), 3), dtype=object)
            for i in range(len(all_models)):
                for j in range(3):
                    axes[i, j] = fig.add_subplot(gs_main[i, j])
            if len(all_models) == 1:
                axes = axes.reshape(1, -1)
        
        sns.set_theme(style="white", font_scale=1.2)

        # Store collapsed data for bar plot if collapsing
        collapsed_data_all_models = [] if collapse_over_goals else None

        for i, model_name in enumerate(all_models):
            if model_name == 'behavior':
                # Use original behavior data
                df = self.df.copy()
            else:
                # Load simulated data for this model
                behavior_dataframe_csv = f"{CACHE_DIR}all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
                df = pd.read_csv(behavior_dataframe_csv)
            
            # Apply the replacement
            df['block_type'] = df['block_type'].replace(block_labels)

            melted = pd.melt(
                df,
                id_vars=['goal_selected', 'block_type'],
                value_vars=[f'{g}_progress' for g in goals],
                var_name='goal_progressed',
                value_name='progress'
            )
            # Clean goal_progressed to just "SH", "BR", "HO"
            melted['goal'] = melted['goal_progressed'].str.extract(r'(\w+)_progress')

            # Step 4: Compute counts per (goal, progress_bin, block_type)
            counts = melted.groupby(['goal', 'progress', 'block_type']).size().reset_index(name='total')
            selected = melted[melted['goal'] == melted['goal_selected']]
            selected_counts = selected.groupby(['goal', 'progress', 'block_type']).size().reset_index(name='selected')
            # Step 6: Merge and compute probability
            merged = pd.merge(counts, selected_counts, on=['goal', 'progress', 'block_type'], how='left')
            merged['selected'] = merged['selected'].fillna(0)
            merged['prob'] = merged['selected'] / merged['total']
            merged = merged[merged['progress'].isin([0, 1, 2, 3, 4, 5])]

            # Filter and order progress bins
            merged['progress'] = pd.Categorical(merged['progress'], categories=[0, 1, 2, 3, 4, 5], ordered=True)

            # Define markers for different block types (only once)
            if marker_dict is None:
                all_block_types = sorted(merged['block_type'].unique())
                marker_dict = {bt: markers[k % len(markers)] for k, bt in enumerate(all_block_types)}

            if collapse_over_goals:
                # Create a single plot collapsing over both goals and block types
                # Aggregate data across goals and block types
                collapsed_data = merged.groupby('progress').agg({
                    'prob': ['mean', 'std', 'count']
                }).reset_index()
                
                # Flatten column names
                collapsed_data.columns = ['progress', 'prob_mean', 'prob_std', 'prob_count']
                
                # Calculate standard error
                collapsed_data['prob_se'] = collapsed_data['prob_std'] / np.sqrt(collapsed_data['prob_count'])
                
                # Add model name for grouping
                collapsed_data['model'] = model_display_names[model_name]
                
                # Store for later plotting
                collapsed_data_all_models.append(collapsed_data)
            else:
                # Plot each goal in separate columns
                goal_names = ['SH', 'BR', 'HO']
                goal_titles = {'SH': 'Goal: SH', 'BR': 'Goal: OB', 'HO': 'Goal: HO'}
                
                for j, goal in enumerate(goal_names):
                    ax = axes[i, j]
                    
                    # Filter data for this goal
                    goal_data = merged[merged['goal'] == goal]
                    
                    if not goal_data.empty:
                        # Plot each block type as separate line with black color and different markers
                        for k, block_type in enumerate(goal_data['block_type'].unique()):
                            block_data = goal_data[goal_data['block_type'] == block_type]
                            # Sort by progress for proper line plotting
                            block_data = block_data.sort_values('progress')
                            ax.plot(
                                block_data['progress'],
                                block_data['prob'],
                                marker=marker_dict[block_type],
                                linestyle=':',
                                linewidth=2,
                                color='black',
                                markersize=5,
                                label=f'{block_type}'
                            )
                    
                    # Only add title for the first row (momentum model)
                    if i == 0:
                        ax.set_title(goal_titles[goal], fontsize=14, weight='bold')
                    
                    ax.set_ylim(0, 1.05)
                    ax.grid(True, alpha=0.3)
                    
                    # Only add xlabel for the last two rows (behavior and td_persistence)
                    if i >= 2:
                        ax.set_xlabel("Goal Progress", fontsize=12, weight='bold')
                    
                    # Only add ylabel to the first column
                    if j == 0:
                        ax.set_ylabel(f'{model_display_names[model_name]}\nGoal Selection Probability', fontsize=12, weight='bold')
                    
                    # Legend will be added separately after all plots

        # Format the collapsed overlay plot with grouped bar plots
        if collapse_over_goals:
            # Combine all model data
            combined_data = pd.concat(collapsed_data_all_models, ignore_index=True)
            
            # Reorder models so behavior is first (leftmost)
            models_ordered = []
            if 'behavior' in all_models:
                models_ordered.append('behavior')
            models_ordered.extend([m for m in all_models if m != 'behavior'])
            
            # Set up bar positions
            progress_values = sorted(combined_data['progress'].unique())
            n_models = len(models_ordered)
            bar_width = 0.15  # Width of each individual bar
            group_width = bar_width * n_models  # Total width for all bars in a group
            
            # Plot grouped bars
            for j, model_name in enumerate(models_ordered):
                model_display = model_display_names[model_name]
                model_data = combined_data[combined_data['model'] == model_display]
                
                # Calculate x positions for this model's bars
                # Center bars around each progress value
                offset = (j - (n_models - 1) / 2) * bar_width
                x_positions = [p + offset for p in progress_values]
                
                # Get values in order
                values = []
                errors = []
                for p in progress_values:
                    row = model_data[model_data['progress'] == p]
                    if not row.empty:
                        values.append(row['prob_mean'].values[0])
                        errors.append(row['prob_se'].values[0])
                    else:
                        values.append(0)
                        errors.append(0)
                
                # Plot bars with error bars
                ax.bar(
                    x_positions,
                    values,
                    bar_width,
                    yerr=errors,
                    label=model_display,
                    color=model_colors[model_name],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1,
                    capsize=5
                )
            
            ax.set_xlabel("Goal Progress", fontsize=14, weight='bold')
            ax.set_ylabel("Goal Selection Probability", fontsize=14, weight='bold')
            ax.set_xticks(progress_values)
            ax.set_xticklabels(progress_values)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(title='Model', fontsize=12, loc='best')
            ax.set_title('Goal Selection Probability by Progress (All Models)', fontsize=16, weight='bold')

        # Overall title
        if self.data_type == "online":
            experiment = "H2"
        elif self.data_type == "fmri":
            experiment = "H1"
        
        if not collapse_over_goals:
            fig.suptitle(f'Goal Selection Probability by Progress', fontsize=18, weight='bold', y=0.98)
            
            # Add legend in separate subplot
            if marker_dict:
                # Create custom legend with markers
                legend_handles = []
                for block_type in sorted(marker_dict.keys()):
                    legend_handles.append(plt.Line2D([0], [0], 
                                                      marker=marker_dict[block_type], 
                                                      linestyle=':', 
                                                      color='black', 
                                                      linewidth=2, 
                                                      markersize=5, 
                                                      label=block_type))
                
                # Create legend subplot spanning all rows in the rightmost column
                legend_ax = fig.add_subplot(gs_main[:, 3])  # Span all rows in last column
                legend_ax.axis('off')
                legend_ax.legend(handles=legend_handles, loc='center left', title='Block Type', frameon=False)
        
        # Save the plot
        behavior_suffix = "_with_behavior" if include_behavior else ""
        collapse_suffix = "_collapsed" if collapse_over_goals else ""
        plt.savefig(self.figure_dir + f"/goal_selection_related_goal_progress_four_models_{self.experiment}{collapse_suffix}{behavior_suffix}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_subgoal_selection_related_subgoal_progress_four_models(self, models=['momentum_learn_alt_goal', 'prospective', 'td_persistence'], 
                                                                 include_behavior=True, collapse_over_subgoals=False):
        """
        Plot subgoal selection related subgoal progress for different models in four separate rows.
        
        Args:
            models (list): List of model names to simulate
            include_behavior (bool): If True, includes actual behavior data as 'Behavior' model
            collapse_over_subgoals (bool): If True, collapses over subgoals
        """
        # Define all possible subgoal progress variables to plot
        progress_configs = [
            # SH goal subgoals
            ('G_progress_SH', 'G', 'SH'),
            ('M_progress_SH', 'M', 'SH'),
            ('C_progress_SH', 'C', 'SH'),
            ('S_progress_SH', 'S', 'SH'),
            
            # BR goal subgoals  
            ('G_progress_BR', 'G', 'BR'),
            ('S_progress_BR', 'S', 'BR'),
            ('C_progress_BR', 'C', 'BR'),
            ('M_progress_BR', 'M', 'BR'),
            
            # HO goal subgoals
            ('G_progress_HO', 'G', 'HO'),
            ('S_progress_HO', 'S', 'HO'),
            ('C_progress_HO', 'C', 'HO'),
            ('M_progress_HO', 'M', 'HO')
        ]
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }

        # Model display names and colors
        model_display_names = {
            'momentum': 'TD-momentum',
            'momentum_learn_alt_goal': 'TD-momentum-alt-goal',
            'prospective': 'Prospective',
            'td_persistence': 'TD-persistence',
            'behavior': 'Behavior'
        }
        
        model_colors = {
            'momentum': '#1f77b4',  # Dark blue
            'momentum_learn_alt_goal': '#1f77b4',  # Dark blue
            'prospective': '#ff7f0e',  # Dark orange
            'td_persistence': '#2ca02c',  # Dark green
            'behavior': '#d62728'  # Dark red
        }

        # Create list of all models to plot
        all_models = models.copy()
        if include_behavior:
            all_models.append('behavior')

        # Define markers for different block types (constant black color, different markers)
        markers = ['o', 's', '^', 'D', 'v', 'p']  # circle, square, triangle, diamond, triangle_down, pentagon
        marker_dict = None  # Will be set in first iteration
        gs_main = None  # Will be set for non-collapsed mode

        # Create figure - single plot when collapsing, multiple rows otherwise
        if collapse_over_subgoals:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            axes = None  # Not used in collapsed mode
        else:
            # Create figure with extra space for legend subplot
            # Increased height and reduced width for subplots
            fig = plt.figure(figsize=(12, 5 * len(all_models)))
            gs_main = gridspec.GridSpec(len(all_models), 4, width_ratios=[1, 1, 1, 0.3], figure=fig, wspace=0.02)
            # Create subplots for data (first 3 columns)
            axes = np.empty((len(all_models), 3), dtype=object)
            for i in range(len(all_models)):
                for j in range(3):
                    axes[i, j] = fig.add_subplot(gs_main[i, j])
            if len(all_models) == 1:
                axes = axes.reshape(1, -1)
        
        sns.set_theme(style="white", font_scale=1.2)

        # Store collapsed data for bar plot if collapsing
        collapsed_data_all_models = [] if collapse_over_subgoals else None

        for i, model_name in enumerate(all_models):
            if model_name == 'behavior':
                # Use original behavior data
                df = self.df.copy()
            else:
                # Load simulated data for this model
                behavior_dataframe_csv = f"{CACHE_DIR}all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
                df = pd.read_csv(behavior_dataframe_csv)
            
            # Apply the replacement
            df['block_type'] = df['block_type'].replace(block_labels)
            
            # Replace 'BR' with 'OB' for consistency
            df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')

            if collapse_over_subgoals:
                # Create a single plot collapsing over subgoals and block types
                all_subgoal_data = []
                
                for progress_var, subgoal_type, goal_type in progress_configs:
                    # Convert goal_type for consistency
                    goal_display = 'OB' if goal_type == 'BR' else goal_type
                    
                    # Filter data where this progress variable exists and goal matches
                    mask = (df[progress_var].notna()) & (df['goal_selected'] == goal_display)
                    subset = df[mask].copy()
                    
                    # Remove progress = 1 as requested
                    subset = subset[subset[progress_var] != 1.0]
                    
                    # Round progress values to 2 decimal places for exact matching
                    subset[progress_var] = subset[progress_var].round(2)
                    
                    if subset.empty:
                        continue
                    
                    # Compute counts per progress only (collapsed across block types)
                    counts = subset.groupby([progress_var]).size().reset_index(name='total')
                    
                    # Compute selected counts (when subgoal_selected matches the subgoal_type)
                    selected_subset = subset[subset['subgoal_selected'] == subgoal_type]
                    selected_counts = selected_subset.groupby([progress_var]).size().reset_index(name='selected')
                    
                    # Merge and compute probability
                    merged = pd.merge(counts, selected_counts, on=[progress_var], how='left')
                    merged['selected'] = merged['selected'].fillna(0)
                    merged['prob'] = merged['selected'] / merged['total']
                    
                    # Add metadata
                    merged['subgoal_type'] = subgoal_type
                    merged['goal_type'] = goal_display
                    merged['progress'] = merged[progress_var]
                    merged['model'] = model_name
                    
                    all_subgoal_data.append(merged[['subgoal_type', 'goal_type', 'progress', 'prob', 'model']])
                
                if all_subgoal_data:
                    # Combine all subgoal data
                    combined_data = pd.concat(all_subgoal_data, ignore_index=True)
                    
                    # Aggregate data across subgoals and goals
                    collapsed_data = combined_data.groupby('progress').agg({
                        'prob': ['mean', 'std', 'count']
                    }).reset_index()
                    
                    # Flatten column names
                    collapsed_data.columns = ['progress', 'prob_mean', 'prob_std', 'prob_count']
                    
                    # Calculate standard error
                    collapsed_data['prob_se'] = collapsed_data['prob_std'] / np.sqrt(collapsed_data['prob_count'])
                    
                    # Add model name for grouping
                    collapsed_data['model'] = model_display_names[model_name]
                    
                    # Store for later plotting
                    collapsed_data_all_models.append(collapsed_data)
            else:
                # Plot each goal in separate columns
                goal_names = ['SH', 'OB', 'HO']
                # Map each goal to its specific subgoal to plot
                goal_subgoal_map = {
                    'SH': 'G',  # Plot G for SH
                    'OB': 'S',  # Plot S for OB
                    'HO': 'C'   # Plot C for HO
                }
                goal_titles = {
                    'SH': 'P(subgoal = G | goal = SH)',
                    'OB': 'P(subgoal = S | goal = OB)',
                    'HO': 'P(subgoal = C | goal = HO)'
                }
                
                for j, goal in enumerate(goal_names):
                    ax = axes[i, j]
                    
                    # Get the specific subgoal to plot for this goal
                    target_subgoal = goal_subgoal_map[goal]
                    goal_for_progress = 'BR' if goal == 'OB' else goal
                    
                    # Find the progress config for this specific subgoal-goal combination
                    progress_var = f'{target_subgoal}_progress_{goal_for_progress}'
                    goal_config = None
                    for config in progress_configs:
                        if config[0] == progress_var and config[1] == target_subgoal:
                            goal_config = config
                            break
                    
                    if goal_config:
                        progress_var, subgoal_type, goal_type = goal_config
                        goal_display = 'OB' if goal == 'OB' else goal
                        
                        # Filter data where this progress variable exists and goal matches
                        mask = (df[progress_var].notna()) & (df['goal_selected'] == goal_display)
                        subset = df[mask].copy()
                        
                        # Remove progress = 1 and progress = 0.5 as requested
                        subset = subset[subset[progress_var] != 1.0]
                        subset = subset[subset[progress_var] != 0.5]
                        
                        # Round progress values to 2 decimal places for exact matching
                        subset[progress_var] = subset[progress_var].round(2)
                        
                        if not subset.empty:
                            # Compute counts per (progress, block_type)
                            counts = subset.groupby([progress_var, 'block_type']).size().reset_index(name='total')
                            
                            # Compute selected counts (when subgoal_selected matches the subgoal_type)
                            selected_subset = subset[subset['subgoal_selected'] == subgoal_type]
                            selected_counts = selected_subset.groupby([progress_var, 'block_type']).size().reset_index(name='selected')
                            
                            # Merge and compute probability
                            merged = pd.merge(counts, selected_counts, on=[progress_var, 'block_type'], how='left')
                            merged['selected'] = merged['selected'].fillna(0)
                            merged['prob'] = merged['selected'] / merged['total']
                            merged['progress'] = merged[progress_var]
                            
                            # Define markers for different block types (only once)
                            if marker_dict is None:
                                all_block_types = sorted(merged['block_type'].unique())
                                marker_dict = {bt: markers[k % len(markers)] for k, bt in enumerate(all_block_types)}
                            
                            # Plot each block type as separate line with black color and different markers
                            for k, block_type in enumerate(merged['block_type'].unique()):
                                block_data = merged[merged['block_type'] == block_type]
                                if not block_data.empty:
                                    # Sort by progress for proper line plotting
                                    block_data = block_data.sort_values('progress')
                                    ax.plot(
                                        block_data['progress'],
                                        block_data['prob'],
                                        marker=marker_dict[block_type],
                                        linestyle=':',
                                        linewidth=2,
                                        color='black',
                                        markersize=5,
                                        label=f'{block_type}',
                                        alpha=0.7
                                    )
                    
                    # Only add title for the first row (momentum model)
                    if i == 0:
                        ax.set_title(goal_titles[goal], fontsize=14, weight='bold')
                    
                    ax.set_ylim(0, 1.05)
                    ax.grid(True, alpha=0.3)
                    
                    # Only add xlabel for the last two rows (behavior and td_persistence)
                    if i >= 2:
                        ax.set_xlabel("Subgoal Progress", fontsize=12, weight='bold')
                    
                    # Only add ylabel to the first column
                    if j == 0:
                        ax.set_ylabel(f'{model_display_names[model_name]}\nSubgoal Selection Probability', fontsize=12, weight='bold')
                    
                    # Legend will be added separately after all plots

        # Format the collapsed overlay plot with grouped bar plots
        if collapse_over_subgoals:
            # Combine all model data
            combined_data = pd.concat(collapsed_data_all_models, ignore_index=True)
            
            # Reorder models so behavior is first (leftmost)
            models_ordered = []
            if 'behavior' in all_models:
                models_ordered.append('behavior')
            models_ordered.extend([m for m in all_models if m != 'behavior'])
            
            # Set up bar positions
            progress_values = sorted(combined_data['progress'].unique())
            n_models = len(models_ordered)
            
            # Calculate adaptive bar width based on minimum spacing between progress values
            if len(progress_values) > 1:
                # Calculate minimum spacing between adjacent progress values
                min_spacing = min([progress_values[i+1] - progress_values[i] 
                                   for i in range(len(progress_values)-1)])
                # Use 80% of minimum spacing divided by number of models to ensure no overlap
                bar_width = min_spacing * 0.8 / n_models
                # But don't make bars too small - set a minimum
                bar_width = max(bar_width, 0.02)
            else:
                bar_width = 0.05  # Fallback if only one progress value
            
            # Plot grouped bars
            for j, model_name in enumerate(models_ordered):
                model_display = model_display_names[model_name]
                model_data = combined_data[combined_data['model'] == model_display]
                
                # Calculate x positions for this model's bars
                # Center bars around each progress value
                offset = (j - (n_models - 1) / 2) * bar_width
                x_positions = [p + offset for p in progress_values]
                
                # Get values in order
                values = []
                errors = []
                for p in progress_values:
                    row = model_data[model_data['progress'] == p]
                    if not row.empty:
                        values.append(row['prob_mean'].values[0])
                        errors.append(row['prob_se'].values[0])
                    else:
                        values.append(0)
                        errors.append(0)
                
                # Plot bars with error bars
                ax.bar(
                    x_positions,
                    values,
                    bar_width,
                    yerr=errors,
                    label=model_display,
                    color=model_colors[model_name],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1,
                    capsize=5
                )
            
            ax.set_xlabel("Subgoal Progress", fontsize=14, weight='bold')
            ax.set_ylabel("Subgoal Selection Probability", fontsize=14, weight='bold')
            ax.set_xticks(progress_values)
            ax.set_xticklabels(progress_values)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(title='Model', fontsize=12, loc='best')
            ax.set_title('Subgoal Selection Probability by Progress (All Models)', fontsize=16, weight='bold')

        # Overall title
        if self.data_type == "online":
            experiment = "H2"
        elif self.data_type == "fmri":
            experiment = "H1"
        
        if not collapse_over_subgoals:
            fig.suptitle(f'Subgoal Selection Probability by Progress', fontsize=18, weight='bold', y=0.98)
            
            # Add legend in separate subplot
            if marker_dict and gs_main:
                # Create custom legend with markers
                legend_handles = []
                for block_type in sorted(marker_dict.keys()):
                    legend_handles.append(plt.Line2D([0], [0], 
                                                      marker=marker_dict[block_type], 
                                                      linestyle=':', 
                                                      color='black', 
                                                      linewidth=2, 
                                                      markersize=5, 
                                                      label=block_type))
                
                # Create legend subplot spanning all rows in the rightmost column
                legend_ax = fig.add_subplot(gs_main[:, 3])  # Span all rows in last column
                legend_ax.axis('off')
                legend_ax.legend(handles=legend_handles, loc='center left', title='Block Type', frameon=False)
        
        # Save the plot
        behavior_suffix = "_with_behavior" if include_behavior else ""
        collapse_suffix = "_collapsed" if collapse_over_subgoals else ""
        plt.savefig(self.figure_dir + f"/subgoal_selection_related_subgoal_progress_four_models_{self.experiment}{collapse_suffix}{behavior_suffix}.png", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_goal_switching_characteristics(self):
        """
        Characterize switches in goal selection patterns using pandas operations.
        """
        df = self.df.copy()
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create previous selections for comparison
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal'])
        
        # Define goal switching categories
        def classify_goal_switch(row):
            current_goal = row['goal_selected']
            prev_goal = row['prev_goal']
            
            if current_goal == prev_goal:
                return 0  # No switch
            else:
                return 1  # Switch
            
        df['switch_category'] = df.apply(classify_goal_switch, axis=1)
        
        # Calculate proportions for each subject and category
        subject_proportions = []
        subjects = df['subject_id'].unique()
        
        for subject in subjects:
            subject_data = df[df['subject_id'] == subject]
            total_switches = len(subject_data)
            
            if total_switches > 0:
                count_no_switch = len(subject_data[subject_data['switch_category'] == 0])
                count_switch = len(subject_data[subject_data['switch_category'] == 1])
                
                prop_no_switch = count_no_switch / total_switches
                prop_switch = count_switch / total_switches
                
                subject_proportions.append([prop_no_switch, prop_switch])
        
        # Convert to numpy array for easier manipulation
        data = np.array(subject_proportions)
        
        # Calculate means for each condition
        means = np.mean(data, axis=0)
        
        # Create the plot
        plt.figure(figsize=(8, 5))
        x = np.arange(1, len(means) + 1)
        
        # Plot individual lines and points for each subject
        for subject_data in data:
            plt.plot(x, subject_data, color='gray', alpha=0.5, linewidth=0.8, zorder=1)
            plt.scatter(x, subject_data, color='blue', alpha=0.6, s=10, zorder=2)
        
        # Overlay the means
        plt.scatter(x, means, color='red', label='Mean', zorder=5, s=50)
        plt.plot(x, means, color='red', linestyle='--', linewidth=1.5, zorder=4)
        
        # Formatting to match the style of other plots
        plt.xticks(x, ["No Switch", "Switch"], rotation=0, ha='center', fontsize=13, fontweight='bold')
        plt.yticks(fontsize=13)
        
        # Make y-tick labels bold
        for label in plt.gca().get_yticklabels():
            label.set_weight('bold')
        
        # Labels and styling
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'

        plt.xlabel("Switch Type", fontsize=15, fontweight='bold')
        plt.ylabel("Proportion", fontsize=15, fontweight='bold')
        plt.title(f"Goal switching patterns: Experiment {experiment}", 
                 fontsize=15, weight='bold')
        
        # Make legend bold
        legend = plt.legend(prop={'size': 13, 'weight': 'bold'})
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.figure_dir + f"/goal_switching_characteristics_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\nGoal switching characteristics summary:")
        print(f"Number of subjects: {len(subjects)}")
        print(f"Mean proportions by category:")
        for i, mean_prop in enumerate(means):
            print(f"  {i}: {mean_prop:.3f}")
        
        return data

    def plot_subgoal_switching_characteristics(self):
        """
        Characterize switches in subgoal selection patterns using pandas operations.
        """
        df = self.df.copy()
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create previous selections for comparison
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df['prev_subgoal'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal', 'prev_subgoal'])
        
        # Define subgoal switching categories
        def classify_subgoal_switch(row):
            current_goal = row['goal_selected']
            prev_goal = row['prev_goal']
            current_subgoal = row['subgoal_selected']
            prev_subgoal = row['prev_subgoal']
            
            if current_subgoal == prev_subgoal:
                return 0  # Staying with the subgoal
            elif current_goal == prev_goal:
                return 1  # Switching to the subgoal of the same goal
            else:
                # Need to determine if alternate goal is relevant or irrelevant
                # This would need goal relevance mapping - simplified for now
                # Could be category 2 (relevant) or 3 (irrelevant)
                return 2  # Switching to the subgoal of relevant alternate goal
            
        df['switch_category'] = df.apply(classify_subgoal_switch, axis=1)
        
        labels = [
            "Staying with the subgoal",
            "Switching to the subgoal of the same goal", 
            "Switching to the subgoal of relevant alternate goal",
            "Switching to the subgoal of irrelevant alternate goal"
        ]
        
        # Calculate proportions for each subject and category
        subject_proportions = []
        subjects = df['subject_id'].unique()
        
        for subject in subjects:
            subject_data = df[df['subject_id'] == subject]
            total_switches = len(subject_data)
            
            if total_switches > 0:
                proportions = []
                for category in range(len(labels)):
                    count = len(subject_data[subject_data['switch_category'] == category])
                    proportion = count / total_switches
                    proportions.append(proportion)
                subject_proportions.append(proportions)
        
        # Convert to numpy array for easier manipulation
        data = np.array(subject_proportions)
        
        # Calculate means for each condition
        means = np.mean(data, axis=0)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        x = np.arange(1, len(means) + 1)
        
        # Plot individual lines and points for each subject
        for subject_data in data:
            plt.plot(x, subject_data, color='gray', alpha=0.5, linewidth=0.8, zorder=1)
            plt.scatter(x, subject_data, color='blue', alpha=0.6, s=10, zorder=2)
        
        # Overlay the means
        plt.scatter(x, means, color='red', label='Mean', zorder=5, s=50)
        plt.plot(x, means, color='red', linestyle='--', linewidth=1.5, zorder=4)
        
        # Wrap the long labels
        wrapped_labels = [textwrap.fill(label, width=15) for label in labels]
        
        # Formatting to match the style of other plots
        plt.xticks(x, wrapped_labels, rotation=0, ha='center', fontsize=13, fontweight='bold')
        plt.yticks(fontsize=13)
        
        # Make y-tick labels bold
        for label in plt.gca().get_yticklabels():
            label.set_weight('bold')
        
        # Labels and styling
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
            
        plt.xlabel("Switch Categories", fontsize=15, fontweight='bold')
        plt.ylabel("Proportion", fontsize=15, fontweight='bold')
        plt.title(f"Subgoal switching patterns: Experiment {experiment}", 
                 fontsize=15, weight='bold')
        
        # Make legend bold
        legend = plt.legend(prop={'size': 13, 'weight': 'bold'})
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.figure_dir + f"/subgoal_switching_characteristics_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\nSubgoal switching characteristics summary:")
        print(f"Number of subjects: {len(subjects)}")
        print(f"Mean proportions by category:")
        for i, (label, mean_prop) in enumerate(zip(labels, means)):
            print(f"  {i}: {label}: {mean_prop:.3f}")
        
        return data, labels

    def plot_switching_characteristics(self, goal_type="subgoal"):
        """
        Characterize switches in goal/subgoal selection patterns using pandas operations.
        
        Parameters:
        goal_type (str): Type of analysis - "goal" or "subgoal"
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency in goal_selected
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create previous selections for comparison
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df['prev_subgoal'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        
        # Also convert previous goal from BR to OB for consistency
        df['prev_goal'] = df['prev_goal'].replace('BR', 'OB')
        
        # Create previous goal progress columns to check if previous goal was completed
        # We need to get the progress for the previous goal, so we'll create a function to handle this
        def get_prev_goal_progress(row):
            prev_goal = row['prev_goal']
            if pd.isna(prev_goal):
                return np.nan
            
            # Map OB back to BR for progress column naming
            goal_for_progress = 'BR' if prev_goal == 'OB' else prev_goal
            progress_col = f'{goal_for_progress}_progress_post'
            
            if progress_col in row:
                return row[progress_col]
            else:
                return np.nan
        
        df['prev_goal_progress'] = df.apply(get_prev_goal_progress, axis=1)
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal', 'prev_subgoal'])
        
        # Define shared mapping functions for both goal and subgoal analysis
        def get_relevant_alternative_subgoals_for_subgoals():
            """
            Get subgoals of other goals that go together with the subgoal of a given goal
            """
            subgoal_rel_subgoal = {
                ("SH", "G"): ["S", "C"],
                ("BR", "G"): ["M"],
                ("OB", "G"): ["M"],  # Add OB mapping same as BR
                ("BR", "C"): ["M", "S", "C"],
                ("OB", "C"): ["M", "S", "C"],  # Add OB mapping same as BR
                ("HO", "C"): ["S", "G"],
                ("SH", "M"): ["S", "C"],
                ("HO", "M"): ["G"],
                ("BR", "S"): ["C", "M"],
                ("OB", "S"): ["C", "M"],  # Add OB mapping same as BR
                ("HO", "S"): ["S", "C", "G"]
            }
            return subgoal_rel_subgoal
        
        def get_relevant_goals_for_subgoals():
            subgoal_rel_goal = {
                "G": ["SH", "BR", "OB"],  # Include OB to match BR
                "C": ["HO", "BR", "OB"],  # Include OB to match BR
                "M": ["SH", "HO"],
                "S": ["BR", "OB", "HO"]  # Include OB to match BR
            }
            return subgoal_rel_goal
        
        if goal_type == "subgoal":
            # Define subgoal switching categories
            def classify_subgoal_switch(row):
                current_goal = row['goal_selected']
                prev_goal = row['prev_goal']
                current_subgoal = row['subgoal_selected']
                prev_subgoal = row['prev_subgoal']
                goal_progress = row['goal_progress_pre'] if 'goal_progress_pre' in row else None
                
                if current_subgoal == prev_subgoal and current_goal == prev_goal:
                    return 0  # Staying with the same subgoal and goal
                elif current_subgoal == prev_subgoal and current_goal != prev_goal:
                    return 1  # Staying with the subgoal, but switching the goal
                elif current_subgoal != prev_subgoal and current_goal == prev_goal:
                    return 2  # Switching to the subgoal of the same goal
                else:
                    # Check if this is a relevant alternative subgoal switch
                    relevant_alt_subgoals = get_relevant_alternative_subgoals_for_subgoals()
                    
                    # Key for previous goal-subgoal combination
                    prev_key = (prev_goal, prev_subgoal)
                    
                    if prev_key in relevant_alt_subgoals:
                        relevant_subgoals = relevant_alt_subgoals[prev_key]
                        
                        if current_subgoal in relevant_subgoals:
                            # Additional check: subgoal progress must be < 1 for it to be truly relevant
                            if goal_progress and current_goal in goal_progress:
                                goal_prog = goal_progress[current_goal]
                                if current_subgoal in goal_prog:
                                    # Check if progress is less than maximum (not completed)
                                    current_progress = goal_prog[current_subgoal][0]
                                    max_progress = goal_prog[current_subgoal][1] 
                                    if current_progress < max_progress:
                                        return 3  # Relevant alternative subgoal
                                    else:
                                        return 4  # Irrelevant (subgoal already completed)
                                else:
                                    return 3  # Subgoal not in current goal progress, treat as relevant
                            else:
                                return 3  # No progress data, assume relevant
                        else:
                            # Check if previous goal was completed (progress = 6)
                            if 'prev_goal_progress' in row and not pd.isna(row['prev_goal_progress']) and row['prev_goal_progress'] == 6:
                                return -1  # Previous goal was completed
                            else:
                                return 4  # Irrelevant alternative subgoal
                    else:
                        # Check if previous goal was completed (progress = 6)
                        if 'prev_goal_progress' in row and not pd.isna(row['prev_goal_progress']) and row['prev_goal_progress'] == 6:
                            return -1  # Previous goal was completed
                        else:
                            return 4  # No mapping found, assume irrelevant
            
            df['switch_category'] = df.apply(classify_subgoal_switch, axis=1)
            
            labels = [
                "Staying with the same subgoal and goal",
                "Staying with the subgoal, but switching the goal",
                "Switching to the subgoal of the same goal", 
                "Switching to the subgoal of relevant alternate goal",
                "Switching to the subgoal of irrelevant alternate goal"
            ]


        elif goal_type == "goal":
            # Define goal switching categories with rigorous relevance logic
            def classify_goal_switch(row):
                current_goal = row['goal_selected']
                prev_goal = row['prev_goal']
                current_subgoal = row['subgoal_selected']
                prev_subgoal = row['prev_subgoal']
                #goal_progress = row['goal_progress_pre'] if 'goal_progress_pre' in row else None
                # Map goal for progress column naming (OB -> BR)
                # Handle NaN/float values in current_goal
                if pd.isna(current_goal) or current_goal is None:
                    goal_for_progress = None
                else:
                    goal_for_progress = 'BR' if str(current_goal) == 'OB' else str(current_goal)
                
                if goal_for_progress is not None:
                    progress_col = goal_for_progress + '_progress'
                    goal_progress = row[progress_col] if progress_col in row else None
                else:
                    goal_progress = None
                
                if current_goal == prev_goal and current_subgoal == prev_subgoal:
                    return 0  # Staying with the same goal and subgoal
                elif current_goal == prev_goal and current_subgoal != prev_subgoal:
                    # Check if the previous subgoal is terminated (progress = 1)
                    # Map goal for progress column naming (OB -> BR)
                    goal_for_progress = 'BR' if prev_goal == 'OB' else prev_goal
                    prev_subgoal_progress_col = f'{prev_subgoal}_progress_{goal_for_progress}'
                    
                    # If subgoal is terminated (progress = 1), create a separate category
                    if prev_subgoal_progress_col in row and row[prev_subgoal_progress_col] == 1:
                        print("Ola")
                        return 2  # Subgoal switch due to termination (goal stays same)
                    else:
                        return 1  # Staying with the goal and switching to a new subgoal
                elif current_goal != prev_goal and current_subgoal == prev_subgoal:
                    #print(current_goal, current_subgoal, prev_goal, prev_subgoal)
                    return -1  # Exclude: Staying with the subgoal, but switching the goal
                elif current_goal != prev_goal and current_subgoal != prev_subgoal:
                    # Determine if this is a relevant or irrelevant goal switch
                    # Use the subgoal-to-goal mapping to determine goal relevance
                    relevant_goals_for_subgoals = get_relevant_goals_for_subgoals()
                    
                    # Check if the current subgoal creates a connection between the goals
                    if current_subgoal in relevant_goals_for_subgoals:
                        relevant_goals = relevant_goals_for_subgoals[current_subgoal]
                        
                        # Both current and previous goals should be relevant to the current subgoal
                        if (current_goal in relevant_goals and prev_goal in relevant_goals):
                            # Additional check: subgoal progress must be < max for it to be truly relevant
                            # if goal_progress and current_goal in goal_progress:
                            #     goal_prog = goal_progress[current_goal]
                            #     if current_subgoal in goal_prog:
                            #         # Check if progress is less than maximum (not completed)
                            #         current_progress = goal_prog[current_subgoal][0]
                            #         max_progress = goal_prog[current_subgoal][1]
                            if goal_progress < 6:
                                return 3  # Relevant goal switch
                            else:
                                return -1  # Irrelevant (subgoal already completed)
                        else:
                            # Check if previous goal was completed (progress = 6)
                            if 'prev_goal_progress' in row and not pd.isna(row['prev_goal_progress']) and row['prev_goal_progress'] == 6:
                                return -1  # Previous goal was completed
                            else:
                                return 4  # Goals not connected through the current subgoal
            
            df['switch_category'] = df.apply(classify_goal_switch, axis=1)
            
            labels = [
                "Staying with goal and subgoal",
                "Staying with goal and switching subgoal",
                "Subgoal switch due to termination (goal stays same)",
                "Switching to subgoal of relevant alternate goal", 
                "Switching to subgoal of irrelevant alternate goal"
            ]
        
        # Create previous action outcome for filtering
        df['prev_action_outcome'] = df.groupby(['subject_id', 'block_num'])['action_outcome'].shift(1)
        
        # Remove rows where previous action outcome is NaN (first trial of each block)
        df_with_outcome = df.dropna(subset=['prev_action_outcome'])
        
        # Remove rows where switch_category is -1 (previous goal was completed)
        df_with_outcome = df_with_outcome[df_with_outcome['switch_category'] != -1]
        
        # Split data based on previous action outcome
        df_success = df_with_outcome[df_with_outcome['prev_action_outcome'] == 1].copy()
        df_failure = df_with_outcome[df_with_outcome['prev_action_outcome'] == 0].copy()
        
        def calculate_proportions_for_dataframe(data_df, outcome_type):
            """Helper function to calculate proportions for a given dataframe"""
            subject_proportions = []
            subjects = data_df['subject_id'].unique()
            
            # Debug: Print overall distribution of switch categories
            print(f"\nOverall switch category distribution for {goal_type} (Previous outcome = {outcome_type}):")
            category_counts = data_df['switch_category'].value_counts().sort_index()
            for cat, count in category_counts.items():
                if cat >= 0 and cat < len(labels):
                    print(f"  Category {cat} ({labels[int(cat)]}): {count} instances")
                elif cat == -1 or cat == -1.0:
                    print(f"  Category {cat} (Previous goal was completed): {count} instances")
            
            for subject in subjects:
                subject_data = data_df[data_df['subject_id'] == subject]
                # Exclude -1 values from the analysis
                subject_data = subject_data[subject_data['switch_category'] != -1]
                total_switches = len(subject_data)
                
                if total_switches > 0:
                    proportions = []
                    for category in range(len(labels)):
                        count = len(subject_data[subject_data['switch_category'] == category])
                        proportion = count / total_switches
                        proportions.append(proportion)
                    subject_proportions.append(proportions)
            
            return np.array(subject_proportions) if subject_proportions else np.array([])
        
        # Calculate proportions for both success and failure conditions
        data_success = calculate_proportions_for_dataframe(df_success, "Success")
        data_failure = calculate_proportions_for_dataframe(df_failure, "Failure")
        
        # Statistical testing between success and failure conditions
        from scipy import stats
        
        significance_results = []
        if len(data_success) > 0 and len(data_failure) > 0:
            for i in range(len(labels)):
                success_values = data_success[:, i]
                failure_values = data_failure[:, i]
                
                # Perform paired t-test
                try:
                    t_stat, p_value = stats.ttest_rel(success_values, failure_values)
                except:
                    # If paired test fails, use independent t-test
                    t_stat, p_value = stats.ttest_ind(success_values, failure_values)
                
                significance_results.append(p_value)
        else:
            significance_results = [1.0] * len(labels)  # No significance if no data
        
        # Additional t-test statistics for specific category combinations
        print(f"\n{goal_type.capitalize()} switching t-test statistics (Success vs Failure):")
        print("="*80)
        
        if len(data_success) > 0 and len(data_failure) > 0:
            # Calculate proportion of total goal switches (categories 3 + 4)
            total_goal_switches_success = data_success[:, 3] + data_success[:, 4]  # Categories 3 and 4
            total_goal_switches_failure = data_failure[:, 3] + data_failure[:, 4]   # Categories 3 and 4
            
            # Calculate proportion of subgoal switches (categories 2 + 3)
            subgoal_switches_success = data_success[:, 2] + data_success[:, 1]  # Categories 2 and 3
            subgoal_switches_failure = data_failure[:, 2] + data_failure[:, 1]   # Categories 2 and 3
            
            # Perform t-tests for total goal switches
            try:
                t_stat_goal, p_value_goal = stats.ttest_rel(total_goal_switches_success, total_goal_switches_failure)
                test_type_goal = "Paired t-test"
            except:
                t_stat_goal, p_value_goal = stats.ttest_ind(total_goal_switches_success, total_goal_switches_failure)
                test_type_goal = "Independent t-test"
            
            # Perform t-tests for subgoal switches  
            try:
                t_stat_subgoal, p_value_subgoal = stats.ttest_rel(subgoal_switches_success, subgoal_switches_failure)
                test_type_subgoal = "Paired t-test"
            except:
                t_stat_subgoal, p_value_subgoal = stats.ttest_ind(subgoal_switches_success, subgoal_switches_failure)
                test_type_subgoal = "Independent t-test"
            
            # Calculate degrees of freedom
            df_goal = len(total_goal_switches_success) - 1 if test_type_goal == "Paired t-test" else len(total_goal_switches_success) + len(total_goal_switches_failure) - 2
            df_subgoal = len(subgoal_switches_success) - 1 if test_type_subgoal == "Paired t-test" else len(subgoal_switches_success) + len(subgoal_switches_failure) - 2
            
            # Print results for total goal switches
            print(f"\n1. PROPORTION OF TOTAL GOAL SWITCHES (Categories 3 + 4):")
            print(f"   Previous outcome SUCCESS: M = {total_goal_switches_success.mean():.4f}, SD = {total_goal_switches_success.std():.4f}")
            print(f"   Previous outcome FAILURE: M = {total_goal_switches_failure.mean():.4f}, SD = {total_goal_switches_failure.std():.4f}")
            print(f"   {test_type_goal}: t({df_goal}) = {t_stat_goal:.4f}, p = {p_value_goal:.6f}")
            
            # Determine significance level for goal switches
            if p_value_goal < 0.001:
                sig_level_goal = "***"
            elif p_value_goal < 0.01:
                sig_level_goal = "**"
            elif p_value_goal < 0.05:
                sig_level_goal = "*"
            else:
                sig_level_goal = "ns"
            print(f"   Significance: {sig_level_goal}")
            
            # Calculate effect size (Cohen's d) for goal switches
            if test_type_goal == "Paired t-test":
                diff = total_goal_switches_success - total_goal_switches_failure
                cohens_d_goal = diff.mean() / diff.std()
            else:
                pooled_std = np.sqrt(((len(total_goal_switches_success) - 1) * total_goal_switches_success.var() + 
                                     (len(total_goal_switches_failure) - 1) * total_goal_switches_failure.var()) / 
                                    (len(total_goal_switches_success) + len(total_goal_switches_failure) - 2))
                cohens_d_goal = (total_goal_switches_success.mean() - total_goal_switches_failure.mean()) / pooled_std
            print(f"   Effect size (Cohen's d): {cohens_d_goal:.4f}")
            
            # Print results for subgoal switches
            print(f"\n2. PROPORTION OF SUBGOAL SWITCHES (Categories 2 + 3):")
            print(f"   Previous outcome SUCCESS: M = {subgoal_switches_success.mean():.4f}, SD = {subgoal_switches_success.std():.4f}")
            print(f"   Previous outcome FAILURE: M = {subgoal_switches_failure.mean():.4f}, SD = {subgoal_switches_failure.std():.4f}")
            print(f"   {test_type_subgoal}: t({df_subgoal}) = {t_stat_subgoal:.4f}, p = {p_value_subgoal:.6f}")
            
            # Determine significance level for subgoal switches
            if p_value_subgoal < 0.001:
                sig_level_subgoal = "***"
            elif p_value_subgoal < 0.01:
                sig_level_subgoal = "**"
            elif p_value_subgoal < 0.05:
                sig_level_subgoal = "*"
            else:
                sig_level_subgoal = "ns"
            print(f"   Significance: {sig_level_subgoal}")
            
            # Calculate effect size (Cohen's d) for subgoal switches
            if test_type_subgoal == "Paired t-test":
                diff = subgoal_switches_success - subgoal_switches_failure
                cohens_d_subgoal = diff.mean() / diff.std()
            else:
                pooled_std = np.sqrt(((len(subgoal_switches_success) - 1) * subgoal_switches_success.var() + 
                                     (len(subgoal_switches_failure) - 1) * subgoal_switches_failure.var()) / 
                                    (len(subgoal_switches_success) + len(subgoal_switches_failure) - 2))
                cohens_d_subgoal = (subgoal_switches_success.mean() - subgoal_switches_failure.mean()) / pooled_std
            print(f"   Effect size (Cohen's d): {cohens_d_subgoal:.4f}")
            
            # Additional t-test: Compare total goal switches vs subgoal switches within conditions
            print(f"\n3. COMPARISON: TOTAL GOAL SWITCHES vs SUBGOAL SWITCHES (Within Conditions):")
            
            # T-test for SUCCESS condition: total goal switches vs subgoal switches
            try:
                t_stat_success_comparison, p_value_success_comparison = stats.ttest_rel(total_goal_switches_success, subgoal_switches_success)
                test_type_success_comparison = "Paired t-test"
            except:
                t_stat_success_comparison, p_value_success_comparison = stats.ttest_ind(total_goal_switches_success, subgoal_switches_success)
                test_type_success_comparison = "Independent t-test"
            
            # Calculate degrees of freedom for success comparison
            df_success_comparison = len(total_goal_switches_success) - 1 if test_type_success_comparison == "Paired t-test" else len(total_goal_switches_success) + len(subgoal_switches_success) - 2
            
            print(f"\n   SUCCESS condition - Total Goal Switches vs Subgoal Switches:")
            print(f"   Total Goal Switches (3+4): M = {total_goal_switches_success.mean():.4f}, SD = {total_goal_switches_success.std():.4f}")
            print(f"   Subgoal Switches (2+1): M = {subgoal_switches_success.mean():.4f}, SD = {subgoal_switches_success.std():.4f}")
            print(f"   {test_type_success_comparison}: t({df_success_comparison}) = {t_stat_success_comparison:.4f}, p = {p_value_success_comparison:.6f}")
            
            # Determine significance level for success comparison
            if p_value_success_comparison < 0.001:
                sig_level_success_comparison = "***"
            elif p_value_success_comparison < 0.01:
                sig_level_success_comparison = "**"
            elif p_value_success_comparison < 0.05:
                sig_level_success_comparison = "*"
            else:
                sig_level_success_comparison = "ns"
            print(f"   Significance: {sig_level_success_comparison}")
            
            # Calculate effect size (Cohen's d) for success comparison
            if test_type_success_comparison == "Paired t-test":
                diff_success = total_goal_switches_success - subgoal_switches_success
                cohens_d_success_comparison = diff_success.mean() / diff_success.std()
            else:
                pooled_std_success = np.sqrt(((len(total_goal_switches_success) - 1) * total_goal_switches_success.var() + 
                                             (len(subgoal_switches_success) - 1) * subgoal_switches_success.var()) / 
                                            (len(total_goal_switches_success) + len(subgoal_switches_success) - 2))
                cohens_d_success_comparison = (total_goal_switches_success.mean() - subgoal_switches_success.mean()) / pooled_std_success
            print(f"   Effect size (Cohen's d): {cohens_d_success_comparison:.4f}")
            
            # T-test for FAILURE condition: total goal switches vs subgoal switches
            try:
                t_stat_failure_comparison, p_value_failure_comparison = stats.ttest_rel(total_goal_switches_failure, subgoal_switches_failure)
                test_type_failure_comparison = "Paired t-test"
            except:
                t_stat_failure_comparison, p_value_failure_comparison = stats.ttest_ind(total_goal_switches_failure, subgoal_switches_failure)
                test_type_failure_comparison = "Independent t-test"
            
            # Calculate degrees of freedom for failure comparison
            df_failure_comparison = len(total_goal_switches_failure) - 1 if test_type_failure_comparison == "Paired t-test" else len(total_goal_switches_failure) + len(subgoal_switches_failure) - 2
            
            print(f"\n   FAILURE condition - Total Goal Switches vs Subgoal Switches:")
            print(f"   Total Goal Switches (3+4): M = {total_goal_switches_failure.mean():.4f}, SD = {total_goal_switches_failure.std():.4f}")
            print(f"   Subgoal Switches (2+1): M = {subgoal_switches_failure.mean():.4f}, SD = {subgoal_switches_failure.std():.4f}")
            print(f"   {test_type_failure_comparison}: t({df_failure_comparison}) = {t_stat_failure_comparison:.4f}, p = {p_value_failure_comparison:.6f}")
            
            # Determine significance level for failure comparison
            if p_value_failure_comparison < 0.001:
                sig_level_failure_comparison = "***"
            elif p_value_failure_comparison < 0.01:
                sig_level_failure_comparison = "**"
            elif p_value_failure_comparison < 0.05:
                sig_level_failure_comparison = "*"
            else:
                sig_level_failure_comparison = "ns"
            print(f"   Significance: {sig_level_failure_comparison}")
            
            # Calculate effect size (Cohen's d) for failure comparison
            if test_type_failure_comparison == "Paired t-test":
                diff_failure = total_goal_switches_failure - subgoal_switches_failure
                cohens_d_failure_comparison = diff_failure.mean() / diff_failure.std()
            else:
                pooled_std_failure = np.sqrt(((len(total_goal_switches_failure) - 1) * total_goal_switches_failure.var() + 
                                             (len(subgoal_switches_failure) - 1) * subgoal_switches_failure.var()) / 
                                            (len(total_goal_switches_failure) + len(subgoal_switches_failure) - 2))
                cohens_d_failure_comparison = (total_goal_switches_failure.mean() - subgoal_switches_failure.mean()) / pooled_std_failure
            print(f"   Effect size (Cohen's d): {cohens_d_failure_comparison:.4f}")
            
            # Additional t-test: Compare total goal switches vs subgoal switches COLLAPSED over all conditions
            print(f"\n4. COMPARISON: TOTAL GOAL SWITCHES vs SUBGOAL SWITCHES (Collapsed over ALL conditions):")
            
            # Combine data from both success and failure conditions
            total_goal_switches_combined = np.concatenate([total_goal_switches_success, total_goal_switches_failure])
            subgoal_switches_combined = np.concatenate([subgoal_switches_success, subgoal_switches_failure])
            
            # T-test for COMBINED conditions: total goal switches vs subgoal switches
            try:
                t_stat_combined_comparison, p_value_combined_comparison = stats.ttest_rel(total_goal_switches_combined, subgoal_switches_combined)
                test_type_combined_comparison = "Paired t-test"
            except:
                t_stat_combined_comparison, p_value_combined_comparison = stats.ttest_ind(total_goal_switches_combined, subgoal_switches_combined)
                test_type_combined_comparison = "Independent t-test"
            
            # Calculate degrees of freedom for combined comparison
            df_combined_comparison = len(total_goal_switches_combined) - 1 if test_type_combined_comparison == "Paired t-test" else len(total_goal_switches_combined) + len(subgoal_switches_combined) - 2
            
            print(f"\n   ALL conditions combined - Total Goal Switches vs Subgoal Switches:")
            print(f"   Total Goal Switches (3+4): M = {total_goal_switches_combined.mean():.4f}, SD = {total_goal_switches_combined.std():.4f}, N = {len(total_goal_switches_combined)}")
            print(f"   Subgoal Switches (2+1): M = {subgoal_switches_combined.mean():.4f}, SD = {subgoal_switches_combined.std():.4f}, N = {len(subgoal_switches_combined)}")
            print(f"   {test_type_combined_comparison}: t({df_combined_comparison}) = {t_stat_combined_comparison:.4f}, p = {p_value_combined_comparison:.6f}")
            
            # Determine significance level for combined comparison
            if p_value_combined_comparison < 0.001:
                sig_level_combined_comparison = "***"
            elif p_value_combined_comparison < 0.01:
                sig_level_combined_comparison = "**"
            elif p_value_combined_comparison < 0.05:
                sig_level_combined_comparison = "*"
            else:
                sig_level_combined_comparison = "ns"
            print(f"   Significance: {sig_level_combined_comparison}")
            
            # Calculate effect size (Cohen's d) for combined comparison
            if test_type_combined_comparison == "Paired t-test":
                diff_combined = total_goal_switches_combined - subgoal_switches_combined
                cohens_d_combined_comparison = diff_combined.mean() / diff_combined.std()
            else:
                pooled_std_combined = np.sqrt(((len(total_goal_switches_combined) - 1) * total_goal_switches_combined.var() + 
                                              (len(subgoal_switches_combined) - 1) * subgoal_switches_combined.var()) / 
                                             (len(total_goal_switches_combined) + len(subgoal_switches_combined) - 2))
                cohens_d_combined_comparison = (total_goal_switches_combined.mean() - subgoal_switches_combined.mean()) / pooled_std_combined
            print(f"   Effect size (Cohen's d): {cohens_d_combined_comparison:.4f}")
            
            print("="*80)
            
        else:
            print("Insufficient data for t-test analysis")
            print("="*80)
        
        # Create the first plot: collapsed over all action outcomes (line plot)
        # Combine data from both success and failure conditions
        if len(data_success) > 0 and len(data_failure) > 0:
            # Calculate proportions for combined data (collapsed over outcomes)
            df_combined = df_with_outcome.copy()
            data_combined = calculate_proportions_for_dataframe(df_combined, "All outcomes")
        elif len(data_success) > 0:
            data_combined = data_success
        elif len(data_failure) > 0:
            data_combined = data_failure
        else:
            data_combined = np.array([])
        
        if len(data_combined) > 0:
            fig1 = plt.figure(figsize=(10, 7))
            ax1 = fig1.add_subplot(1, 1, 1)
            
            # Wrap the long labels
            wrapped_labels = [textwrap.fill(label, width=15) for label in labels]
            
            # Calculate means for combined data
            means_combined = np.mean(data_combined, axis=0)
            
            # Create the plot
            x = np.arange(1, len(means_combined) + 1)
            
            # Plot individual lines and points for each subject
            for subject_data in data_combined:
                ax1.plot(x, subject_data, color='gray', alpha=0.3, linewidth=1, zorder=1)
                ax1.scatter(x, subject_data, color='gray', alpha=0.3, s=30, zorder=2)
            
            # Overlay the means
            ax1.scatter(x, means_combined, color='red', label='Mean', zorder=5, s=80)
            ax1.plot(x, means_combined, color='red', linestyle='-', linewidth=3, zorder=4)
            
            # Formatting
            ax1.set_xticks(x + 0.3)
            ax1.set_xticklabels(wrapped_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='y', labelsize=13)
            
            # Make y-tick labels bold
            for label in ax1.get_yticklabels():
                label.set_weight('bold')
            
            # Labels and styling
            if self.data_type == "online":
                experiment = 'H2'
            elif self.data_type == "fmri":
                experiment = 'H1'
            
            ax1.set_xlabel("Switch Categories", fontsize=15, fontweight='bold')
            ax1.set_ylabel("Proportion", fontsize=15, fontweight='bold')
            ax1.set_title(f"{goal_type.capitalize()} switching patterns", 
                         fontsize=15, weight='bold')
            
            # Make legend bold
            legend = ax1.legend(prop={'size': 13, 'weight': 'bold'})
            
            ax1.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save and show first figure
            plt.savefig(self.figure_dir + f"/switching_characteristics_{goal_type}_combined_experiment_{self.experiment}_lineplot.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        # Create the second plot with side-by-side bar plots (success/failure comparison)
        if len(data_success) > 0 and len(data_failure) > 0:
            fig2 = plt.figure(figsize=(11, 8))
            ax2 = fig2.add_subplot(1, 1, 1)
            
            # Wrap the long labels
            wrapped_labels = [textwrap.fill(label, width=15) for label in labels]
            
            means_success = np.mean(data_success, axis=0)
            means_failure = np.mean(data_failure, axis=0)
            
            # Calculate standard errors
            se_success = np.std(data_success, axis=0) / np.sqrt(len(data_success))
            se_failure = np.std(data_failure, axis=0) / np.sqrt(len(data_failure))
            
            # Set up bar positions
            x = np.arange(len(labels))
            width = 0.35  # Width of the bars
            
            # Create side-by-side bars
            bars1 = ax2.bar(x - width/2, means_success, width, yerr=se_success, 
                          label='Success', color='white', alpha=0.8, 
                          edgecolor='black', linewidth=1, capsize=5)
            bars2 = ax2.bar(x + width/2, means_failure, width, yerr=se_failure,
                          label='Failure', color='black', alpha=0.8, 
                          edgecolor='black', linewidth=1, capsize=5)
            
            # # Add significance markers above the bars
            y_max = max(max(means_success + se_success), max(means_failure + se_failure))
            # for i, p_val in enumerate(significance_results):
            #     if p_val < 0.001:
            #         marker = '***'
            #     elif p_val < 0.01:
            #         marker = '**'
            #     elif p_val < 0.05:
            #         marker = '*'
            #     else:
            #         marker = 'ns'
                
            #     # Position marker above the higher bar
            #     bar_height = max(means_success[i] + se_success[i], means_failure[i] + se_failure[i])
            #     y_pos = bar_height + y_max * 0.02
                
            #     # Add horizontal line connecting the bars
            #     line_y = bar_height + y_max * 0.01
            #     ax2.plot([x[i] - width/2, x[i] + width/2], [line_y, line_y], 'k-', linewidth=1)
                
            #     # Add significance marker
            #     ax2.text(x[i], y_pos, marker, ha='center', va='bottom', 
            #            fontsize=14, fontweight='bold', color='blue')
            
            # Formatting
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(wrapped_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='y', labelsize=14)
            
            # Make y-tick labels bold
            for label in ax2.get_yticklabels():
                label.set_weight('bold')
            
            ax2.set_xlabel("Switch Categories", fontsize=16, fontweight='bold')
            ax2.set_ylabel("Proportion", fontsize=16, fontweight='bold')
            
            # Labels and styling
            if self.data_type == "online":
                experiment = 'H2'
            elif self.data_type == "fmri":
                experiment = 'H1'
            
            ax2.set_title(f"{goal_type.capitalize()} Switching Characteristics by Previous Outcome", 
                        fontsize=16, weight='bold', pad=20)
            
            # Make legend bold and position it better
            legend = ax2.legend(title='Previous Action Outcome', loc='upper right', 
                             prop={'size': 14, 'weight': 'bold'})
            legend.get_title().set_fontweight('bold')
            legend.get_title().set_fontsize(14)
            
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim(0, y_max * 1.15)  # Add some space at the top for significance markers
            
            plt.tight_layout()
            
            # Save and show second figure
            plt.savefig(self.figure_dir + f"/switching_characteristics_{goal_type}_by_outcome_experiment_{self.experiment}_barplot.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Insufficient data for success/failure comparison plot")
        
        # Create the second plot with depth-first analysis (four subplots in 2x2 grid)
        if goal_type == "goal":
            fig2 = plt.figure(figsize=(16, 12))
            ax3 = fig2.add_subplot(2, 2, 1)  # Top left
            ax4 = fig2.add_subplot(2, 2, 2)  # Top right
            ax5 = fig2.add_subplot(2, 2, 3)  # Bottom left
            ax6 = fig2.add_subplot(2, 2, 4)  # Bottom right
        
        # Calculate depth-first metric for each participant (combined across conditions)
        if goal_type == "goal":
            # Combine data from both success and failure conditions
            if len(data_success) > 0 and len(data_failure) > 0:
                # Calculate depth-first metric: difference between staying with same goal+subgoal (category 0) 
                # and staying with same goal but switching subgoal (category 1)
                # Average across both conditions for each participant
                #depth_first_combined = (data_success[:, 0] - data_success[:, 1] + data_failure[:, 0] - data_failure[:, 1]) / (data_success[:, 0] + data_failure[:, 0] + data_success[:, 1] + data_failure[:, 1])
                depth_first_combined = (data_success[:, 0]  + data_failure[:, 0] ) / (data_success[:, 0] + data_failure[:, 0] + data_success[:, 1] + data_failure[:, 1])
                
                # Calculate total number of goal switches for each participant from original dataframe
                # Count all non-staying categories (categories 1, 2, 3, 4) from the original dataframe
                total_switches_combined = []
                subjects = df_with_outcome['subject_id'].unique()
                
                for subject in subjects:
                    subject_data = df_with_outcome[df_with_outcome['subject_id'] == subject]
                    # Count all switch categories (exclude category 0 which is staying)
                    switch_categories = subject_data[subject_data['switch_category'].isin([3, 4])]
                    total_switches = len(switch_categories)
                    total_switches_combined.append(total_switches)
                
                total_switches_combined = np.array(total_switches_combined)
                
                # Plot depth-first metric vs total switches (all participants together)
                ax3.scatter(depth_first_combined, total_switches_combined, color='black', marker='o', alpha=0.7, s=60, zorder=3)
                
                # Add correlation analysis
                from scipy.stats import pearsonr
                
                # Remove any NaN values
                valid_mask = ~(np.isnan(depth_first_combined) | np.isnan(total_switches_combined))
                if np.sum(valid_mask) > 1:
                    corr_coef, p_value = pearsonr(depth_first_combined[valid_mask], total_switches_combined[valid_mask])
                    
                    # Add correlation line
                    z = np.polyfit(depth_first_combined[valid_mask], total_switches_combined[valid_mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(depth_first_combined[valid_mask].min(), depth_first_combined[valid_mask].max(), 100)
                    ax3.plot(x_line, p(x_line), 'k--', alpha=0.8, linewidth=2, zorder=2)
                    
                    # Add correlation text with scientific notation for p-value
                    if p_value < 0.001:
                        p_text = f'p = {p_value:.2e}'
                    else:
                        p_text = f'p = {p_value:.3f}'
                    
                    ax3.text(0.95, 0.95, f'r = {corr_coef:.3f}\n{p_text}', 
                            transform=ax3.transAxes, fontsize=16, weight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            ha='right', va='top')
            
            # Formatting for the third subplot
            ax3.set_xlabel("Depth-First Metric", 
                          fontsize=15, weight='bold')
            ax3.set_ylabel("Total Goal Switches", fontsize=15, weight='bold')
            ax3.set_title("A. Depth-First Metric vs Total Goal Switches", fontsize=15, weight='bold')
            ax3.grid(alpha=0.3)
            
            # Make tick labels bold
            for label in ax3.get_xticklabels():
                label.set_weight('bold')
            for label in ax3.get_yticklabels():
                label.set_weight('bold')

            # increase x tick labels and y tick labels font size
            for label in ax3.get_xticklabels():
                label.set_fontsize(14)
            for label in ax3.get_yticklabels():
                label.set_fontsize(14)

            
            # Fourth subplot: Depth-first metric vs (relevant - irrelevant) goal switches
            if len(data_success) > 0 and len(data_failure) > 0:
                # Calculate the actual number of irrelevant and relevant switches for each participant
                # Get the raw counts from the original data
                subjects = df_with_outcome['subject_id'].unique()
                irrelevant_minus_relevant_counts = []
                
                for subject in subjects:
                    subject_data = df_with_outcome[df_with_outcome['subject_id'] == subject]
                    # Count actual number of switches (not proportions)
                    relevant_count = len(subject_data[subject_data['switch_category'] == 2])
                    irrelevant_count = len(subject_data[subject_data['switch_category'] == 3])
                    irrelevant_minus_relevant_counts.append(irrelevant_count / (irrelevant_count + relevant_count))
                
                irrelevant_minus_relevant_combined = np.array(irrelevant_minus_relevant_counts)
                
                # Plot depth-first metric vs (irrelevant - relevant) switches
                ax4.scatter(depth_first_combined, irrelevant_minus_relevant_combined, color='black', marker='s', alpha=0.7, s=60, zorder=3)
                
                # Add correlation analysis for fourth subplot
                valid_mask_4 = ~(np.isnan(depth_first_combined) | np.isnan(irrelevant_minus_relevant_combined))
                if np.sum(valid_mask_4) > 1:
                    corr_coef_4, p_value_4 = pearsonr(depth_first_combined[valid_mask_4], irrelevant_minus_relevant_combined[valid_mask_4])
                    
                    # Add correlation line
                    z_4 = np.polyfit(depth_first_combined[valid_mask_4], irrelevant_minus_relevant_combined[valid_mask_4], 1)
                    p_4 = np.poly1d(z_4)
                    x_line_4 = np.linspace(depth_first_combined[valid_mask_4].min(), depth_first_combined[valid_mask_4].max(), 100)
                    ax4.plot(x_line_4, p_4(x_line_4), 'k--', alpha=0.8, linewidth=2, zorder=2)
                    
                    # Add correlation text with scientific notation for p-value
                    if p_value_4 < 0.001:
                        p_text_4 = f'p = {p_value_4:.2e}'
                    else:
                        p_text_4 = f'p = {p_value_4:.3f}'
                    
                    ax4.text(0.95, 0.95, f'r = {corr_coef_4:.3f}\n{p_text_4}', 
                            transform=ax4.transAxes, fontsize=16, weight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            ha='right', va='top')
                
                # Formatting for the fourth subplot
                ax4.set_xlabel("Depth-First Metric", 
                              fontsize=15, weight='bold')
                ax4.set_ylabel("Irrelevant Alt-Goal Switch Proportion", fontsize=15, weight='bold')
                ax4.set_title("B. Depth-First Metric vs Irrelevant Alt-Goal Switch Proportion", fontsize=15, weight='bold')
                ax4.grid(alpha=0.3)
                
                # Make tick labels bold
                for label in ax4.get_xticklabels():
                    label.set_weight('bold')
                for label in ax4.get_yticklabels():
                    label.set_weight('bold')
            
                # increase x tick labels and y tick labels font size
                for label in ax4.get_xticklabels():
                    label.set_fontsize(14)
                for label in ax4.get_yticklabels():
                    label.set_fontsize(14)
            # Fifth subplot: Depth-first metric vs Task Performance
            # Calculate task performance (number of goals completed) for each participant
            subjects = df_with_outcome['subject_id'].unique()
            task_performance = []
            
            for subject in subjects:
                subject_data = df_with_outcome[df_with_outcome['subject_id'] == subject]
                goals_completed = 0
                
                # Count actual goal completion events by checking progress changes
                for goal in ['SH', 'BR', 'HO']:
                    progress_col = f'{goal}_progress'
                    progress_post_col = f'{goal}_progress_post'
                    
                    if progress_col in subject_data.columns and progress_post_col in subject_data.columns:
                        # Count transitions from < 6 to 6 (goal completion events)
                        for i in range(len(subject_data) - 1):
                            pre_progress = subject_data.iloc[i][progress_col]
                            post_progress = subject_data.iloc[i][progress_post_col]
                            
                            # Check if goal was completed in this trial (progress went from < 6 to = 6)
                            if pre_progress < 6.0 and post_progress >= 6.0:
                                goals_completed += 1
                
                task_performance.append(goals_completed)
            
            task_performance = np.array(task_performance)
            
            # Plot depth-first metric vs task performance
            ax5.scatter(depth_first_combined, task_performance, color='black', marker='^', alpha=0.7, s=60, zorder=3)
            
            # Add correlation analysis for fifth subplot
            valid_mask_5 = ~(np.isnan(depth_first_combined) | np.isnan(task_performance))
            if np.sum(valid_mask_5) > 1:
                corr_coef_5, p_value_5 = pearsonr(depth_first_combined[valid_mask_5], task_performance[valid_mask_5])
                
                # Add correlation line
                z_5 = np.polyfit(depth_first_combined[valid_mask_5], task_performance[valid_mask_5], 1)
                p_5 = np.poly1d(z_5)
                x_line_5 = np.linspace(depth_first_combined[valid_mask_5].min(), depth_first_combined[valid_mask_5].max(), 100)
                ax5.plot(x_line_5, p_5(x_line_5), 'k--', alpha=0.8, linewidth=2, zorder=2)
                
                # Add correlation text with scientific notation for p-value
                if p_value_5 < 0.001:
                    p_text_5 = f'p = {p_value_5:.2e}'
                else:
                    p_text_5 = f'p = {p_value_5:.3f}'
                
                ax5.text(0.95, 0.95, f'r = {corr_coef_5:.3f}\n{p_text_5}', 
                        transform=ax5.transAxes, fontsize=16, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        ha='right', va='top')
            
            # Formatting for the fifth subplot
            ax5.set_xlabel("Depth-First Metric", 
                          fontsize=15, weight='bold')
            ax5.set_ylabel("Number of Goals\nCompleted", fontsize=15, weight='bold')
            ax5.set_title("C. Depth-First Metric vs Task Performance", fontsize=15, weight='bold')
            ax5.grid(alpha=0.3)
            
            # Make tick labels bold
            for label in ax5.get_xticklabels():
                label.set_weight('bold')
            for label in ax5.get_yticklabels():
                label.set_weight('bold')
            # increase x tick labels and y tick labels font size
            for label in ax5.get_xticklabels():
                label.set_fontsize(14)
            for label in ax5.get_yticklabels():
                label.set_fontsize(14)
            
            # Sixth subplot: Depth-first metric vs Goal-Action Congruence
            # Calculate goal-action congruence for each participant
            goal_action_congruence = []
            
            for subject in subjects:
                subject_data = df_with_outcome[df_with_outcome['subject_id'] == subject]
                total_actions = 0
                congruent_actions = 0
                
                for _, trial in subject_data.iterrows():
                    if pd.isna(trial['goal_selected']) or pd.isna(trial['subgoal_selected']):
                        continue
                    
                    goal_selected = trial['goal_selected']
                    action_selected = trial['subgoal_selected']
                    
                    # Reconstruct goal_progress dictionary from individual columns
                    goal_progress = {
                        'SH': {
                            'G': [trial.get('G_progress_SH', 0) * 3, 3],  # [current, max]
                            'M': [trial.get('M_progress_SH', 0) * 3, 3],
                            'C': [0, 0],  # Not relevant for SH
                            'S': [0, 0]   # Not relevant for SH
                        },
                        'BR': {
                            'G': [trial.get('G_progress_BR', 0) * 2, 2],  # [current, max]
                            'S': [trial.get('S_progress_BR', 0) * 3, 3],
                            'C': [trial.get('C_progress_BR', 0) * 1, 1],
                            'M': [0, 0]   # Not relevant for BR
                        },
                        'HO': {
                            'S': [trial.get('S_progress_HO', 0) * 1, 1],  # [current, max]
                            'M': [trial.get('M_progress_HO', 0) * 2, 2],
                            'C': [trial.get('C_progress_HO', 0) * 3, 3],
                            'G': [0, 0]   # Not relevant for HO
                        }
                    }
                    
                    # Check if action is relevant to the goal using the same logic as in measures.py
                    relevance = 0
                    if goal_selected == "SH":
                        if action_selected == "S" or action_selected == "C":
                            relevance = 0
                        elif action_selected == "M":
                            if goal_progress[goal_selected]["M"][0] < 3:
                                relevance = 1
                            else:
                                relevance = 0
                        elif action_selected == "G":
                            if goal_progress[goal_selected]["G"][0] < 3:
                                relevance = 1
                            else:
                                relevance = 0
                    elif goal_selected == "HO":
                        if action_selected == "G":
                            relevance = 0
                        elif action_selected == "S":
                            if goal_progress[goal_selected]["S"][0] < 1:
                                relevance = 1
                            else:
                                relevance = 0
                        elif action_selected == "M":
                            if goal_progress[goal_selected]["M"][0] < 2:
                                relevance = 1
                            else:
                                relevance = 0
                        elif action_selected == "C":
                            if goal_progress[goal_selected]["C"][0] < 3:
                                relevance = 1
                            else:
                                relevance = 0
                    elif goal_selected == "BR":
                        if action_selected == "M":
                            relevance = 0
                        elif action_selected == "S":
                            if goal_progress[goal_selected]["S"][0] < 3:
                                relevance = 1
                            else:
                                relevance = 0
                        elif action_selected == "G":
                            if goal_progress[goal_selected]["G"][0] < 2:
                                relevance = 1
                            else:
                                relevance = 0
                        elif action_selected == "C":
                            if goal_progress[goal_selected]["C"][0] < 1:
                                relevance = 1
                            else:
                                relevance = 0
                    
                    total_actions += 1
                    if relevance == 1:
                        congruent_actions += 1
                
                # Calculate congruence proportion
                if total_actions > 0:
                    congruence_prop = congruent_actions / total_actions
                else:
                    congruence_prop = 0
                
                goal_action_congruence.append(congruence_prop)
            
            goal_action_congruence = np.array(goal_action_congruence)
            
            # Plot depth-first metric vs goal-action congruence
            ax6.scatter(depth_first_combined, goal_action_congruence, color='black', marker='D', alpha=0.7, s=60, zorder=3)
            
            # Add correlation analysis for sixth subplot
            valid_mask_6 = ~(np.isnan(depth_first_combined) | np.isnan(goal_action_congruence))
            if np.sum(valid_mask_6) > 1:
                corr_coef_6, p_value_6 = pearsonr(depth_first_combined[valid_mask_6], goal_action_congruence[valid_mask_6])
                
                # Add correlation line
                z_6 = np.polyfit(depth_first_combined[valid_mask_6], goal_action_congruence[valid_mask_6], 1)
                p_6 = np.poly1d(z_6)
                x_line_6 = np.linspace(depth_first_combined[valid_mask_6].min(), depth_first_combined[valid_mask_6].max(), 100)
                ax6.plot(x_line_6, p_6(x_line_6), 'k--', alpha=0.8, linewidth=2, zorder=2)
                
                # Add correlation text with scientific notation for p-value
                if p_value_6 < 0.001:
                    p_text_6 = f'p = {p_value_6:.2e}'
                else:
                    p_text_6 = f'p = {p_value_6:.3f}'
                
                ax6.text(0.95, 0.95, f'r = {corr_coef_6:.3f}\n{p_text_6}', 
                        transform=ax6.transAxes, fontsize=16, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        ha='right', va='top')
            
            # Formatting for the sixth subplot
            ax6.set_xlabel("Depth-First Metric", 
                          fontsize=15, weight='bold')
            ax6.set_ylabel("Goal-Action\nCongruence", fontsize=15, weight='bold')
            ax6.set_title("D. Depth-First Metric vs Goal-Action Congruence", fontsize=15, weight='bold')
            ax6.grid(alpha=0.3)
            
            # Make tick labels bold
            for label in ax6.get_xticklabels():
                label.set_weight('bold')
            for label in ax6.get_yticklabels():
                label.set_weight('bold')
            
            # increase x tick labels and y tick labels font size
            for label in ax6.get_xticklabels():
                label.set_fontsize(14)
            for label in ax6.get_yticklabels():
                label.set_fontsize(14)
            # Add spacing between subplots
            plt.tight_layout(pad=2.0, w_pad=8.0, h_pad=8.0)
            
            # Save and show second figure
            plt.savefig(self.figure_dir + f"/depth_first_analysis_{goal_type}_experiment_{self.experiment}.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Create a new separate plot: Depth-first metric vs Goal RT increase proportion
            fig3, ax7 = plt.subplots(1, 1, figsize=(8, 6))
            
            # Calculate goal RT increase proportion for each subject
            rt_increase_proportions = []
            subjects = df_with_outcome['subject_id'].unique()
            
            for subject in subjects:
                subject_data = df_with_outcome[df_with_outcome['subject_id'] == subject]
                
                # Separate goal switches vs stays
                goal_switches = subject_data[subject_data['switch_category'].isin([3, 4])]  # Goal switches
                goal_stays = subject_data[subject_data['switch_category'].isin([0, 1, 2])]  # Goal stays (including subgoal switches)
                
                if len(goal_switches) > 0 and len(goal_stays) > 0 and 'goal_rt' in subject_data.columns:
                    # Calculate mean RT for switches vs stays
                    mean_rt_switch = goal_switches['goal_rt'].mean()
                    mean_rt_stay = goal_stays['goal_rt'].mean()
                    
                    # Calculate RT increase proportion
                    if mean_rt_stay > 0:
                        rt_increase_proportion = (mean_rt_switch - mean_rt_stay) / mean_rt_stay
                        rt_increase_proportions.append(rt_increase_proportion)
                    else:
                        rt_increase_proportions.append(np.nan)
                else:
                    rt_increase_proportions.append(np.nan)
            
            rt_increase_proportions = np.array(rt_increase_proportions)
            
            # Plot depth-first metric vs RT increase proportion
            ax7.scatter(depth_first_combined, rt_increase_proportions, color='black', marker='x', alpha=0.7, s=60, zorder=3)
            
            # Add correlation analysis
            valid_mask_rt = ~(np.isnan(depth_first_combined) | np.isnan(rt_increase_proportions))
            if np.sum(valid_mask_rt) > 1:
                corr_coef_rt, p_value_rt = pearsonr(depth_first_combined[valid_mask_rt], rt_increase_proportions[valid_mask_rt])
                
                # Add correlation line
                z_rt = np.polyfit(depth_first_combined[valid_mask_rt], rt_increase_proportions[valid_mask_rt], 1)
                p_rt = np.poly1d(z_rt)
                x_line_rt = np.linspace(depth_first_combined[valid_mask_rt].min(), depth_first_combined[valid_mask_rt].max(), 100)
                ax7.plot(x_line_rt, p_rt(x_line_rt), 'k--', alpha=0.8, linewidth=2, zorder=2)
                
                # Add correlation text
                ax7.text(0.95, 0.95, f'r = {corr_coef_rt:.3f}\np = {p_value_rt:.3f}', 
                        transform=ax7.transAxes, fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        ha='right', va='top')
            
            # Formatting
            ax7.set_xlabel("Depth-First Metric", 
                          fontsize=15, weight='bold')
            ax7.set_ylabel("Goal RT Increase Proportion\n(Switch RT - Stay RT) / Stay RT", 
                          fontsize=15, weight='bold')
            ax7.set_title("Depth-First Metric vs Goal RT Increase Proportion", fontsize=16, weight='bold')
            ax7.grid(alpha=0.3)
            
            # Make tick labels bold
            for label in ax7.get_xticklabels():
                label.set_weight('bold')
            for label in ax7.get_yticklabels():
                label.set_weight('bold')
            # increase x tick labels and y tick labels font size
            for label in ax7.get_xticklabels():
                label.set_fontsize(14)
            for label in ax7.get_yticklabels():
                label.set_fontsize(14)
            
            plt.tight_layout()
            
            # Save and show the new figure
            plt.savefig(self.figure_dir + f"/depth_first_vs_rt_increase_experiment_{self.experiment}.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        # Print summary statistics for both conditions
        print(f"\n{goal_type.capitalize()} switching characteristics summary:")
        
        if len(data_success) > 0:
            print(f"\nPrevious Action Outcome = Success:")
            print(f"Number of subjects: {len(data_success)}")
            means_success = np.mean(data_success, axis=0)
            print(f"Mean proportions by category:")
            for i, (label, mean) in enumerate(zip(labels, means_success)):
                print(f"  {label}: {mean:.3f}")
            
            # Calculate and print standard errors
            if len(data_success) > 1:
                std_errors_success = np.std(data_success, axis=0) / np.sqrt(len(data_success))
                print(f"\nStandard errors by category:")
                for i, (label, se) in enumerate(zip(labels, std_errors_success)):
                    print(f"  {label}: {se:.3f}")
        
        if len(data_failure) > 0:
            print(f"\nPrevious Action Outcome = Failure:")
            print(f"Number of subjects: {len(data_failure)}")
            means_failure = np.mean(data_failure, axis=0)
            print(f"Mean proportions by category:")
            for i, (label, mean) in enumerate(zip(labels, means_failure)):
                print(f"  {label}: {mean:.3f}")
            
            # Calculate and print standard errors
            if len(data_failure) > 1:
                std_errors_failure = np.std(data_failure, axis=0) / np.sqrt(len(data_failure))
                print(f"\nStandard errors by category:")
                for i, (label, se) in enumerate(zip(labels, std_errors_failure)):
                    print(f"  {label}: {se:.3f}")
        
        return data_success, data_failure, labels


    def plot_retrospective_bias_subgoals(self):
        pros_optimal_csv = CACHE_DIR + "all_subjects_model_parameters_prospective_" + self.data_type + "_optimal_True.csv"
        df_pros = pd.read_csv(pros_optimal_csv)
        
        # Replace 'BR' with 'OB' for consistency
        df_pros['goal_selected'] = df_pros['goal_selected'].replace('BR', 'OB')
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }
        df_pros['block_type'] = df_pros['block_type'].replace(block_labels)
        
        # Create a copy to work with
        df = df_pros.copy()
        
        # Step 1: Assign prospective option (highest value among G_value, S_value, C_value, M_value)
        value_cols = ['G_value', 'S_value', 'C_value', 'M_value']
        
        # Check which value columns exist and add missing ones as NaN
        for col in value_cols:
            if col not in df.columns:
                df[col] = np.nan
                print(f"Warning: {col} column not found in data, filling with NaN")
        
        # Check if any value columns have non-NaN values
        has_valid_values = df[value_cols].notna().any().any()
        if not has_valid_values:
            print("Error: All value columns (G_value, S_value, C_value, M_value) are missing or contain only NaN values.")
            print("Cannot proceed with prospective option assignment.")
            return None
        
        # Handle NaN values by only considering rows where at least one value column is not NaN
        # Filter to rows that have at least one non-NaN value column
        valid_rows = df[value_cols].notna().any(axis=1)
        if not valid_rows.any():
            print("Error: No rows found with valid value data.")
            return None
        
        # Only assign prospective option for rows with valid data
        df.loc[valid_rows, 'prospective_option'] = df.loc[valid_rows, value_cols].idxmax(axis=1, skipna=True).str.replace('_value', '')
        
        
        # Step 2: Assign retrospective option (highest progress among subgoal progress for selected goal)
        # Only consider subgoals that have non-NaN values in the value columns
        def get_retrospective_option(row):
            goal_selected = row['goal_selected']
            if pd.isna(goal_selected):
                return np.nan
            
            # Map goal names for progress columns (BR -> BR in progress columns)
            goal_for_progress = 'BR' if goal_selected == 'OB' else goal_selected
            
            # Only consider subgoals that have non-NaN values in the value columns
            subgoals_with_values = []
            for subgoal in ['G', 'S', 'C', 'M']:
                value_col = f'{subgoal}_value'
                if value_col in df.columns and not pd.isna(row[value_col]):
                    subgoals_with_values.append(subgoal)
            
            if not subgoals_with_values:
                return np.nan
            
            # Among subgoals with non-NaN values, find the one with highest progress for selected goal
            valid_progress = {}
            for subgoal in subgoals_with_values:
                progress_col = f'{subgoal}_progress_{goal_for_progress}'
                if progress_col in df.columns and not pd.isna(row[progress_col]):
                    valid_progress[subgoal] = row[progress_col]
            
            if not valid_progress:
                return np.nan
            
            # Return subgoal with highest progress among those with non-NaN values
            return max(valid_progress, key=valid_progress.get)
        
        df['retrospective_option'] = df.apply(get_retrospective_option, axis=1)
        
        # Filter out rows where prospective_option or retrospective_option is NaN
        df = df.dropna(subset=['prospective_option', 'retrospective_option'])
        
        # Step 3: Filter out rows where prospective and retrospective options have same progress
        def get_progress_safe(row, option_col):
            option = row[option_col]
            if pd.isna(option):
                return np.nan
            
            goal_selected = row['goal_selected']
            goal_for_progress = 'BR' if goal_selected == 'OB' else goal_selected
            progress_col = f"{option}_progress_{goal_for_progress}"
            return row.get(progress_col, np.nan)
        
        df['prospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'prospective_option'), axis=1)
        df['retrospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'retrospective_option'), axis=1)
        
        # Filter out rows where prospective and retrospective options are the same OR have same progress
        df_filtered = df[
            (df['prospective_option'] != df['retrospective_option']) & 
            (df['prospective_progress'] != df['retrospective_progress'])
       ].copy()
        
        print(f"Original data: {len(df)} rows")
        print(f"After filtering same prospective/retrospective options AND same progress: {len(df_filtered)} rows")
        
        if len(df_filtered) == 0:
            print("No cases found where prospective and retrospective options differ")
            return None
        
        # Step 4: Assign third option (neither prospective nor retrospective)
        all_subgoals = ['G', 'S', 'C', 'M']
        
        def get_third_options(row):
            return [s for s in all_subgoals if s not in [row['prospective_option'], row['retrospective_option']]]
        
        df_filtered['third_options'] = df_filtered.apply(get_third_options, axis=1)
        
        # Step 5: Analyze choice patterns (keep third option as separate category, add incongruent)
        def classify_choice(row):
            selected = row['subgoal_selected']
            prospective = row['prospective_option']
            retrospective = row['retrospective_option']
            third_options = row['third_options']
            
            # Initialize choice flags
            chose_retrospective = 0
            chose_prospective = 0
            chose_third = 0
            chose_incongruent = 0
            
            # Check if selected subgoal has NaN value (incongruent choice)
            selected_value_col = f'{selected}_value'
            is_incongruent = (selected_value_col in df.columns and pd.isna(row[selected_value_col]))
            
            if is_incongruent:
                chose_incongruent = 1
            elif selected == retrospective:
                chose_retrospective = 1
            elif selected == prospective:
                chose_prospective = 1
            elif selected in third_options:
                # Third option selected - keep as third option (no reclassification)
                chose_third = 1
            
            return pd.Series({
                'chose_retrospective': chose_retrospective,
                'chose_prospective': chose_prospective, 
                'chose_third': chose_third,
                'chose_incongruent': chose_incongruent
            })
        
        # Apply the classification function
        choice_classifications = df_filtered.apply(classify_choice, axis=1)
        df_filtered = pd.concat([df_filtered, choice_classifications], axis=1)
        
        # Print diagnostic information 
        print(f"\nChoice classification summary:")
        print(f"  Direct retrospective choices: {df_filtered[df_filtered['subgoal_selected'] == df_filtered['retrospective_option']].shape[0]}")
        print(f"  Direct prospective choices: {df_filtered[df_filtered['subgoal_selected'] == df_filtered['prospective_option']].shape[0]}")
        
        # Count third option choices (no reclassification)
        third_selected = df_filtered[df_filtered['subgoal_selected'].isin([item for sublist in df_filtered['third_options'] for item in sublist])]
        total_third_choices = third_selected.shape[0]
        
        # Count incongruent choices (selected subgoal has NaN value)
        incongruent_selected = df_filtered[df_filtered.apply(lambda row: pd.isna(row.get(f"{row['subgoal_selected']}_value", np.nan)), axis=1)]
        total_incongruent_choices = incongruent_selected.shape[0]
        
        print(f"  Third option choices (kept as third): {total_third_choices}")
        print(f"  Incongruent choices (selected subgoal with NaN value): {total_incongruent_choices}")
        
        # Final totals
        total_retrospective = df_filtered['chose_retrospective'].sum()
        total_prospective = df_filtered['chose_prospective'].sum()
        total_third = df_filtered['chose_third'].sum()
        total_incongruent = df_filtered['chose_incongruent'].sum()
        print(f"\nFinal classification totals:")
        print(f"  Retrospective: {total_retrospective}")
        print(f"  Prospective: {total_prospective}")
        print(f"  Third: {total_third}")
        print(f"  Incongruent: {total_incongruent}")
        print(f"  Total: {total_retrospective + total_prospective + total_third + total_incongruent}")
        
        # Aggregate by block type
        block_summary = df_filtered.groupby('block_type').agg({
            'chose_retrospective': ['sum', 'count'],
            'chose_prospective': 'sum',
            'chose_third': 'sum',
            'chose_incongruent': 'sum'
        }).reset_index()
        
        # Flatten column names
        block_summary.columns = ['block_type', 'retrospective_count', 'total_trials', 'prospective_count', 'third_count', 'incongruent_count']
        
        # Calculate proportions
        block_summary['prop_retrospective'] = block_summary['retrospective_count'] / block_summary['total_trials']
        block_summary['prop_prospective'] = block_summary['prospective_count'] / block_summary['total_trials']
        block_summary['prop_third'] = block_summary['third_count'] / block_summary['total_trials']
        block_summary['prop_incongruent'] = block_summary['incongruent_count'] / block_summary['total_trials']
        
        print("\nSubgoal choice proportions by block type:")
        print(block_summary[['block_type', 'prop_retrospective', 'prop_prospective', 'prop_third', 'prop_incongruent']])
        
        # Analysis: Collapse across goals but separate high/low conditions
        df_collapsed = df_filtered.copy()
        
        # Extract condition (high/low) from block_type
        df_collapsed['condition'] = df_collapsed['block_type'].str.split('-').str[1]  # Extract 'high' or 'low'
        
        # Aggregate by condition (high/low) instead of specific block types
        condition_summary = df_collapsed.groupby('condition').agg({
            'chose_retrospective': ['sum', 'count'],
            'chose_prospective': 'sum',
            'chose_third': 'sum',
            'chose_incongruent': 'sum'
        }).reset_index()
        
        # Flatten column names
        condition_summary.columns = ['condition', 'retrospective_count', 'total_trials', 'prospective_count', 'third_count', 'incongruent_count']
        
        # Calculate proportions
        condition_summary['prop_retrospective'] = condition_summary['retrospective_count'] / condition_summary['total_trials']
        condition_summary['prop_prospective'] = condition_summary['prospective_count'] / condition_summary['total_trials']
        condition_summary['prop_third'] = condition_summary['third_count'] / condition_summary['total_trials']
        condition_summary['prop_incongruent'] = condition_summary['incongruent_count'] / condition_summary['total_trials']
        
        print("\nSubgoal choice proportions by condition (collapsed across goals):")
        print(condition_summary[['condition', 'prop_retrospective', 'prop_prospective', 'prop_third', 'prop_incongruent']])
        
        # Statistical significance testing between retrospective and prospective choices
        from scipy import stats
        
        # For each condition, test if retrospective vs prospective proportions are significantly different
        def get_sig_label(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return 'ns'
        
        sig_results = {}
        for condition in condition_summary['condition']:
            cond_data = df_collapsed[df_collapsed['condition'] == condition]
            
            # Create contingency table for chi-square test
            retrospective_count = cond_data['chose_retrospective'].sum()
            prospective_count = cond_data['chose_prospective'].sum()
            total_choices = retrospective_count + prospective_count
            
            if total_choices > 0:
                # Chi-square test for proportions
                # H0: retrospective proportion = prospective proportion (0.5 each)
                expected_each = total_choices / 2
                observed = [retrospective_count, prospective_count]
                expected = [expected_each, expected_each]
                
                chi2, p_value = stats.chisquare(observed, expected)
                sig_results[condition] = {
                    'p_value': p_value,
                    'sig_label': get_sig_label(p_value),
                    'retrospective_count': retrospective_count,
                    'prospective_count': prospective_count,
                    'total': total_choices
                }
                
                print(f"\n{condition} condition:")
                print(f"  Retrospective: {retrospective_count}/{total_choices} ({retrospective_count/total_choices:.3f})")
                print(f"  Prospective: {prospective_count}/{total_choices} ({prospective_count/total_choices:.3f})")
                print(f"  Chi-square test p-value: {p_value:.4f} ({get_sig_label(p_value)})")
        
        # Plot collapsed data
        plt.figure(figsize=(8, 5))
        
        # Prepare data for plotting (collapsed version)
        plot_data_collapsed = pd.melt(
            condition_summary,
            id_vars=['condition'],
            value_vars=['prop_retrospective', 'prop_prospective', 'prop_third', 'prop_incongruent'],
            var_name='choice_type',
            value_name='proportion'
        )
        
        # Clean up choice type labels
        plot_data_collapsed['choice_type'] = plot_data_collapsed['choice_type'].str.replace('prop_', '')
        
        # Create bar plot for collapsed data
        # Map colors: prospective=white, retrospective=black, others=gray shades
        color_map = {
            'prospective': '#FFFFFF',  # White
            'retrospective': '#000000',  # Black
            'third': '#808080',  # Gray
            'incongruent': '#A9A9A9'  # Dark gray
        }
        # Use explicit order matching value_vars order
        choice_type_order = ['retrospective', 'prospective', 'third', 'incongruent']
        palette = [color_map.get(ct, '#808080') for ct in choice_type_order]
        
        ax3 = sns.barplot(
            data=plot_data_collapsed,
            x='condition',
            y='proportion',
            hue='choice_type',
            hue_order=choice_type_order,
            palette=palette,
            edgecolor='black',
            width=0.6
        )

        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        
        # Customize plot
        #ax3.set_title("Subgoal selection: Experiment " + str(experiment), fontsize=15, weight='bold')
        ax3.set_xlabel("Condition", fontsize=15, fontweight='bold')
        ax3.set_ylabel("Choice probability", fontsize=15, fontweight='bold')
        ax3.set_ylim(0, 1.0)
        ax3.legend(prop={'size': 13, 'weight': 'bold'})
        
        # Increase font size of tick labels and make them bold
        ax3.tick_params(axis='x', labelsize=14)
        ax3.tick_params(axis='y', labelsize=14)

        # Make tick labels bold
        for label in ax3.get_xticklabels():
            label.set_weight('bold')
        for label in ax3.get_yticklabels():
            label.set_weight('bold')
        
        # Add significance markers for retrospective vs prospective comparisons
        # Position significance markers between retrospective and prospective bars
        # conditions = condition_summary['condition'].unique()
        # for i, condition in enumerate(conditions):
        #     if condition in sig_results:
        #         sig_label = sig_results[condition]['sig_label']
                
        #         # Get the positions of retrospective and prospective bars
        #         retro_bar = ax3.containers[0][i]  # Retrospective bars (first container)
        #         pros_bar = ax3.containers[1][i]   # Prospective bars (second container)
                
        #         # Calculate position between the two bars
        #         x_pos = (retro_bar.get_x() + retro_bar.get_width()/2 + 
        #                 pros_bar.get_x() + pros_bar.get_width()/2) / 2
                
        #         # Get the maximum height of the two bars for positioning
        #         max_height = max(retro_bar.get_height(), pros_bar.get_height())
        #         y_pos = max_height + 0.075
                
        #         # Add significance marker
        #         ax3.text(x_pos, y_pos, sig_label, ha='center', va='bottom', 
        #                 fontsize=12, fontweight='bold')
                
        #         # Add bracket if significant
        #         if sig_label != 'ns':
        #             retro_x = retro_bar.get_x() + retro_bar.get_width()/2
        #             pros_x = pros_bar.get_x() + pros_bar.get_width()/2
        #             bracket_y = y_pos - 0.02  # Position bracket slightly below text
                    
        #             # Draw bracket
        #             ax3.plot([retro_x, retro_x], [max_height + 0.01, bracket_y], 'k-', linewidth=1)
        #             ax3.plot([pros_x, pros_x], [max_height + 0.01, bracket_y], 'k-', linewidth=1)
        #             ax3.plot([retro_x, pros_x], [bracket_y, bracket_y], 'k-', linewidth=1)
        
        plt.tight_layout()
        sns.despine(top=True, right=True)
        plt.savefig(self.figure_dir + "/retrospective_bias_subgoals_" + str(self.experiment) + ".png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return block_summary, condition_summary

    def plot_retrospective_bias_goals(self):
        pros_optimal_csv = CACHE_DIR + "all_subjects_model_parameters_prospective_" + self.data_type + "_optimal_True.csv"
        df_pros = pd.read_csv(pros_optimal_csv)
        
        # Replace 'BR' with 'OB' for consistency
        df_pros['goal_selected'] = df_pros['goal_selected'].replace('BR', 'OB')
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }
        df_pros['block_type'] = df_pros['block_type'].replace(block_labels)
        
        # Create a copy to work with
        df = df_pros.copy()
        
        # Step 1: Assign prospective option (highest value among SH_value, BR_value, HO_value)
        value_cols = ['SH_value', 'BR_value', 'HO_value']
        # Handle NaN values by only considering rows where at least one value column is not NaN
        df['prospective_option'] = df[value_cols].idxmax(axis=1, skipna=True).str.replace('_value', '')
        
        # Step 2: Assign retrospective option (highest progress among SH_progress, BR_progress, HO_progress)
        progress_cols = ['SH_progress', 'BR_progress', 'HO_progress']
        df['retrospective_option'] = df[progress_cols].idxmax(axis=1, skipna=True).str.replace('_progress', '')
        
        # Filter out rows where prospective_option or retrospective_option is NaN
        df = df.dropna(subset=['prospective_option', 'retrospective_option'])
        
        # Step 3: Assign third option (neither prospective nor retrospective)
        all_goals = ['SH', 'BR', 'HO']
        df['third_option'] = df.apply(
            lambda row: [g for g in all_goals if g not in [row['prospective_option'], row['retrospective_option']]][0], 
            axis=1
        )
        
        # Step 4: Filter out rows where prospective and retrospective options have same progress
        def get_progress_safe(row, option_col):
            option = row[option_col]
            if pd.isna(option):
                return np.nan
            progress_col = f"{option}_progress"
            return row.get(progress_col, np.nan)
        
        df['prospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'prospective_option'), axis=1)
        df['retrospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'retrospective_option'), axis=1)
        df = df[
            (df['prospective_option'] != df['retrospective_option']) 
        ].copy()
        df = df[df['prospective_progress'] != df['retrospective_progress']].copy()
        
        # Step 5: Analyze choice patterns
        # Replace 'BR' with 'OB' for option matching
        df['prospective_option'] = df['prospective_option'].replace('BR', 'OB')
        df['retrospective_option'] = df['retrospective_option'].replace('BR', 'OB')
        
        # Compute proportion of retrospective vs prospective choices
        df['chose_retrospective'] = (df['goal_selected'] == df['retrospective_option']).astype(int)
        df['chose_prospective'] = (df['goal_selected'] == df['prospective_option']).astype(int)
        df['chose_third'] = (df['goal_selected'] == df['third_option']).astype(int)
        
        # Aggregate by block type
        block_summary = df.groupby('block_type').agg({
            'chose_retrospective': ['sum', 'count'],
            'chose_prospective': 'sum',
            'chose_third': 'sum'
        }).reset_index()
        
        # Flatten column names
        block_summary.columns = ['block_type', 'retrospective_count', 'total_trials', 'prospective_count', 'third_count']
        
        # Calculate proportions
        block_summary['prop_retrospective'] = block_summary['retrospective_count'] / block_summary['total_trials']
        block_summary['prop_prospective'] = block_summary['prospective_count'] / block_summary['total_trials']
        block_summary['prop_third'] = block_summary['third_count'] / block_summary['total_trials']
        
        print("\nChoice proportions by block type:")
        print(block_summary[['block_type', 'prop_retrospective', 'prop_prospective', 'prop_third']])
        
        # Analysis: Collapse across goals but separate high/low conditions
        df_collapsed = df.copy()
        
        # Extract condition (high/low) from block_type
        df_collapsed['condition'] = df_collapsed['block_type'].str.split('-').str[1]  # Extract 'high' or 'low'
        
        # Aggregate by condition (high/low) instead of specific block types
        condition_summary = df_collapsed.groupby('condition').agg({
            'chose_retrospective': ['sum', 'count'],
            'chose_prospective': 'sum',
            'chose_third': 'sum'
        }).reset_index()
        
        # Flatten column names
        condition_summary.columns = ['condition', 'retrospective_count', 'total_trials', 'prospective_count', 'third_count']
        
        # Calculate proportions
        condition_summary['prop_retrospective'] = condition_summary['retrospective_count'] / condition_summary['total_trials']
        condition_summary['prop_prospective'] = condition_summary['prospective_count'] / condition_summary['total_trials']
        condition_summary['prop_third'] = condition_summary['third_count'] / condition_summary['total_trials']
        
        print("\nChoice proportions by condition (collapsed across goals):")
        print(condition_summary[['condition', 'prop_retrospective', 'prop_prospective', 'prop_third']])
        
        # Statistical significance testing between retrospective and prospective choices
        from scipy import stats
        
        # For each condition, test if retrospective vs prospective proportions are significantly different
        def get_sig_label(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return 'ns'
        
        sig_results = {}
        for condition in condition_summary['condition']:
            cond_data = df_collapsed[df_collapsed['condition'] == condition]
            
            # Create contingency table for chi-square test
            retrospective_count = cond_data['chose_retrospective'].sum()
            prospective_count = cond_data['chose_prospective'].sum()
            total_choices = retrospective_count + prospective_count
            
            if total_choices > 0:
                # Chi-square test for proportions
                # H0: retrospective proportion = prospective proportion (0.5 each)
                expected_each = total_choices / 2
                observed = [retrospective_count, prospective_count]
                expected = [expected_each, expected_each]
                
                chi2, p_value = stats.chisquare(observed, expected)
                sig_results[condition] = {
                    'p_value': p_value,
                    'sig_label': get_sig_label(p_value),
                    'retrospective_count': retrospective_count,
                    'prospective_count': prospective_count,
                    'total': total_choices
                }
                
                print(f"\n{condition} condition:")
                print(f"  Retrospective: {retrospective_count}/{total_choices} ({retrospective_count/total_choices:.3f})")
                print(f"  Prospective: {prospective_count}/{total_choices} ({prospective_count/total_choices:.3f})")
                print(f"  Chi-square test p-value: {p_value:.4f} ({get_sig_label(p_value)})")
        
        # Plot collapsed data
        plt.figure(figsize=(7, 5))
        
        # Prepare data for plotting (collapsed version)
        plot_data_collapsed = pd.melt(
            condition_summary,
            id_vars=['condition'],
            value_vars=['prop_retrospective', 'prop_prospective', 'prop_third'],
            var_name='choice_type',
            value_name='proportion'
        )
        
        # Clean up choice type labels
        plot_data_collapsed['choice_type'] = plot_data_collapsed['choice_type'].str.replace('prop_', '')
        
        # Create bar plot for collapsed data
        # Map colors: prospective=white, retrospective=black, others=gray shades
        color_map = {
            'prospective': '#FFFFFF',  # White
            'retrospective': '#000000',  # Black
            'third': '#808080'  # Gray
        }
        # Use explicit order matching value_vars order
        choice_type_order = ['retrospective', 'prospective', 'third']
        palette = [color_map.get(ct, '#808080') for ct in choice_type_order]
        
        ax3 = sns.barplot(
            data=plot_data_collapsed,
            x='condition',
            y='proportion',
            hue='choice_type',
            hue_order=choice_type_order,
            palette=palette,
            edgecolor='black',
            width=0.6
        )
        
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        
        # Customize plot
        #ax3.set_title("", fontsize=14, weight='bold')
        ax3.set_xlabel("Condition", fontsize=15, fontweight='bold')
        ax3.set_ylabel("Choice probability", fontsize=15, fontweight='bold')
        ax3.set_ylim(0, 1.0)
        ax3.legend(prop={'size': 13, 'weight': 'bold'})
        
        # Increase font size of tick labels and make them bold
        ax3.tick_params(axis='x', labelsize=14)
        ax3.tick_params(axis='y', labelsize=14)

        # Make tick labels bold
        for label in ax3.get_xticklabels():
            label.set_weight('bold')
        for label in ax3.get_yticklabels():
            label.set_weight('bold')
        
        # Add significance markers for retrospective vs prospective comparisons
        # Position significance markers between retrospective and prospective bars
        # conditions = condition_summary['condition'].unique()
        # for i, condition in enumerate(conditions):
        #     if condition in sig_results:
        #         sig_label = sig_results[condition]['sig_label']
                
        #         # Get the positions of retrospective and prospective bars
        #         retro_bar = ax3.containers[0][i]  # Retrospective bars (first container)
        #         pros_bar = ax3.containers[1][i]   # Prospective bars (second container)
                
        #         # Calculate position between the two bars
        #         x_pos = (retro_bar.get_x() + retro_bar.get_width()/2 + 
        #                 pros_bar.get_x() + pros_bar.get_width()/2) / 2
                
        #         # Get the maximum height of the two bars for positioning
        #         max_height = max(retro_bar.get_height(), pros_bar.get_height())
        #         y_pos = max_height + 0.05
                
        #         # Add significance marker
        #         ax3.text(x_pos, y_pos, sig_label, ha='center', va='bottom', 
        #                 fontsize=12, fontweight='bold')
                
        #         # Add bracket if significant
        #         if sig_label != 'ns':
        #             retro_x = retro_bar.get_x() + retro_bar.get_width()/2
        #             pros_x = pros_bar.get_x() + pros_bar.get_width()/2
        #             bracket_y = y_pos - 0.02  # Position bracket slightly below text
                    
        #             # Draw bracket
        #             ax3.plot([retro_x, retro_x], [max_height + 0.01, bracket_y], 'k-', linewidth=1)
        #             ax3.plot([pros_x, pros_x], [max_height + 0.01, bracket_y], 'k-', linewidth=1)
        #             ax3.plot([retro_x, pros_x], [bracket_y, bracket_y], 'k-', linewidth=1)
        
        plt.tight_layout()
        sns.despine(top=True, right=True)
        plt.savefig(self.figure_dir + "/retrospective_bias_goals_" + str(self.experiment) + ".png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return block_summary, condition_summary

    def plot_retrospective_bias_subgoals_simulated(self, model_name='momentum_learn_alt_goal'):
        """
        Plot retrospective bias for subgoals using simulated data from a model.
        
        Args:
            model_name (str): Name of the model to simulate (e.g., 'momentum_learn_alt_goal', 'prospective')
        """
        # Load simulated behavior data
        behavior_dataframe_csv = f"{CACHE_DIR}all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
        df_sim = pd.read_csv(behavior_dataframe_csv)
        
        # Load model parameters CSV to get value columns
        # Try different possible CSV names for model parameters
        model_params_csvs = [
            f"{CACHE_DIR}all_subjects_model_parameters_{model_name}_{self.data_type}_optimal_True.csv",
            f"{CACHE_DIR}all_subjects_model_parameters_{model_name}_{self.data_type}_optimal_False.csv",
            f"{CACHE_DIR}all_subjects_model_parameters_{self.data_type}.csv"  # Fallback for momentum_learn_alt_goal
        ]
        
        df_params = None
        for csv_path in model_params_csvs:
            try:
                df_params = pd.read_csv(csv_path)
                print(f"Loaded model parameters from: {csv_path}")
                break
            except FileNotFoundError:
                continue
        
        if df_params is None:
            print(f"Error: Could not find model parameters CSV for {model_name}")
            print("Tried:", model_params_csvs)
            return None
        
        # Merge model parameters with simulated behavior data
        merge_keys = ['subject_id', 'block_num', 'trial_num']
        # Get all value columns that might exist
        value_cols_needed = ['G_value', 'S_value', 'C_value', 'M_value']
        available_value_cols = [col for col in value_cols_needed if col in df_params.columns]
        
        if not available_value_cols:
            print(f"Error: No value columns found in model parameters CSV")
            return None
        
        df = pd.merge(df_sim, df_params[merge_keys + available_value_cols], 
                     on=merge_keys, how='left')
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }
        df['block_type'] = df['block_type'].replace(block_labels)
        
        # Step 1: Assign prospective option (highest value among G_value, S_value, C_value, M_value)
        value_cols = ['G_value', 'S_value', 'C_value', 'M_value']
        
        # Check which value columns exist and add missing ones as NaN
        for col in value_cols:
            if col not in df.columns:
                df[col] = np.nan
                print(f"Warning: {col} column not found in data, filling with NaN")
        
        # Check if any value columns have non-NaN values
        has_valid_values = df[value_cols].notna().any().any()
        if not has_valid_values:
            print("Error: All value columns (G_value, S_value, C_value, M_value) are missing or contain only NaN values.")
            print("Cannot proceed with prospective option assignment.")
            return None
        
        # Handle NaN values by only considering rows where at least one value column is not NaN
        valid_rows = df[value_cols].notna().any(axis=1)
        if not valid_rows.any():
            print("Error: No rows found with valid value data.")
            return None
        
        # Only assign prospective option for rows with valid data
        df.loc[valid_rows, 'prospective_option'] = df.loc[valid_rows, value_cols].idxmax(axis=1, skipna=True).str.replace('_value', '')
        
        # Step 2: Assign retrospective option (highest progress among subgoal progress for selected goal)
        def get_retrospective_option(row):
            goal_selected = row['goal_selected']
            if pd.isna(goal_selected):
                return np.nan
            
            goal_for_progress = 'BR' if goal_selected == 'OB' else goal_selected
            
            subgoals_with_values = []
            for subgoal in ['G', 'S', 'C', 'M']:
                value_col = f'{subgoal}_value'
                if value_col in df.columns and not pd.isna(row[value_col]):
                    subgoals_with_values.append(subgoal)
            
            if not subgoals_with_values:
                return np.nan
            
            valid_progress = {}
            for subgoal in subgoals_with_values:
                progress_col = f'{subgoal}_progress_{goal_for_progress}'
                if progress_col in df.columns and not pd.isna(row[progress_col]):
                    valid_progress[subgoal] = row[progress_col]
            
            if not valid_progress:
                return np.nan
            
            return max(valid_progress, key=valid_progress.get)
        
        df['retrospective_option'] = df.apply(get_retrospective_option, axis=1)
        
        # Filter out rows where prospective_option or retrospective_option is NaN
        df = df.dropna(subset=['prospective_option', 'retrospective_option'])
        
        # Step 3: Filter out rows where prospective and retrospective options have same progress
        def get_progress_safe(row, option_col):
            option = row[option_col]
            if pd.isna(option):
                return np.nan
            
            goal_selected = row['goal_selected']
            goal_for_progress = 'BR' if goal_selected == 'OB' else goal_selected
            progress_col = f"{option}_progress_{goal_for_progress}"
            return row.get(progress_col, np.nan)
        
        df['prospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'prospective_option'), axis=1)
        df['retrospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'retrospective_option'), axis=1)
        
        # Filter out rows where prospective and retrospective options are the same OR have same progress
        df_filtered = df[
            (df['prospective_option'] != df['retrospective_option']) & 
            (df['prospective_progress'] != df['retrospective_progress'])
        ].copy()
        
        print(f"\n{model_name} - Original data: {len(df)} rows")
        print(f"After filtering same prospective/retrospective options AND same progress: {len(df_filtered)} rows")
        
        if len(df_filtered) == 0:
            print("No cases found where prospective and retrospective options differ")
            return None
        
        # Step 4: Assign third option (neither prospective nor retrospective)
        all_subgoals = ['G', 'S', 'C', 'M']
        
        def get_third_options(row):
            return [s for s in all_subgoals if s not in [row['prospective_option'], row['retrospective_option']]]
        
        df_filtered['third_options'] = df_filtered.apply(get_third_options, axis=1)
        
        # Step 5: Analyze choice patterns
        def classify_choice(row):
            selected = row['subgoal_selected']
            prospective = row['prospective_option']
            retrospective = row['retrospective_option']
            third_options = row['third_options']
            
            chose_retrospective = 0
            chose_prospective = 0
            chose_third = 0
            chose_incongruent = 0
            
            selected_value_col = f'{selected}_value'
            is_incongruent = (selected_value_col in df.columns and pd.isna(row[selected_value_col]))
            
            if is_incongruent:
                chose_incongruent = 1
            elif selected == retrospective:
                chose_retrospective = 1
            elif selected == prospective:
                chose_prospective = 1
            elif selected in third_options:
                chose_third = 1
            
            return pd.Series({
                'chose_retrospective': chose_retrospective,
                'chose_prospective': chose_prospective, 
                'chose_third': chose_third,
                'chose_incongruent': chose_incongruent
            })
        
        choice_classifications = df_filtered.apply(classify_choice, axis=1)
        df_filtered = pd.concat([df_filtered, choice_classifications], axis=1)
        
        # Aggregate by condition (high/low)
        df_collapsed = df_filtered.copy()
        df_collapsed['condition'] = df_collapsed['block_type'].str.split('-').str[1]
        
        condition_summary = df_collapsed.groupby('condition').agg({
            'chose_retrospective': ['sum', 'count'],
            'chose_prospective': 'sum',
            'chose_third': 'sum',
            'chose_incongruent': 'sum'
        }).reset_index()
        
        condition_summary.columns = ['condition', 'retrospective_count', 'total_trials', 'prospective_count', 'third_count', 'incongruent_count']
        
        condition_summary['prop_retrospective'] = condition_summary['retrospective_count'] / condition_summary['total_trials']
        condition_summary['prop_prospective'] = condition_summary['prospective_count'] / condition_summary['total_trials']
        condition_summary['prop_third'] = condition_summary['third_count'] / condition_summary['total_trials']
        condition_summary['prop_incongruent'] = condition_summary['incongruent_count'] / condition_summary['total_trials']
        
        print(f"\n{model_name} - Subgoal choice proportions by condition:")
        print(condition_summary[['condition', 'prop_retrospective', 'prop_prospective', 'prop_third', 'prop_incongruent']])
        
        # Plot
        plt.figure(figsize=(8, 5))
        
        plot_data_collapsed = pd.melt(
            condition_summary,
            id_vars=['condition'],
            value_vars=['prop_retrospective', 'prop_prospective', 'prop_third', 'prop_incongruent'],
            var_name='choice_type',
            value_name='proportion'
        )
        
        plot_data_collapsed['choice_type'] = plot_data_collapsed['choice_type'].str.replace('prop_', '')
        
        ax = sns.barplot(
            data=plot_data_collapsed,
            x='condition',
            y='proportion',
            hue='choice_type',
            palette=['#e74c3c', '#3498db', '#95a5a6', '#9b59b6'],
            edgecolor='black',
            width=0.6
        )
        
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        
        ax.set_xlabel("Condition", fontsize=15, fontweight='bold')
        ax.set_ylabel("Choice probability", fontsize=15, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend(prop={'size': 13, 'weight': 'bold'}, title='Choice type')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        for label in ax.get_xticklabels():
            label.set_weight('bold')
        for label in ax.get_yticklabels():
            label.set_weight('bold')
        
        plt.tight_layout()
        sns.despine(top=True, right=True)
        plt.savefig(self.figure_dir + f"/retrospective_bias_subgoals_{model_name}_{self.experiment}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return condition_summary

    def plot_retrospective_bias_goals_simulated(self, model_name='momentum_learn_alt_goal'):
        """
        Plot retrospective bias for goals using simulated data from a model.
        
        Args:
            model_name (str): Name of the model to simulate (e.g., 'momentum_learn_alt_goal', 'prospective')
        """
        # Load simulated behavior data
        behavior_dataframe_csv = f"{CACHE_DIR}all_subjects_behavior_data_{self.data_type}_{model_name}_simulated.csv"
        df_sim = pd.read_csv(behavior_dataframe_csv)
        
        # Load model parameters CSV to get value columns
        # Try different possible CSV names for model parameters
        model_params_csvs = [
            f"{CACHE_DIR}all_subjects_model_parameters_{model_name}_{self.data_type}_optimal_True.csv",
            f"{CACHE_DIR}all_subjects_model_parameters_{model_name}_{self.data_type}_optimal_False.csv",
            f"{CACHE_DIR}all_subjects_model_parameters_{self.data_type}.csv"  # Fallback for momentum_learn_alt_goal
        ]
        
        df_params = None
        for csv_path in model_params_csvs:
            try:
                df_params = pd.read_csv(csv_path)
                print(f"Loaded model parameters from: {csv_path}")
                break
            except FileNotFoundError:
                continue
        
        if df_params is None:
            print(f"Error: Could not find model parameters CSV for {model_name}")
            print("Tried:", model_params_csvs)
            return None
        
        # Merge model parameters with simulated behavior data
        merge_keys = ['subject_id', 'block_num', 'trial_num']
        df = pd.merge(df_sim, df_params[merge_keys + ['SH_value', 'BR_value', 'HO_value']], 
                     on=merge_keys, how='left')
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }
        df['block_type'] = df['block_type'].replace(block_labels)
        
        # Step 1: Assign prospective option (highest value among SH_value, BR_value, HO_value)
        value_cols = ['SH_value', 'BR_value', 'HO_value']
        df['prospective_option'] = df[value_cols].idxmax(axis=1, skipna=True).str.replace('_value', '')
        
        # Step 2: Assign retrospective option (highest progress among SH_progress, BR_progress, HO_progress)
        progress_cols = ['SH_progress', 'BR_progress', 'HO_progress']
        df['retrospective_option'] = df[progress_cols].idxmax(axis=1, skipna=True).str.replace('_progress', '')
        
        # Filter out rows where prospective_option or retrospective_option is NaN
        df = df.dropna(subset=['prospective_option', 'retrospective_option'])
        
        # Step 3: Assign third option
        all_goals = ['SH', 'BR', 'HO']
        df['third_option'] = df.apply(
            lambda row: [g for g in all_goals if g not in [row['prospective_option'], row['retrospective_option']]][0], 
            axis=1
        )
        
        # Step 4: Calculate progress values
        def get_progress_safe(row, option_col):
            option = row[option_col]
            if pd.isna(option):
                return np.nan
            progress_col = f"{option}_progress"
            return row.get(progress_col, np.nan)
        
        df['prospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'prospective_option'), axis=1)
        df['retrospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'retrospective_option'), axis=1)
        
        # Replace 'BR' with 'OB' for option matching
        df['prospective_option'] = df['prospective_option'].replace('BR', 'OB')
        df['retrospective_option'] = df['retrospective_option'].replace('BR', 'OB')
        df['third_option'] = df['third_option'].replace('BR', 'OB')
        
        # Filter to cases where options differ and progress differs
        df = df[
            (df['prospective_option'] != df['retrospective_option']) & 
            (df['prospective_progress'] != df['retrospective_progress'])
        ].copy()
        
        print(f"\n{model_name} - Filtered data: {len(df)} rows")
        
        if len(df) == 0:
            print("No cases found where prospective and retrospective options differ")
            return None
        
        # Step 5: Analyze choice patterns
        df['chose_retrospective'] = (df['goal_selected'] == df['retrospective_option']).astype(int)
        df['chose_prospective'] = (df['goal_selected'] == df['prospective_option']).astype(int)
        df['chose_third'] = (df['goal_selected'] == df['third_option']).astype(int)
        
        # Aggregate by condition
        df['condition'] = df['block_type'].str.split('-').str[1]
        
        condition_summary = df.groupby('condition').agg({
            'chose_retrospective': ['sum', 'count'],
            'chose_prospective': 'sum',
            'chose_third': 'sum'
        }).reset_index()
        
        condition_summary.columns = ['condition', 'retrospective_count', 'total_trials', 'prospective_count', 'third_count']
        
        condition_summary['prop_retrospective'] = condition_summary['retrospective_count'] / condition_summary['total_trials']
        condition_summary['prop_prospective'] = condition_summary['prospective_count'] / condition_summary['total_trials']
        condition_summary['prop_third'] = condition_summary['third_count'] / condition_summary['total_trials']
        
        print(f"\n{model_name} - Goal choice proportions by condition:")
        print(condition_summary[['condition', 'prop_retrospective', 'prop_prospective', 'prop_third']])
        
        # Plot
        plt.figure(figsize=(7, 5))
        
        plot_data_collapsed = pd.melt(
            condition_summary,
            id_vars=['condition'],
            value_vars=['prop_retrospective', 'prop_prospective', 'prop_third'],
            var_name='choice_type',
            value_name='proportion'
        )
        
        plot_data_collapsed['choice_type'] = plot_data_collapsed['choice_type'].str.replace('prop_', '')
        
        ax = sns.barplot(
            data=plot_data_collapsed,
            x='condition',
            y='proportion',
            hue='choice_type',
            palette=['#e74c3c', '#3498db', '#95a5a6'],
            edgecolor='black',
            width=0.6
        )
        
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        
        ax.set_xlabel("Condition", fontsize=15, fontweight='bold')
        ax.set_ylabel("Choice probability", fontsize=15, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend(prop={'size': 13, 'weight': 'bold'}, title='Choice type')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        for label in ax.get_xticklabels():
            label.set_weight('bold')
        for label in ax.get_yticklabels():
            label.set_weight('bold')
        
        plt.tight_layout()
        sns.despine(top=True, right=True)
        plt.savefig(self.figure_dir + f"/retrospective_bias_goals_{model_name}_{self.experiment}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return condition_summary


    def plot_stay_proportion_with_progress(self):
        """
        Plot goal selection stay percentages based on progress levels.
        Divides goal progress into two blocks:
        - Low: [0-2]
        - High: [3-5]
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create previous goal selection for comparison
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df['prev_goal'] = df['prev_goal'].replace('BR', 'OB')
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal'])
        
        # Define progress columns and create progress level categories
        progress_cols = ['SH_progress', 'BR_progress', 'HO_progress']
        
        # Function to categorize progress levels
        def categorize_progress(progress_val):
            if pd.isna(progress_val):
                return None
            elif 0 <= progress_val <= 2:
                return 'Low Progress (0-2)'
            elif 3 <= progress_val <= 5:
                return 'High Progress (3-5)'
            else:
                return None
        
        # Analyze stay proportion collapsed across all goals
        # Create a combined progress level column based on the selected goal's progress
        def get_goal_progress_level(row):
            goal = row['goal_selected']
            # Use BR for progress column but OB for goal matching
            progress_col = 'BR_progress' if goal == 'OB' else f'{goal}_progress'
            
            if progress_col in df.columns:
                progress_val = row[progress_col]
                return categorize_progress(progress_val)
            return None
        
        df['progress_level'] = df.apply(get_goal_progress_level, axis=1)
        
        # Remove rows with undefined progress levels
        df_filtered = df.dropna(subset=['progress_level'])
        
        if len(df_filtered) == 0:
            print("No valid data found for stay proportion analysis")
            return None
        
        # Calculate stay proportion for each progress level (mean over subjects)
        results_data = []
        subjects = df_filtered['subject_id'].unique()
        
        for progress_level in ['Low Progress (0-2)', 'High Progress (3-5)']:
            level_data = df_filtered[df_filtered['progress_level'] == progress_level]
            
            if len(level_data) == 0:
                continue
            
            # Calculate stay proportion for each subject at this progress level
            subject_proportions = []
            total_stayed_all_subjects = 0
            total_trials_all_subjects = 0
            
            for subject in subjects:
                subject_level_data = level_data[level_data['subject_id'] == subject]
                
                if len(subject_level_data) > 0:
                    # Calculate stay proportion for this subject
                    stayed = (subject_level_data['goal_selected'] == subject_level_data['prev_goal']).sum()
                    total = len(subject_level_data)
                    subject_stay_prop = stayed / total if total > 0 else np.nan
                    
                    if not np.isnan(subject_stay_prop):
                        subject_proportions.append(subject_stay_prop)
                        total_stayed_all_subjects += stayed
                        total_trials_all_subjects += total
            
            # Calculate mean stay proportion across subjects
            if len(subject_proportions) > 0:
                mean_stay_proportion = np.mean(subject_proportions)
                sem_stay_proportion = np.std(subject_proportions) / np.sqrt(len(subject_proportions))
            else:
                mean_stay_proportion = np.nan
                sem_stay_proportion = np.nan
            
            results_data.append({
                'progress_level': progress_level,
                'stay_proportion': mean_stay_proportion,
                'sem': sem_stay_proportion,
                'n_subjects': len(subject_proportions),
                'stayed_count': total_stayed_all_subjects,
                'total_count': total_trials_all_subjects,
                'subject_proportions': subject_proportions
            })
        
        if not results_data:
            print("No valid data found for stay proportion analysis")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Print summary
        print("\nStay proportion by progress level (collapsed across goals):")
        for _, row in results_df.iterrows():
            print(f"{row['progress_level']}: "
                  f"{row['stay_proportion']:.3f} ({row['stayed_count']}/{row['total_count']})")
        
        # Create the plot
        plt.figure(figsize=(7, 6))
        
        # Create bar plot with progress levels on x-axis
        ax = sns.barplot(
            data=results_df,
            x='progress_level',
            y='stay_proportion',
            palette=['#D3D3D3', '#696969'],  # Light gray for low, dark gray for high
            edgecolor='black',
            width=0.4
        )
        
        # Add error bars (standard error of the mean)
        x_positions = range(len(results_df))
        ax.errorbar(x_positions, results_df['stay_proportion'], 
                   yerr=results_df['sem'], 
                   fmt='none', 
                   color='black', 
                   capsize=5, 
                   capthick=2, 
                   linewidth=2)
        
        # Customize the plot
        ax.set_xlabel('Progress level', fontsize=16, fontweight='bold')
        ax.set_ylabel('Goal stay probability', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.0)
        
        # Center the bars by adjusting x-axis limits
        ax.set_xlim(-0.4, 1.4)
        
        # Determine experiment number
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        else:
            experiment = ""
        
        ax.set_title(f'Goal Stay Probability: Experiment {experiment}', 
                    fontsize=15, weight='bold')
        
        # Make tick labels bold
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_weight('bold')
        for label in ax.get_yticklabels():
            label.set_weight('bold')
        
        # Add value labels on bars (only for bar containers, not error bar containers)
        # for container in ax.containers:
        #     if hasattr(container, 'patches'):  # Only label bar containers
        #         ax.bar_label(container, fmt='%.3f', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Statistical analysis: compare low vs high progress
        from scipy import stats
        
        print("\nStatistical comparison (Low vs High progress):")
        if len(results_df) == 2:  # Both low and high present
            low_row = results_df[results_df['progress_level'] == 'Low Progress (0-2)'].iloc[0]
            high_row = results_df[results_df['progress_level'] == 'High Progress (3-5)'].iloc[0]
            
            # Chi-square test for difference in proportions
            # Create contingency table: [stayed, switched] for each condition
            low_stayed = low_row['stayed_count']
            low_total = low_row['total_count']
            low_switched = low_total - low_stayed
            
            high_stayed = high_row['stayed_count']
            high_total = high_row['total_count']
            high_switched = high_total - high_stayed
            
            # Contingency table
            contingency = [[low_stayed, low_switched], 
                          [high_stayed, high_switched]]
            
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
            
            print(f"Low={low_row['stay_proportion']:.3f}, "
                  f"High={high_row['stay_proportion']:.3f}, "
                  f"χ²={chi2:.3f}, p={p_value:.4f}")
            
            # Add significance markers to the plot
            if p_value < 0.05:
                # Get bar heights for positioning the significance marker
                bars = ax.patches  # Get all bar rectangles
                bar_heights = [bar.get_height() for bar in bars]
                print(bar_heights)
                if len(bar_heights) >= 2:
                    max_height = max(bar_heights[0], bar_heights[1])
                    # Position the bracket above the bars
                    bracket_height = max_height + 0.05
                    line_height = bracket_height + 0.02

                    # Get the x positions of the bars
                    x0 = bars[0].get_x() + bars[0].get_width() / 2
                    x1 = bars[1].get_x() + bars[1].get_width() / 2

                    # Draw the bracket
                    ax.plot([x0, x0, x1, x1], [bracket_height, line_height, line_height, bracket_height], 
                            'k-', linewidth=1.5)

                    # Add significance marker
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    else:
                        sig_text = '*'

                    ax.text((x0 + x1) / 2, line_height + 0.01, sig_text, 
                            ha='center', va='bottom', fontsize=16, fontweight='bold')
                else:
                    print("Warning: Less than two bars found for significance annotation.")
        else:
            print("Cannot compare - missing low or high progress data")
        
        # Save figure
        plt.savefig(self.figure_dir + f"/stay_proportion_with_progress_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df


    def plot_stay_proportion_with_progress_subgoals(self):
        """
        Plot subgoal selection stay percentages based on subgoal progress levels.
        Divides subgoal progress into two blocks:
        - Low: [0, 0.5)
        - High: [0.5, 1.0]
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create previous goal and subgoal selection for comparison
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df['prev_goal'] = df['prev_goal'].replace('BR', 'OB')
        df['prev_subgoal'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal', 'prev_subgoal'])
        
        # Function to categorize progress levels
        def categorize_progress(progress_val):
            if pd.isna(progress_val):
                return None
            elif 0 <= progress_val < 0.5:
                return 'Low Progress (0-0.5)'
            elif 0.5 <= progress_val <= 1.0:
                return 'High Progress (0.5-1.0)'
            else:
                return None
        
        # Create a combined progress level column based on the selected subgoal's progress for the selected goal
        def get_subgoal_progress_level(row):
            subgoal = row['subgoal_selected']
            goal = row['goal_selected']
            
            # Map OB back to BR for progress column naming
            goal_for_progress = 'BR' if goal == 'OB' else goal
            progress_col = f'{subgoal}_progress_{goal_for_progress}'
            
            if progress_col in df.columns:
                progress_val = row[progress_col]
                return categorize_progress(progress_val)
            return None
        
        df['progress_level'] = df.apply(get_subgoal_progress_level, axis=1)
        
        # Remove rows with undefined progress levels
        df_filtered = df.dropna(subset=['progress_level'])
        
        if len(df_filtered) == 0:
            print("No valid data found for subgoal stay proportion analysis")
            return None
        
        # Calculate stay proportion for each progress level (both goal and subgoal must stay)
        results_data = []
        subjects = df_filtered['subject_id'].unique()
        
        for progress_level in ['Low Progress (0-0.5)', 'High Progress (0.5-1.0)']:
            level_data = df_filtered[df_filtered['progress_level'] == progress_level]
            
            if len(level_data) == 0:
                continue
            
            # Calculate stay proportion for each subject at this progress level
            subject_proportions = []
            total_stayed_all_subjects = 0
            total_trials_all_subjects = 0
            
            for subject in subjects:
                subject_level_data = level_data[level_data['subject_id'] == subject]
                
                if len(subject_level_data) > 0:
                    # Calculate stay proportion for this subject (both goal and subgoal must stay)
                    stayed = ((subject_level_data['goal_selected'] == subject_level_data['prev_goal']) &
                             (subject_level_data['subgoal_selected'] == subject_level_data['prev_subgoal'])).sum()
                    total = len(subject_level_data)
                    subject_stay_prop = stayed / total if total > 0 else np.nan
                    
                    if not np.isnan(subject_stay_prop):
                        subject_proportions.append(subject_stay_prop)
                        total_stayed_all_subjects += stayed
                        total_trials_all_subjects += total
            
            # Calculate mean stay proportion across subjects
            if len(subject_proportions) > 0:
                mean_stay_proportion = np.mean(subject_proportions)
                sem_stay_proportion = np.std(subject_proportions) / np.sqrt(len(subject_proportions))
            else:
                mean_stay_proportion = np.nan
                sem_stay_proportion = np.nan
            
            results_data.append({
                'progress_level': progress_level,
                'stay_proportion': mean_stay_proportion,
                'sem': sem_stay_proportion,
                'n_subjects': len(subject_proportions),
                'stayed_count': total_stayed_all_subjects,
                'total_count': total_trials_all_subjects,
                'subject_proportions': subject_proportions
            })
        
        if not results_data:
            print("No valid data found for subgoal stay proportion analysis")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Print summary
        print("\nSubgoal stay proportion by progress level (both goal and subgoal must stay):")
        for _, row in results_df.iterrows():
            print(f"{row['progress_level']}: "
                  f"{row['stay_proportion']:.3f} ({row['stayed_count']}/{row['total_count']})")
        
        # Create the plot
        plt.figure(figsize=(7, 6))
        
        # Create bar plot with progress levels on x-axis
        ax = sns.barplot(
            data=results_df,
            x='progress_level',
            y='stay_proportion',
            palette=['#D3D3D3', '#696969'],  # Light gray for low, dark gray for high
            edgecolor='black',
            width=0.4
        )
        
        # Add error bars (standard error of the mean)
        x_positions = range(len(results_df))
        ax.errorbar(x_positions, results_df['stay_proportion'], 
                   yerr=results_df['sem'], 
                   fmt='none', 
                   color='black', 
                   capsize=5, 
                   capthick=2, 
                   linewidth=2)
        
        # Customize the plot
        ax.set_xlabel('Subgoal progress level', fontsize=15, fontweight='bold')
        ax.set_ylabel('Subgoal stay probability', fontsize=15, fontweight='bold')
        ax.set_ylim(0, 1.0)
        
        # Center the bars by adjusting x-axis limits
        ax.set_xlim(-0.4, 1.4)
        
        # Determine experiment number
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        else:
            experiment = ""
        
        ax.set_title(f'Subgoal Stay Probability: Experiment {experiment}', 
                    fontsize=15, weight='bold')
        
        # Make tick labels bold
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_weight('bold')
        for label in ax.get_yticklabels():
            label.set_weight('bold')
        
        plt.tight_layout()
        
        # Statistical analysis: compare low vs high progress
        from scipy import stats
        
        print("\nStatistical comparison (Low vs High subgoal progress):")
        if len(results_df) == 2:  # Both low and high present
            low_row = results_df[results_df['progress_level'] == 'Low Progress (0-0.5)'].iloc[0]
            high_row = results_df[results_df['progress_level'] == 'High Progress (0.5-1.0)'].iloc[0]
            
            # Chi-square test for difference in proportions
            # Create contingency table: [stayed, switched] for each condition
            low_stayed = low_row['stayed_count']
            low_total = low_row['total_count']
            low_switched = low_total - low_stayed
            
            high_stayed = high_row['stayed_count']
            high_total = high_row['total_count']
            high_switched = high_total - high_stayed
            
            # Contingency table
            contingency = [[low_stayed, low_switched], 
                          [high_stayed, high_switched]]
            
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
            
            print(f"Low={low_row['stay_proportion']:.3f}, "
                  f"High={high_row['stay_proportion']:.3f}, "
                  f"χ²={chi2:.3f}, p={p_value:.4f}")
            
            # Add significance markers to the plot
            if p_value < 0.05:
                # Get bar heights for positioning the significance marker
                bars = ax.patches  # Get all bar rectangles
                bar_heights = [bar.get_height() for bar in bars]
                if len(bar_heights) >= 2:
                    max_height = max(bar_heights[0], bar_heights[1])
                    # Position the bracket above the bars
                    bracket_height = max_height + 0.05
                    line_height = bracket_height + 0.02

                    # Get the x positions of the bars
                    x0 = bars[0].get_x() + bars[0].get_width() / 2
                    x1 = bars[1].get_x() + bars[1].get_width() / 2

                    # Draw the bracket
                    ax.plot([x0, x0, x1, x1], [bracket_height, line_height, line_height, bracket_height], 
                            'k-', linewidth=1.5)

                    # Add significance marker
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    else:
                        sig_text = '*'

                    ax.text((x0 + x1) / 2, line_height + 0.01, sig_text, 
                            ha='center', va='bottom', fontsize=16, fontweight='bold')
                else:
                    print("Warning: Less than two bars found for significance annotation.")
        else:
            print("Cannot compare - missing low or high progress data")
        
        # Save figure
        plt.savefig(self.figure_dir + f"/stay_proportion_with_progress_subgoals_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df


    def plot_staying_patterns_prospective_retrospective(self):
        """
        Analyze goal staying patterns by examining when the previous goal was 
        the prospective vs retrospective option.
        """
        pros_optimal_csv = CACHE_DIR + "all_subjects_model_parameters_prospective_" + self.data_type + "_optimal_True.csv"
        df_pros = pd.read_csv(pros_optimal_csv)
        
        # Replace 'BR' with 'OB' for consistency
        df_pros['goal_selected'] = df_pros['goal_selected'].replace('BR', 'OB')
        
        # Create a copy to work with
        df = df_pros.copy()
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Step 1: Define prospective and retrospective options (similar to plot_retrospective_bias_goals)
        # Assign prospective option (highest value among SH_value, BR_value, HO_value)
        value_cols = ['SH_value', 'BR_value', 'HO_value']
        df['prospective_option'] = df[value_cols].idxmax(axis=1, skipna=True).str.replace('_value', '')
        
        # Assign retrospective option (highest progress among SH_progress, BR_progress, HO_progress)
        progress_cols = ['SH_progress', 'BR_progress', 'HO_progress']
        df['retrospective_option'] = df[progress_cols].idxmax(axis=1, skipna=True).str.replace('_progress', '')
        
        # Replace 'BR' with 'OB' for option matching
        df['prospective_option'] = df['prospective_option'].replace('BR', 'OB')
        df['retrospective_option'] = df['retrospective_option'].replace('BR', 'OB')
        
        # Step 2: Add previous goal selection column
        df['prev_goal_selected'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df['prev_goal_selected'] = df['prev_goal_selected'].replace('BR', 'OB')
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal_selected', 'prospective_option', 'retrospective_option'])

        # drop rows where prospective_option progress is same as retrospective_option progress
        def get_progress_safe(row, option_col):
            option = row[option_col]
            if pd.isna(option):
                return np.nan
            progress_col = f"{option}_progress"
            return row.get(progress_col, np.nan)
        df['prospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'prospective_option'), axis=1)
        df['retrospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'retrospective_option'), axis=1)
        df = df[df['prospective_progress'] != df['retrospective_progress']].copy()
        
        # Step 3: Add block condition labels
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }
        df['block_condition'] = df['block_type'].replace(block_labels)
        # Extract high/low condition
        df['condition'] = df['block_condition'].str.split('-').str[1]  # Extract 'high' or 'low'
        
        # Step 4: Filter to goal stays first, then analyze by action_outcome and condition
        # Filter to only trials where prev_goal_selected == goal_selected
        df_stays = df[df['prev_goal_selected'] == df['goal_selected']].copy()
        
        if len(df_stays) == 0:
            print("No goal stays found")
            return None
            
        print(f"Total goal stays found: {len(df_stays)}")
        
        # Calculate proportions for each subject, action_outcome, and condition
        results_data = []
        subjects = df_stays['subject_id'].unique()
        
        # Initialize totals for tracking across all subjects
        totals = {}
        for action_outcome in [0, 1]:
            for condition in ['high', 'low']:
                totals[(action_outcome, condition)] = {
                    'prev_prospective': 0, 
                    'prev_retrospective': 0, 
                    'stays': 0
                }
        
        for subject in subjects:
            subject_stays = df_stays[df_stays['subject_id'] == subject]
            
            # Analyze by action_outcome and condition for this subject
            for action_outcome in [0, 1]:
                for condition in ['high', 'low']:
                    condition_stays = subject_stays[
                        (subject_stays['action_outcome'] == action_outcome) & 
                        (subject_stays['condition'] == condition)
                    ]
                    
                    if len(condition_stays) > 0:
                        # Count when previous goal was prospective vs retrospective for this subject/condition
                        prev_was_prospective = (condition_stays['prev_goal_selected'] == condition_stays['prospective_option']).sum()
                        prev_was_retrospective = (condition_stays['prev_goal_selected'] == condition_stays['retrospective_option']).sum()
                        total_condition_stays = len(condition_stays)
                        
                        # Calculate proportions for this subject and condition combination
                        prop_prev_prospective = prev_was_prospective / total_condition_stays if total_condition_stays > 0 else 0
                        prop_prev_retrospective = prev_was_retrospective / total_condition_stays if total_condition_stays > 0 else 0
                        
                        results_data.append({
                            'subject_id': subject,
                            'action_outcome': action_outcome,
                            'condition': condition,
                            'prop_prev_prospective': prop_prev_prospective,
                            'prop_prev_retrospective': prop_prev_retrospective,
                            'n_stays': total_condition_stays
                        })
                        
                        # Update totals across all subjects
                        totals[(action_outcome, condition)]['prev_prospective'] += prev_was_prospective
                        totals[(action_outcome, condition)]['prev_retrospective'] += prev_was_retrospective
                        totals[(action_outcome, condition)]['stays'] += total_condition_stays
        
        if not results_data:
            print("No valid staying data found")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Calculate means and standard errors by action_outcome and condition
        summary_stats = {}
        for action_outcome in [0, 1]:
            for condition in ['high', 'low']:
                condition_data = results_df[
                    (results_df['action_outcome'] == action_outcome) & 
                    (results_df['condition'] == condition)
                ]
                
                if len(condition_data) > 0:
                    mean_prev_prospective = condition_data['prop_prev_prospective'].mean()
                    mean_prev_retrospective = condition_data['prop_prev_retrospective'].mean()
                    sem_prev_prospective = condition_data['prop_prev_prospective'].std() / np.sqrt(len(condition_data))
                    sem_prev_retrospective = condition_data['prop_prev_retrospective'].std() / np.sqrt(len(condition_data))
                    
                    summary_stats[(action_outcome, condition)] = {
                        'mean_prev_prospective': mean_prev_prospective,
                        'mean_prev_retrospective': mean_prev_retrospective,
                        'sem_prev_prospective': sem_prev_prospective,
                        'sem_prev_retrospective': sem_prev_retrospective,
                        'n_subjects': len(condition_data['subject_id'].unique()),
                        'total_stays': totals[(action_outcome, condition)]['stays'],
                        'total_prev_prospective': totals[(action_outcome, condition)]['prev_prospective'],
                        'total_prev_retrospective': totals[(action_outcome, condition)]['prev_retrospective']
                    }
        
        # Print summary statistics by condition
        print(f"\nStaying patterns analysis by action_outcome and block condition:")
        print(f"Analysis approach: Calculate proportions per subject, then take mean across subjects")
        print(f"Number of subjects: {len(results_df['subject_id'].unique())}")
        
        for action_outcome in [0, 1]:
            action_name = "Unsuccessful" if action_outcome == 0 else "Successful"
            print(f"\n{action_name} trials (action_outcome = {action_outcome}):")
            
            for condition in ['high', 'low']:
                key = (action_outcome, condition)
                if key in summary_stats:
                    stats = summary_stats[key]
                    print(f"  {condition.capitalize()} condition:")
                    print(f"    Subjects with data: {stats['n_subjects']}")
                    print(f"    Total stays across all subjects: {stats['total_stays']}")
                    if stats['total_stays'] > 0:
                        print(f"    Overall prev was prospective: {stats['total_prev_prospective']}/{stats['total_stays']} ({stats['total_prev_prospective']/stats['total_stays']:.3f})")
                        print(f"    Overall prev was retrospective: {stats['total_prev_retrospective']}/{stats['total_stays']} ({stats['total_prev_retrospective']/stats['total_stays']:.3f})")
                        print(f"    Mean proportion prev was prospective (across subjects): {stats['mean_prev_prospective']:.3f} ± {stats['sem_prev_prospective']:.3f}")
                        print(f"    Mean proportion prev was retrospective (across subjects): {stats['mean_prev_retrospective']:.3f} ± {stats['sem_prev_retrospective']:.3f}")
        
        # Create a single bar plot with conditions on x-axis and legend for bar types
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Determine experiment number
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        else:
            experiment = ""
        
        action_names = {0: "Unsuccessful", 1: "Successful"}
        condition_names = {'high': "High", 'low': "Low"}
        
        # Prepare data for plotting
        x_labels = []
        prospective_values = []
        retrospective_values = []
        prospective_sem = []
        retrospective_sem = []
        
        for action_outcome in [0, 1]:
            for condition in ['high', 'low']:
                key = (action_outcome, condition)
                if key in summary_stats:
                    stats = summary_stats[key]
                    x_labels.append(f'{action_names[action_outcome]}-{condition_names[condition]}')
                    prospective_values.append(stats['mean_prev_prospective'])
                    retrospective_values.append(stats['mean_prev_retrospective'])
                    prospective_sem.append(stats['sem_prev_prospective'])
                    retrospective_sem.append(stats['sem_prev_retrospective'])
        
        # Set up bar positions
        x = np.arange(len(x_labels))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, prospective_values, width, 
                      label='Previous was Prospective', color='#3498db', 
                      edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x + width/2, retrospective_values, width,
                      label='Previous was Retrospective', color='#e74c3c',
                      edgecolor='black', alpha=0.8)
        
        # Add error bars
        ax.errorbar(x - width/2, prospective_values, yerr=prospective_sem, 
                   fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
        ax.errorbar(x + width/2, retrospective_values, yerr=retrospective_sem,
                   fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
        
        # Customize the plot
        ax.set_xlabel('Conditions', fontsize=14, fontweight='bold')
        ax.set_ylabel('Goal Stay Probability', fontsize=14, fontweight='bold')
        ax.set_title(f'Goal Staying Patterns by Condition: Experiment {experiment}', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        
        # Format y-axis
        ax.tick_params(axis='y', labelsize=12)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Add legend
        ax.legend(fontsize=12, loc='upper right')
        
        # Add horizontal line at chance level
        #ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add grid
        ax.grid(alpha=0.3, axis='y')
        
        # Add significance testing between prospective and retrospective for each condition
        for i, (prosp_val, retro_val, prosp_sem, retro_sem) in enumerate(zip(prospective_values, retrospective_values, prospective_sem, retrospective_sem)):
            # Get the key for this condition
            keys = list(summary_stats.keys())
            if i < len(keys):
                key = keys[i]
                stats = summary_stats[key]
                
                if stats['total_stays'] >= 10:  # Only test if we have enough data
                    # Statistical analysis for this condition
                    condition_data = results_df[
                        (results_df['action_outcome'] == key[0]) & 
                        (results_df['condition'] == key[1])
                    ]
                    
                    if len(condition_data) > 1:  # Need at least 2 subjects for t-test
                        from scipy import stats as scipy_stats
                        
                        # Paired t-test comparing proportions within subjects for this condition
                        prospective_props = condition_data['prop_prev_prospective'].values
                        retrospective_props = condition_data['prop_prev_retrospective'].values
                        
                        if len(prospective_props) == len(retrospective_props) and len(prospective_props) > 1:
                            t_stat, p_value = scipy_stats.ttest_rel(prospective_props, retrospective_props)
                            
                            print(f"\nStatistical comparison for {x_labels[i]} trials:")
                            print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
                            
                            # Add significance markers to the plot
                            if p_value < 0.05:
                                bracket_height = max(prosp_val, retro_val) + 0.05
                                line_height = bracket_height + 0.02
                                
                                # Draw the bracket
                                ax.plot([i - width/2, i - width/2, i + width/2, i + width/2], 
                                       [bracket_height, line_height, line_height, bracket_height], 
                                       'k-', linewidth=1.5)
                                
                                # Add significance marker
                                if p_value < 0.001:
                                    sig_text = '***'
                                elif p_value < 0.01:
                                    sig_text = '**'
                                else:
                                    sig_text = '*'
                                
                                ax.text(i, line_height + 0.01, sig_text, 
                                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.figure_dir + f"/staying_patterns_prospective_retrospective_by_condition_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df


    def plot_staying_patterns_prospective_retrospective_subgoals(self):
        """
        Analyze subgoal staying patterns by examining when the previous subgoal was 
        the prospective vs retrospective option.
        """
        pros_optimal_csv = CACHE_DIR + "all_subjects_model_parameters_prospective_" + self.data_type + "_optimal_True.csv"
        df_pros = pd.read_csv(pros_optimal_csv)
        
        # Replace 'BR' with 'OB' for consistency
        df_pros['goal_selected'] = df_pros['goal_selected'].replace('BR', 'OB')
        
        # Create a copy to work with
        df = df_pros.copy()
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Step 1: Define prospective and retrospective options for subgoals
        # Assign prospective option (highest value among G_value, C_value, S_value, M_value)
        value_cols = ['G_value', 'C_value', 'S_value', 'M_value']
        
        # Check which value columns exist and add missing ones as NaN
        for col in value_cols:
            if col not in df.columns:
                df[col] = np.nan
                print(f"Warning: {col} column not found in data, filling with NaN")
        
        # Check if any value columns have non-NaN values
        has_valid_values = df[value_cols].notna().any().any()
        if not has_valid_values:
            print("Error: All value columns (G_value, C_value, S_value, M_value) are missing or contain only NaN values.")
            print("Cannot proceed with prospective option assignment.")
            return None
        
        # Handle NaN values by only considering rows where at least one value column is not NaN
        valid_rows = df[value_cols].notna().any(axis=1)
        if not valid_rows.any():
            print("Error: No rows found with valid value data.")
            return None
        
        # Only assign prospective option for rows with valid data
        df.loc[valid_rows, 'prospective_option'] = df.loc[valid_rows, value_cols].idxmax(axis=1, skipna=True).str.replace('_value', '')
        
        # Step 2: Assign retrospective option (highest progress among subgoal progress for selected goal)
        def get_retrospective_option(row):
            goal_selected = row['goal_selected']
            if pd.isna(goal_selected):
                return np.nan
            
            # Map goal names for progress columns (OB -> BR in progress columns)
            goal_for_progress = 'BR' if goal_selected == 'OB' else goal_selected
            
            # Only consider subgoals that have non-NaN values in the value columns
            subgoals_with_values = []
            for subgoal in ['G', 'C', 'S', 'M']:
                value_col = f'{subgoal}_value'
                if value_col in df.columns and not pd.isna(row[value_col]):
                    subgoals_with_values.append(subgoal)
            
            if not subgoals_with_values:
                return np.nan
            
            # Among subgoals with non-NaN values, find the one with highest progress for selected goal
            valid_progress = {}
            for subgoal in subgoals_with_values:
                progress_col = f'{subgoal}_progress_{goal_for_progress}'
                if progress_col in df.columns and not pd.isna(row[progress_col]):
                    valid_progress[subgoal] = row[progress_col]
            
            if not valid_progress:
                return np.nan
            
            # Return subgoal with highest progress among those with non-NaN values
            return max(valid_progress, key=valid_progress.get)
        
        df['retrospective_option'] = df.apply(get_retrospective_option, axis=1)
        
        # Filter out rows where prospective_option or retrospective_option is NaN
        df = df.dropna(subset=['prospective_option', 'retrospective_option'])
        
        # Step 3: Filter out rows where prospective and retrospective options have same progress
        def get_progress_safe(row, option_col):
            option = row[option_col]
            if pd.isna(option):
                return np.nan
            goal_selected = row['goal_selected']
            goal_for_progress = 'BR' if goal_selected == 'OB' else goal_selected
            progress_col = f"{option}_progress_{goal_for_progress}"
            return row.get(progress_col, np.nan)
        
        df['prospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'prospective_option'), axis=1)
        df['retrospective_progress'] = df.apply(lambda row: get_progress_safe(row, 'retrospective_option'), axis=1)
        df = df[df['prospective_progress'] != df['retrospective_progress']].copy()
        
        # Step 4: Add block condition labels
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }
        df['block_condition'] = df['block_type'].replace(block_labels)
        df['condition'] = df['block_condition'].str.split('-').str[1]  # Extract 'high' or 'low'
        
        # Step 5: Add previous subgoal selection column
        df['prev_subgoal_selected'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_subgoal_selected', 'prospective_option', 'retrospective_option'])
        
        # Step 6: Filter to subgoal stays first, then analyze by action_outcome and condition
        # Filter to only trials where prev_subgoal_selected == subgoal_selected
        df_stays = df[df['prev_subgoal_selected'] == df['subgoal_selected']].copy()
        
        if len(df_stays) == 0:
            print("No subgoal stays found")
            return None
            
        print(f"Total subgoal stays found: {len(df_stays)}")
        
        # Calculate proportions for each subject, action_outcome, and condition
        results_data = []
        subjects = df_stays['subject_id'].unique()
        
        # Initialize totals for tracking across all subjects
        totals = {}
        for action_outcome in [0, 1]:
            for condition in ['high', 'low']:
                totals[(action_outcome, condition)] = {
                    'prev_prospective': 0, 
                    'prev_retrospective': 0, 
                    'stays': 0
                }
        
        for subject in subjects:
            subject_stays = df_stays[df_stays['subject_id'] == subject]
            
            # Analyze by action_outcome and condition for this subject
            for action_outcome in [0, 1]:
                for condition in ['high', 'low']:
                    condition_stays = subject_stays[
                        (subject_stays['action_outcome'] == action_outcome) & 
                        (subject_stays['condition'] == condition)
                    ]
                    
                    if len(condition_stays) > 0:
                        # Count when previous subgoal was prospective vs retrospective for this subject/condition
                        prev_was_prospective = (condition_stays['prev_subgoal_selected'] == condition_stays['prospective_option']).sum()
                        prev_was_retrospective = (condition_stays['prev_subgoal_selected'] == condition_stays['retrospective_option']).sum()
                        total_condition_stays = len(condition_stays)
                        
                        # Calculate proportions for this subject and condition combination
                        prop_prev_prospective = prev_was_prospective / total_condition_stays if total_condition_stays > 0 else 0
                        prop_prev_retrospective = prev_was_retrospective / total_condition_stays if total_condition_stays > 0 else 0
                        
                        results_data.append({
                            'subject_id': subject,
                            'action_outcome': action_outcome,
                            'condition': condition,
                            'prop_prev_prospective': prop_prev_prospective,
                            'prop_prev_retrospective': prop_prev_retrospective,
                            'n_stays': total_condition_stays
                        })
                        
                        # Update totals across all subjects
                        totals[(action_outcome, condition)]['prev_prospective'] += prev_was_prospective
                        totals[(action_outcome, condition)]['prev_retrospective'] += prev_was_retrospective
                        totals[(action_outcome, condition)]['stays'] += total_condition_stays
        
        if not results_data:
            print("No valid staying data found")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Calculate means and standard errors by action_outcome and condition
        summary_stats = {}
        for action_outcome in [0, 1]:
            for condition in ['high', 'low']:
                condition_data = results_df[
                    (results_df['action_outcome'] == action_outcome) & 
                    (results_df['condition'] == condition)
                ]
                
                if len(condition_data) > 0:
                    mean_prev_prospective = condition_data['prop_prev_prospective'].mean()
                    mean_prev_retrospective = condition_data['prop_prev_retrospective'].mean()
                    sem_prev_prospective = condition_data['prop_prev_prospective'].std() / np.sqrt(len(condition_data))
                    sem_prev_retrospective = condition_data['prop_prev_retrospective'].std() / np.sqrt(len(condition_data))
                    
                    summary_stats[(action_outcome, condition)] = {
                        'mean_prev_prospective': mean_prev_prospective,
                        'mean_prev_retrospective': mean_prev_retrospective,
                        'sem_prev_prospective': sem_prev_prospective,
                        'sem_prev_retrospective': sem_prev_retrospective,
                        'n_subjects': len(condition_data['subject_id'].unique()),
                        'total_stays': totals[(action_outcome, condition)]['stays'],
                        'total_prev_prospective': totals[(action_outcome, condition)]['prev_prospective'],
                        'total_prev_retrospective': totals[(action_outcome, condition)]['prev_retrospective']
                    }
        
        # Print summary statistics by condition
        print(f"\nSubgoal staying patterns analysis by action_outcome and block condition:")
        print(f"Analysis approach: Calculate proportions per subject, then take mean across subjects")
        print(f"Number of subjects: {len(results_df['subject_id'].unique())}")
        
        for action_outcome in [0, 1]:
            action_name = "Unsuccessful" if action_outcome == 0 else "Successful"
            print(f"\n{action_name} trials (action_outcome = {action_outcome}):")
            
            for condition in ['high', 'low']:
                key = (action_outcome, condition)
                if key in summary_stats:
                    stats = summary_stats[key]
                    print(f"  {condition.capitalize()} condition:")
                    print(f"    Subjects with data: {stats['n_subjects']}")
                    print(f"    Total stays across all subjects: {stats['total_stays']}")
                    if stats['total_stays'] > 0:
                        print(f"    Overall prev was prospective: {stats['total_prev_prospective']}/{stats['total_stays']} ({stats['total_prev_prospective']/stats['total_stays']:.3f})")
                        print(f"    Overall prev was retrospective: {stats['total_prev_retrospective']}/{stats['total_stays']} ({stats['total_prev_retrospective']/stats['total_stays']:.3f})")
                        print(f"    Mean proportion prev was prospective (across subjects): {stats['mean_prev_prospective']:.3f} ± {stats['sem_prev_prospective']:.3f}")
                        print(f"    Mean proportion prev was retrospective (across subjects): {stats['mean_prev_retrospective']:.3f} ± {stats['sem_prev_retrospective']:.3f}")
        
        # Create a single bar plot with conditions on x-axis and legend for bar types
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Determine experiment number
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        else:
            experiment = ""
        
        action_names = {0: "Unsuccessful", 1: "Successful"}
        condition_names = {'high': "High", 'low': "Low"}
        
        # Prepare data for plotting
        x_labels = []
        prospective_values = []
        retrospective_values = []
        prospective_sem = []
        retrospective_sem = []
        
        for action_outcome in [0, 1]:
            for condition in ['high', 'low']:
                key = (action_outcome, condition)
                if key in summary_stats:
                    stats = summary_stats[key]
                    x_labels.append(f'{action_names[action_outcome]}-{condition_names[condition]}')
                    prospective_values.append(stats['mean_prev_prospective'])
                    retrospective_values.append(stats['mean_prev_retrospective'])
                    prospective_sem.append(stats['sem_prev_prospective'])
                    retrospective_sem.append(stats['sem_prev_retrospective'])
        
        # Set up bar positions
        x = np.arange(len(x_labels))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, prospective_values, width, 
                      label='Previous was Prospective', color='#3498db', 
                      edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x + width/2, retrospective_values, width,
                      label='Previous was Retrospective', color='#e74c3c',
                      edgecolor='black', alpha=0.8)
        
        # Add error bars
        ax.errorbar(x - width/2, prospective_values, yerr=prospective_sem, 
                   fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
        ax.errorbar(x + width/2, retrospective_values, yerr=retrospective_sem,
                   fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
        
        # Customize the plot
        ax.set_xlabel('Conditions', fontsize=14, fontweight='bold')
        ax.set_ylabel('Subgoal Stay Probability', fontsize=14, fontweight='bold')
        ax.set_title(f'Subgoal Staying Patterns by Condition: Experiment {experiment}', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        
        # Format y-axis
        ax.tick_params(axis='y', labelsize=12)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Add legend
        ax.legend(fontsize=12, loc='upper right')
        
        # Add horizontal line at chance level
        #ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add grid
        ax.grid(alpha=0.3, axis='y')
        
        # Add significance testing between prospective and retrospective for each condition
        for i, (prosp_val, retro_val, prosp_sem, retro_sem) in enumerate(zip(prospective_values, retrospective_values, prospective_sem, retrospective_sem)):
            # Get the key for this condition
            keys = list(summary_stats.keys())
            if i < len(keys):
                key = keys[i]
                stats = summary_stats[key]
                
                if stats['total_stays'] >= 10:  # Only test if we have enough data
                    # Statistical analysis for this condition
                    condition_data = results_df[
                        (results_df['action_outcome'] == key[0]) & 
                        (results_df['condition'] == key[1])
                    ]
                    
                    if len(condition_data) > 1:  # Need at least 2 subjects for t-test
                        from scipy import stats as scipy_stats
                        
                        # Paired t-test comparing proportions within subjects for this condition
                        prospective_props = condition_data['prop_prev_prospective'].values
                        retrospective_props = condition_data['prop_prev_retrospective'].values
                        
                        if len(prospective_props) == len(retrospective_props) and len(prospective_props) > 1:
                            t_stat, p_value = scipy_stats.ttest_rel(prospective_props, retrospective_props)
                            
                            print(f"\nStatistical comparison for {x_labels[i]} trials:")
                            print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
                            
                            # Add significance markers to the plot
                            if p_value < 0.05:
                                bracket_height = max(prosp_val, retro_val) + 0.05
                                line_height = bracket_height + 0.02
                                
                                # Draw the bracket
                                ax.plot([i - width/2, i - width/2, i + width/2, i + width/2], 
                                       [bracket_height, line_height, line_height, bracket_height], 
                                       'k-', linewidth=1.5)
                                
                                # Add significance marker
                                if p_value < 0.001:
                                    sig_text = '***'
                                elif p_value < 0.01:
                                    sig_text = '**'
                                else:
                                    sig_text = '*'
                                
                                ax.text(i, line_height + 0.01, sig_text, 
                                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.figure_dir + f"/subgoal_staying_patterns_prospective_retrospective_by_condition_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df


    def plot_goal_transition_heatmaps(self):
        """
        Plot heatmaps of goal transitions for each condition in a single row.
        Shows transition probabilities from previous goal to current goal.
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Sort by subject, block, and trial to maintain temporal order
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Shift goal selections to get transitions
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df = df.dropna(subset=['prev_goal'])
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }
        df['block_condition'] = df['block_type'].replace(block_labels)
        
        # Compute transition probabilities by condition
        transition_matrices = {}
        conditions = sorted(df['block_condition'].unique())
        
        for cond in conditions:
            cond_df = df[df['block_condition'] == cond]
            transition_counts = pd.crosstab(cond_df['prev_goal'], cond_df['goal_selected'], normalize='index')
            transition_matrices[cond] = transition_counts
        
        # Create figure with all heatmaps in a single row
        fig, axs = plt.subplots(1, len(conditions), figsize=(3*len(conditions), 4))
        
        # If only one condition, make axs a list for consistency
        if len(conditions) == 1:
            axs = [axs]
        
        for i, cond in enumerate(conditions):
            sns.heatmap(
                transition_matrices[cond],
                annot=True,
                fmt=".2f",
                cmap="crest",
                cbar=False,
                linewidths=0.5,
                linecolor='gray',
                square=True,
                ax=axs[i],
                annot_kws={"size": 13, "weight": "bold"}
            )
            axs[i].set_title(f'{cond}', fontsize=15, weight='bold')
            axs[i].set_xlabel('To Goal', fontsize=14, fontweight='bold')
            axs[i].set_ylabel('From Goal', fontsize=14, fontweight='bold')
            axs[i].tick_params(labelsize=13)
            
            # Make tick labels bold
            for label in axs[i].get_xticklabels():
                label.set_weight('bold')
            for label in axs[i].get_yticklabels():
                label.set_weight('bold')
        
        # Determine experiment number
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        else:
            experiment = ""
        
        plt.suptitle(f'Goal Transition Matrices: Experiment {experiment}', fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 1.05])
        
        # Save figure
        plt.savefig(self.figure_dir + f"/goal_transition_heatmaps_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\nGoal transition analysis summary:")
        print(f"Number of conditions: {len(conditions)}")
        for cond in conditions:
            total_transitions = len(df[df['block_condition'] == cond])
            print(f"{cond}: {total_transitions} transitions")
        
        return transition_matrices

    def analyze_most_viable_goal_persistence(self):
        """
        Calculate statistical measure comparing diagonal persistence probabilities
        (staying with same goal) for most viable vs other goals across all block types.
        
        Most viable goals by block type:
        Blocktype 0: SH, 1: BR, 2: HO, 3: SH, 4: BR, 5: HO
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Sort by subject, block, and trial to maintain temporal order
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Shift goal selections to get transitions
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df = df.dropna(subset=['prev_goal'])
        
        # Define most viable goals by block type
        most_viable_goals = {
            0: 'SH',  # SH-high
            1: 'OB',  # OB-high (BR -> OB)
            2: 'HO',  # HO-high
            3: 'SH',  # SH-low
            4: 'OB',  # OB-low (BR -> OB)
            5: 'HO'   # HO-low
        }
        
        # Add most viable goal for each block
        df['most_viable_goal'] = df['block_type'].map(most_viable_goals)
        
        # Create binary indicators for staying with same goal (diagonal entries)
        df['prev_is_most_viable'] = (df['prev_goal'] == df['most_viable_goal']).astype(int)
        df['stays_with_same_goal'] = (df['prev_goal'] == df['goal_selected']).astype(int)
        
        # Calculate persistence probabilities for each goal type
        # P(stay with same goal | previous was most viable)
        most_viable_prev = df[df['prev_is_most_viable'] == 1]
        p_stay_most_viable = most_viable_prev['stays_with_same_goal'].mean()
        
        # P(stay with same goal | previous was other goal)
        other_goals_prev = df[df['prev_is_most_viable'] == 0]
        p_stay_other = other_goals_prev['stays_with_same_goal'].mean()
        
        # Calculate counts for statistical test
        n_most_viable_transitions = len(most_viable_prev)
        n_most_viable_stays = most_viable_prev['stays_with_same_goal'].sum()
        
        n_other_transitions = len(other_goals_prev)
        n_other_stays = other_goals_prev['stays_with_same_goal'].sum()
        
        # Statistical test: Chi-square test for independence
        # Create contingency table: [stays, switches] for each goal type
        contingency_table = [
            [n_most_viable_stays, n_most_viable_transitions - n_most_viable_stays],  # most viable: stays, switches
            [n_other_stays, n_other_transitions - n_other_stays]  # other: stays, switches
        ]
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Effect size (Cramér's V)
        n_total = n_most_viable_transitions + n_other_transitions
        cramers_v = np.sqrt(chi2 / (n_total * (min(2, 2) - 1)))
        
        # Print results
        print(f"\n=== Goal Persistence Analysis (Diagonal Entries) ===")
        print(f"Data type: {self.data_type}")
        print(f"Total transitions analyzed: {n_total}")
        print(f"\nPersistence probabilities (staying with same goal):")
        print(f"P(stay | previous was most viable): {p_stay_most_viable:.4f} ({n_most_viable_stays}/{n_most_viable_transitions})")
        print(f"P(stay | previous was other goal): {p_stay_other:.4f} ({n_other_stays}/{n_other_transitions})")
        print(f"\nDifference: {p_stay_most_viable - p_stay_other:.4f}")
        print(f"\nStatistical test:")
        print(f"Chi-square test: χ² = {chi2:.4f}, p = {p_value:.6f}")
        print(f"Cramér's V (effect size): {cramers_v:.4f}")
        
        # Interpretation
        if p_value < 0.001:
            significance = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        else:
            significance = "not significant (p ≥ 0.05)"
            
        print(f"\nInterpretation:")
        print(f"The probability of staying with the same goal when it was most viable ({p_stay_most_viable:.4f}) is")
        if p_stay_most_viable > p_stay_other:
            print(f"higher than staying with other goals ({p_stay_other:.4f})")
        else:
            print(f"lower than staying with other goals ({p_stay_other:.4f})")
        print(f"This difference is {significance}.")
        
        # Effect size interpretation
        if cramers_v < 0.1:
            effect_size = "negligible"
        elif cramers_v < 0.3:
            effect_size = "small"
        elif cramers_v < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"The effect size is {effect_size} (Cramér's V = {cramers_v:.4f}).")
        
        # Additional analysis: Show persistence for each individual goal
        print(f"\n=== Individual Goal Persistence ===")
        for goal in ['SH', 'OB', 'HO']:
            goal_prev = df[df['prev_goal'] == goal]
            if len(goal_prev) > 0:
                goal_stays = goal_prev['stays_with_same_goal'].sum()
                goal_total = len(goal_prev)
                p_stay_goal = goal_stays / goal_total
                print(f"P(stay with {goal} | previous was {goal}): {p_stay_goal:.4f} ({goal_stays}/{goal_total})")
        
        return {
            'p_stay_most_viable': p_stay_most_viable,
            'p_stay_other': p_stay_other,
            'difference': p_stay_most_viable - p_stay_other,
            'chi2': chi2,
            'p_value': p_value,
            'cramers_v': cramers_v,
            'n_total': n_total,
            'contingency_table': contingency_table
        }


    def plot_subgoal_transition_heatmaps(self):
        """
        Plot heatmaps of subgoal transitions for each condition in a single row.
        Shows transition probabilities from previous subgoal to current subgoal.
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Sort by subject, block, and trial to maintain temporal order
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Shift subgoal selections to get transitions
        df['prev_subgoal'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        df = df.dropna(subset=['prev_subgoal'])
        
        # Define string labels for block types 0 to 5
        block_labels = {
            0: "SH-high",
            1: "OB-high", 
            2: "HO-high",
            3: "SH-low",
            4: "OB-low",
            5: "HO-low"
        }
        df['block_condition'] = df['block_type'].replace(block_labels)
        
        # Compute transition probabilities by condition
        transition_matrices = {}
        conditions = sorted(df['block_condition'].unique())
        
        for cond in conditions:
            cond_df = df[df['block_condition'] == cond]
            transition_counts = pd.crosstab(cond_df['prev_subgoal'], cond_df['subgoal_selected'], normalize='index')
            transition_matrices[cond] = transition_counts
        
        # Create figure with all heatmaps in a single row
        fig, axs = plt.subplots(1, len(conditions), figsize=(3*len(conditions), 4))
        
        # If only one condition, make axs a list for consistency
        if len(conditions) == 1:
            axs = [axs]
        
        for i, cond in enumerate(conditions):
            sns.heatmap(
                transition_matrices[cond],
                annot=True,
                fmt=".2f",
                cmap="viridis",
                cbar=False,
                linewidths=0.5,
                linecolor='gray',
                square=True,
                ax=axs[i],
                annot_kws={"size": 13, "weight": "bold"}
            )
            axs[i].set_title(f'{cond}', fontsize=15, weight='bold')
            axs[i].set_xlabel('To Subgoal', fontsize=14, fontweight='bold')
            axs[i].set_ylabel('From Subgoal', fontsize=14, fontweight='bold')
            axs[i].tick_params(labelsize=13)
            
            # Make tick labels bold
            for label in axs[i].get_xticklabels():
                label.set_weight('bold')
            for label in axs[i].get_yticklabels():
                label.set_weight('bold')
        
        # Determine experiment number
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        else:
            experiment = ""
        
        plt.suptitle(f'Subgoal Transition Matrices: Experiment {experiment}', fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 1.05])
        
        # Save figure
        plt.savefig(self.figure_dir + f"/subgoal_transition_heatmaps_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\nSubgoal transition analysis summary:")
        print(f"Number of conditions: {len(conditions)}")
        for cond in conditions:
            total_transitions = len(df[df['block_condition'] == cond])
            print(f"{cond}: {total_transitions} transitions")
        
        return transition_matrices

    def analyze_subgoal_switching_relevance_to_dominant_goal(self):
        """
        Analyze whether participants switch more to subgoals that are relevant to the dominant goal.
        
        Dominant goals: 0:SH, 1:BR, 2:HO, 3:SH, 4:BR, 5:HO
        
        Subgoal requirements:
        - SH: 3M, 3G
        - BR: 2G, 3S, 1C  
        - HO: 2M, 3C, 1S
        
        A subgoal is relevant if its progress requirement is not yet met (progress < 1).
        """
        from scipy import stats
        
        df = self.df.copy()
        
        # Replace BR with OB for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Define dominant goals for each block type
        dominant_goals = {
            0: 'SH',  # SH-high
            1: 'OB',  # BR-high (mapped to OB)
            2: 'HO',  # HO-high
            3: 'SH',  # SH-low
            4: 'OB',  # BR-low (mapped to OB)
            5: 'HO'   # HO-low
        }
        
        # Add dominant goal column
        df['dominant_goal'] = df['block_type'].map(dominant_goals)
        
        # Define subgoal requirements for each dominant goal
        subgoal_requirements = {
            'SH': {'M': 3, 'G': 3},
            'OB': {'G': 2, 'S': 3, 'C': 1},
            'HO': {'M': 2, 'C': 3, 'S': 1}
        }
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create previous subgoal selection for comparison
        df['prev_subgoal'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_subgoal'])
        
        # Only analyze trials where subgoal actually changed (switching trials)
        df_switching = df[df['subgoal_selected'] != df['prev_subgoal']].copy()
        
        if len(df_switching) == 0:
            print("No subgoal switching trials found for analysis")
            return
        
        # Function to check if a subgoal is relevant to the dominant goal
        def is_subgoal_relevant_to_dominant(row):
            subgoal = row['subgoal_selected']
            dominant_goal = row['dominant_goal']
            
            # Check if this subgoal is required for the dominant goal
            if dominant_goal not in subgoal_requirements:
                return False
            
            if subgoal not in subgoal_requirements[dominant_goal]:
                return False
            
            # Check if the subgoal progress requirement is not yet met
            # Map goal names for progress column (OB -> BR)
            goal_for_progress = 'BR' if dominant_goal == 'OB' else dominant_goal
            progress_col = f'{subgoal}_progress_{goal_for_progress}'
            
            if progress_col in row and not pd.isna(row[progress_col]):
                return row[progress_col] < 1
            else:
                return False
        
        # Apply relevance check
        df_switching['is_relevant_to_dominant'] = df_switching.apply(is_subgoal_relevant_to_dominant, axis=1)
        
        # Calculate proportion of relevant switches per subject
        subject_relevance = df_switching.groupby('subject_id')['is_relevant_to_dominant'].mean().reset_index()
        subject_relevance.columns = ['subject_id', 'proportion_relevant_switches']
        
        # Calculate overall statistics
        overall_proportion = df_switching['is_relevant_to_dominant'].mean()
        n_switches = len(df_switching)
        n_relevant_switches = df_switching['is_relevant_to_dominant'].sum()
        
        # Print results
        print("\n" + "="*80)
        print("SUBGOAL SWITCHING RELEVANCE TO DOMINANT GOAL ANALYSIS")
        print("="*80)
        print(f"Total subgoal switching trials analyzed: {n_switches}")
        print(f"Switches to subgoals relevant to dominant goal: {n_relevant_switches}")
        print(f"Overall proportion of relevant switches: {overall_proportion:.3f}")
        print(f"Mean proportion per subject: {subject_relevance['proportion_relevant_switches'].mean():.3f} ± {subject_relevance['proportion_relevant_switches'].std():.3f}")
        
        # Analyze pairwise transitions per subject and perform t-tests
        print(f"\nPAIRWISE TRANSITION ANALYSIS BY SUBJECT:")
        print(f"Calculating transition probabilities for each subject separately")
        
        # Get all unique previous subgoals
        prev_subgoals = df_switching['prev_subgoal'].unique()
        
        # Calculate transition probabilities for each subject
        subject_transition_data = []
        
        for subject_id in df_switching['subject_id'].unique():
            subject_data = df_switching[df_switching['subject_id'] == subject_id]
            
            if len(subject_data) == 0:
                continue
            
            # Calculate overall proportions for this subject
            subject_relevant = subject_data['is_relevant_to_dominant'].sum()
            subject_total = len(subject_data)
            subject_prop_relevant = subject_relevant / subject_total if subject_total > 0 else 0
            subject_prop_non_relevant = 1 - subject_prop_relevant
            
            subject_transition_data.append({
                'subject_id': subject_id,
                'total_transitions': subject_total,
                'relevant_transitions': subject_relevant,
                'non_relevant_transitions': subject_total - subject_relevant,
                'prop_relevant': subject_prop_relevant,
                'prop_non_relevant': subject_prop_non_relevant
            })
        
        # Convert to DataFrame for easier analysis
        subject_df = pd.DataFrame(subject_transition_data)
        
        if len(subject_df) > 0:
            print(f"\nSUBJECT-LEVEL TRANSITION PROBABILITIES:")
            print(f"Number of subjects with switching data: {len(subject_df)}")
            print(f"Mean proportion of relevant transitions: {subject_df['prop_relevant'].mean():.3f} ± {subject_df['prop_relevant'].std():.3f}")
            print(f"Mean proportion of non-relevant transitions: {subject_df['prop_non_relevant'].mean():.3f} ± {subject_df['prop_non_relevant'].std():.3f}")
            
            # Perform pairwise t-test comparing relevant vs non-relevant proportions
            from scipy import stats
            
            # One-sample t-test against 0.5 (equal probability)
            t_stat_relevant, p_value_relevant = stats.ttest_1samp(subject_df['prop_relevant'], 0.5)
            t_stat_non_relevant, p_value_non_relevant = stats.ttest_1samp(subject_df['prop_non_relevant'], 0.5)
            
            print(f"\nSTATISTICAL TESTS:")
            print(f"One-sample t-test (H0: proportion = 0.5):")
            print(f"  Relevant transitions: t={t_stat_relevant:.3f}, p={p_value_relevant:.3f}")
            print(f"  Non-relevant transitions: t={t_stat_non_relevant:.3f}, p={p_value_non_relevant:.3f}")
            
            # Paired t-test comparing relevant vs non-relevant within subjects
            t_stat_paired, p_value_paired = stats.ttest_rel(subject_df['prop_relevant'], subject_df['prop_non_relevant'])
            print(f"  Paired t-test (relevant vs non-relevant): t={t_stat_paired:.3f}, p={p_value_paired:.3f}")
            
            # Show individual subject data
            print(f"\nINDIVIDUAL SUBJECT DATA:")
            for _, row in subject_df.iterrows():
                print(f"  Subject {row['subject_id']}: {row['relevant_transitions']}/{row['total_transitions']} relevant ({row['prop_relevant']:.3f})")
        
        # Analyze by dominant goal type with subject-level data
        print(f"\nANALYSIS BY DOMINANT GOAL (SUBJECT-LEVEL):")
        for dominant_goal in ['SH', 'OB', 'HO']:
            goal_data = df_switching[df_switching['dominant_goal'] == dominant_goal]
            
            if len(goal_data) == 0:
                continue
            
            # Calculate per-subject proportions for this dominant goal
            goal_subject_data = []
            for subject_id in goal_data['subject_id'].unique():
                subject_goal_data = goal_data[goal_data['subject_id'] == subject_id]
                if len(subject_goal_data) > 0:
                    subject_relevant = subject_goal_data['is_relevant_to_dominant'].sum()
                    subject_total = len(subject_goal_data)
                    subject_prop = subject_relevant / subject_total if subject_total > 0 else 0
                    goal_subject_data.append(subject_prop)
            
            if len(goal_subject_data) > 0:
                goal_subject_array = np.array(goal_subject_data)
                print(f"  {dominant_goal}: {len(goal_subject_data)} subjects")
                print(f"    Mean proportion relevant: {goal_subject_array.mean():.3f} ± {goal_subject_array.std():.3f}")
                
                # One-sample t-test for this dominant goal
                t_stat_goal, p_value_goal = stats.ttest_1samp(goal_subject_array, 0.5)
                print(f"    t-test vs 0.5: t={t_stat_goal:.3f}, p={p_value_goal:.3f}")
                
                # Show which subgoals are relevant for this dominant goal
                relevant_subgoals = list(subgoal_requirements[dominant_goal].keys())
                print(f"    Relevant subgoals: {relevant_subgoals}")
        
        print("="*80)



if __name__ == "__main__":
    plot_measures = PlotMeasures(data_type="online", read_from_csv=True)

    # Example usage of the new switching characteristics method
    # plot_measures.plot_switching_characteristics(goal_type="goal")
    
    # # Run the refined bias analysis methods
    # plot_measures.plot_retrospective_bias_subgoals()
    # plot_measures.plot_retrospective_bias_goals()
    

    # # Plot stay proportions with progress
    # plot_measures.plot_stay_proportion_with_progress()
    # plot_measures.plot_stay_proportion_with_progress_subgoals()

    
    #plot_measures.plot_goal_selection_related_goal_progress(simulate=False, model_name="momentum_learn_alt_goal", collapse_over_goals=True)
    #plot_measures.plot_subgoal_selection_related_subgoal_progress(simulate=False, collapse_conditions=False, collapse_over_subgoals=False)
    
    
    # Example usage of the four models comparison
    # plot_measures.plot_goal_selection_related_goal_progress_four_models(models=['momentum_learn_alt_goal', 'prospective', 'td_persistence'], 
    #                                                                   include_behavior=True, collapse_over_goals=False)
    
    # ### Example usage of the four models comparison for subgoals
    # plot_measures.plot_subgoal_selection_related_subgoal_progress_four_models(models=['momentum_learn_alt_goal', 'prospective', 'td_persistence'], 
    #                                                                           include_behavior=True, collapse_over_subgoals=False)



    #plot_measures.plot_goal_selection_per_block()
    #plot_measures.plot_viable_goal_selection_high_low()
    #plot_measures.plot_subgoal_selection_per_block()

    
    # Statistical analysis of most viable goal persistence
    # print("\n" + "="*60)
    # print("MOST VIABLE GOAL PERSISTENCE ANALYSIS")
    # print("="*60)
    # plot_measures.analyze_most_viable_goal_persistence()
    
    plot_measures.plot_goal_transition_heatmaps()
    plot_measures.plot_subgoal_transition_heatmaps()
    #plot_measures.analyze_subgoal_switching_relevance_to_dominant_goal()