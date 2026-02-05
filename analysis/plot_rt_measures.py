from plot_measures import *



class PlotRTMeasures(PlotMeasures):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_goal_rt_by_switching_pattern(self):
        """
        Plot reaction time bar plots comparing:
        1. Both goal and subgoal repeated from previous choice
        2. Goal repeated but subgoal switched from previous choice  
        3. Goal switched from previous choice
        
        Returns:
            DataFrame with summary statistics
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create previous selections for comparison
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df['prev_goal'] = df['prev_goal'].replace('BR', 'OB')
        df['prev_subgoal'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal', 'prev_subgoal', 'goal_rt'])
        
        # Define switching patterns
        def classify_switching_pattern(row):
            current_goal = row['goal_selected']
            prev_goal = row['prev_goal']
            current_subgoal = row['subgoal_selected']
            prev_subgoal = row['prev_subgoal']
            
            if current_goal == prev_goal and current_subgoal == prev_subgoal:
                return 'Both Repeated'
            elif current_goal == prev_goal and current_subgoal != prev_subgoal:
                return 'Goal Repeated, Subgoal Switched'
            elif current_goal != prev_goal:
                return 'Goal Switched'
            else:
                return 'Other'
        
        df['switching_pattern'] = df.apply(classify_switching_pattern, axis=1)
        
        # Filter to only the three main patterns we want to compare
        df_filtered = df[df['switching_pattern'].isin([
            'Both Repeated', 
            'Goal Repeated, Subgoal Switched', 
            'Goal Switched'
        ])].copy()
        
        if len(df_filtered) == 0:
            print("No valid data found for RT analysis")
            return None
        
        # Calculate RT statistics for each switching pattern
        results_data = []
        subjects = df_filtered['subject_id'].unique()
        
        for pattern in ['Both Repeated', 'Goal Repeated, Subgoal Switched', 'Goal Switched']:
            pattern_data = df_filtered[df_filtered['switching_pattern'] == pattern]
            
            if len(pattern_data) == 0:
                continue
            
            # Calculate RT statistics for each subject
            subject_rts = []
            total_rt_all_subjects = 0
            total_trials_all_subjects = 0
            
            for subject in subjects:
                subject_pattern_data = pattern_data[pattern_data['subject_id'] == subject]
                
                if len(subject_pattern_data) > 0:
                    # Calculate mean RT for this subject and pattern
                    subject_mean_rt = subject_pattern_data['goal_rt'].mean()
                    subject_rts.append(subject_mean_rt)
                    total_rt_all_subjects += subject_pattern_data['goal_rt'].sum()
                    total_trials_all_subjects += len(subject_pattern_data)
            
            # Calculate mean RT across subjects
            if len(subject_rts) > 0:
                mean_rt = np.mean(subject_rts)
                sem_rt = np.std(subject_rts) / np.sqrt(len(subject_rts))
                overall_mean_rt = total_rt_all_subjects / total_trials_all_subjects if total_trials_all_subjects > 0 else np.nan
            else:
                mean_rt = np.nan
                sem_rt = np.nan
                overall_mean_rt = np.nan
            
            results_data.append({
                'switching_pattern': pattern,
                'mean_rt': mean_rt,
                'sem_rt': sem_rt,
                'n_subjects': len(subject_rts),
                'total_trials': total_trials_all_subjects,
                'overall_mean_rt': overall_mean_rt,
                'subject_rts': subject_rts
            })
        
        if not results_data:
            print("No valid data found for RT analysis")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Print summary
        print("\nGoal RT by switching pattern:")
        for _, row in results_df.iterrows():
            print(f"{row['switching_pattern']}: "
                  f"{row['mean_rt']:.3f} ± {row['sem_rt']:.3f} ms "
                  f"({row['total_trials']} trials, {row['n_subjects']} subjects)")
        
        # Create the plot with two subplots sharing y-axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        
        # Create bar plot for goal RT with switching patterns on x-axis
        sns.barplot(
            data=results_df,
            x='switching_pattern',
            y='mean_rt',
            palette=['#000000', '#FFFFFF', '#808080'],  # Black, white, gray
            edgecolor='black',
            width=0.4,
            ax=ax1
        )
        
        # Add error bars for goal RT (standard error of the mean)
        x_positions = range(len(results_df))
        ax1.errorbar(x_positions, results_df['mean_rt'], 
                    yerr=results_df['sem_rt'], 
                    fmt='none', 
                    color='black', 
                    capsize=5, 
                    capthick=2, 
                    linewidth=2)
        
        # Customize the goal RT plot
        ax1.set_title('Goal RT', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Switching Pattern', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Goal Reaction Time (ms)', fontsize=16, fontweight='bold')
        
        # Wrap x-axis labels over multiple lines for better readability
        import textwrap
        labels = ax1.get_xticklabels()
        wrapped_labels = [textwrap.fill(label.get_text(), width=20) for label in labels]
        ax1.set_xticklabels(wrapped_labels)
        
        # Determine experiment number
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        else:
            experiment = ""
        
        
        # Make tick labels bold for goal RT plot
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        for label in ax1.get_xticklabels():
            label.set_weight('bold')
        for label in ax1.get_yticklabels():
            label.set_weight('bold')
        
        # Set clean white background for goal RT plot
        ax1.set_facecolor('white')
        ax1.set_ylim(0, 1200)
        
        # Now calculate subgoal RT statistics for the same switching patterns
        subgoal_results_data = []
        
        for pattern in ['Both Repeated', 'Goal Repeated, Subgoal Switched', 'Goal Switched']:
            pattern_data = df_filtered[df_filtered['switching_pattern'] == pattern]
            
            if len(pattern_data) == 0:
                continue
            
            # Calculate subgoal RT statistics for each subject
            subject_subgoal_rts = []
            total_subgoal_rt_all_subjects = 0
            total_trials_all_subjects = 0
            
            for subject in subjects:
                subject_pattern_data = pattern_data[pattern_data['subject_id'] == subject]
                
                if len(subject_pattern_data) > 0:
                    # Calculate mean subgoal RT for this subject and pattern
                    subject_mean_subgoal_rt = subject_pattern_data['subgoal_rt'].mean()
                    subject_subgoal_rts.append(subject_mean_subgoal_rt)
                    total_subgoal_rt_all_subjects += subject_pattern_data['subgoal_rt'].sum()
                    total_trials_all_subjects += len(subject_pattern_data)
            
            # Calculate mean subgoal RT across subjects
            if len(subject_subgoal_rts) > 0:
                mean_subgoal_rt = np.mean(subject_subgoal_rts)
                sem_subgoal_rt = np.std(subject_subgoal_rts) / np.sqrt(len(subject_subgoal_rts))
                overall_mean_subgoal_rt = total_subgoal_rt_all_subjects / total_trials_all_subjects if total_trials_all_subjects > 0 else np.nan
            else:
                mean_subgoal_rt = np.nan
                sem_subgoal_rt = np.nan
                overall_mean_subgoal_rt = np.nan
            
            subgoal_results_data.append({
                'switching_pattern': pattern,
                'mean_rt': mean_subgoal_rt,
                'sem_rt': sem_subgoal_rt,
                'n_subjects': len(subject_subgoal_rts),
                'total_trials': total_trials_all_subjects,
                'overall_mean_rt': overall_mean_subgoal_rt,
                'subject_rts': subject_subgoal_rts
            })
        
        if subgoal_results_data:
            # Convert to DataFrame
            subgoal_results_df = pd.DataFrame(subgoal_results_data)
            
            # Print subgoal RT summary
            print("\nSubgoal RT by switching pattern:")
            for _, row in subgoal_results_df.iterrows():
                print(f"{row['switching_pattern']}: "
                      f"{row['mean_rt']:.3f} ± {row['sem_rt']:.3f} ms "
                      f"({row['total_trials']} trials, {row['n_subjects']} subjects)")
            
            # Create bar plot for subgoal RT with switching patterns on x-axis
            sns.barplot(
                data=subgoal_results_df,
                x='switching_pattern',
                y='mean_rt',
                palette=['#000000', '#FFFFFF', '#808080'],  # Black, white, gray
                edgecolor='black',
                width=0.4,
                ax=ax2
            )
            
            # Add error bars for subgoal RT (standard error of the mean)
            ax2.errorbar(x_positions, subgoal_results_df['mean_rt'], 
                        yerr=subgoal_results_df['sem_rt'], 
                        fmt='none', 
                        color='black', 
                        capsize=5, 
                        capthick=2, 
                        linewidth=2)
            
            # Customize the subgoal RT plot
            ax2.set_title('Subgoal RT', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Switching Pattern', fontsize=16, fontweight='bold')
            ax2.set_ylabel('Subgoal Reaction Time (ms)', fontsize=16, fontweight='bold')
            
            # Wrap x-axis labels over multiple lines for better readability
            labels2 = ax2.get_xticklabels()
            wrapped_labels2 = [textwrap.fill(label.get_text(), width=20) for label in labels2]
            ax2.set_xticklabels(wrapped_labels2)
            
            
            # Make tick labels bold for subgoal RT plot
            ax2.tick_params(axis='x', labelsize=14)
            ax2.tick_params(axis='y', labelsize=14)
            for label in ax2.get_xticklabels():
                label.set_weight('bold')
            for label in ax2.get_yticklabels():
                label.set_weight('bold')
            
            # Set clean white background for subgoal RT plot
            ax2.set_facecolor('white')
            ax2.set_ylim(0, 1200)
        else:
            ax2.set_title('No Subgoal RT Data Available', fontsize=16, weight='bold')
            ax2.text(0.5, 0.5, 'No subgoal RT data found', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
        
        # Statistical analysis: compare patterns for both goal and subgoal RT
        from scipy import stats
        
        print("\nStatistical comparisons for Goal RT:")
        if len(results_df) >= 2:
            # One-way ANOVA across all patterns for goal RT
            all_subject_rts = []
            pattern_labels = []
            
            for _, row in results_df.iterrows():
                all_subject_rts.extend(row['subject_rts'])
                pattern_labels.extend([row['switching_pattern']] * len(row['subject_rts']))
            
            if len(all_subject_rts) > 0 and len(set(pattern_labels)) > 1:
                # One-way ANOVA for goal RT
                f_stat, p_value = stats.f_oneway(*[row['subject_rts'] for _, row in results_df.iterrows() if len(row['subject_rts']) > 0])
                
                print(f"Goal RT One-way ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
                
                # Pairwise t-tests if ANOVA is significant for goal RT
                if p_value < 0.05 and len(results_df) >= 2:
                    print("\nGoal RT Pairwise comparisons (t-tests):")
                    patterns = results_df['switching_pattern'].tolist()
                    
                    # Track significance markers to avoid overlap
                    significance_markers = []
                    
                    for i in range(len(patterns)):
                        for j in range(i+1, len(patterns)):
                            pattern1_data = results_df.iloc[i]['subject_rts']
                            pattern2_data = results_df.iloc[j]['subject_rts']
                            
                            if len(pattern1_data) > 1 and len(pattern2_data) > 1:
                                t_stat, p_val = stats.ttest_ind(pattern1_data, pattern2_data)
                                print(f"  {patterns[i]} vs {patterns[j]}: t={t_stat:.3f}, p={p_val:.4f}")
                                
                                # Add significance markers to the goal RT plot
                                if p_val < 0.05:
                                    # Get bar positions
                                    bars = ax1.patches
                                    if i < len(bars) and j < len(bars):
                                        bar1_height = bars[i].get_height()
                                        bar2_height = bars[j].get_height()
                                        max_height = max(bar1_height, bar2_height)
                                        
                                        # Calculate vertical offset based on number of existing markers
                                        marker_offset = len(significance_markers) * 0.08 * max_height
                                        
                                        # Position the bracket above the bars with vertical offset
                                        base_height = max_height + 0.05 * max_height
                                        bracket_height = base_height + marker_offset
                                        line_height = bracket_height + 0.02 * max_height
                                        
                                        # Get the x positions of the bars
                                        x1 = bars[i].get_x() + bars[i].get_width() / 2
                                        x2 = bars[j].get_x() + bars[j].get_width() / 2
                                        
                                        # Draw the bracket
                                        ax1.plot([x1, x1, x2, x2], [bracket_height, line_height, line_height, bracket_height], 
                                                'k-', linewidth=1.5)
                                        
                                        # Add significance marker
                                        if p_val < 0.001:
                                            sig_text = '***'
                                        elif p_val < 0.01:
                                            sig_text = '**'
                                        else:
                                            sig_text = '*'
                                        
                                        ax1.text((x1 + x2) / 2, line_height + 0.01 * max_height, sig_text, 
                                                ha='center', va='bottom', fontsize=14, fontweight='bold')
                                        
                                        # Track this marker for future positioning
                                        significance_markers.append({
                                            'x1': x1, 'x2': x2, 'height': line_height,
                                            'pattern1': patterns[i], 'pattern2': patterns[j]
                                        })
        
        # Statistical analysis for subgoal RT if data is available
        if subgoal_results_data and len(subgoal_results_df) >= 2:
            print("\nStatistical comparisons for Subgoal RT:")
            
            # One-way ANOVA across all patterns for subgoal RT
            all_subgoal_subject_rts = []
            subgoal_pattern_labels = []
            
            for _, row in subgoal_results_df.iterrows():
                all_subgoal_subject_rts.extend(row['subject_rts'])
                subgoal_pattern_labels.extend([row['switching_pattern']] * len(row['subject_rts']))
            
            if len(all_subgoal_subject_rts) > 0 and len(set(subgoal_pattern_labels)) > 1:
                # One-way ANOVA for subgoal RT
                f_stat_subgoal, p_value_subgoal = stats.f_oneway(*[row['subject_rts'] for _, row in subgoal_results_df.iterrows() if len(row['subject_rts']) > 0])
                
                print(f"Subgoal RT One-way ANOVA: F={f_stat_subgoal:.3f}, p={p_value_subgoal:.4f}")
                
                # Pairwise t-tests if ANOVA is significant for subgoal RT
                if p_value_subgoal < 0.05 and len(subgoal_results_df) >= 2:
                    print("\nSubgoal RT Pairwise comparisons (t-tests):")
                    subgoal_patterns = subgoal_results_df['switching_pattern'].tolist()
                    
                    # Track significance markers to avoid overlap for subgoal RT
                    subgoal_significance_markers = []
                    
                    for i in range(len(subgoal_patterns)):
                        for j in range(i+1, len(subgoal_patterns)):
                            pattern1_data = subgoal_results_df.iloc[i]['subject_rts']
                            pattern2_data = subgoal_results_df.iloc[j]['subject_rts']
                            
                            if len(pattern1_data) > 1 and len(pattern2_data) > 1:
                                t_stat, p_val = stats.ttest_ind(pattern1_data, pattern2_data)
                                print(f"  {subgoal_patterns[i]} vs {subgoal_patterns[j]}: t={t_stat:.3f}, p={p_val:.4f}")
                                
                                # Add significance markers to the subgoal RT plot
                                if p_val < 0.05:
                                    # Get bar positions
                                    bars = ax2.patches
                                    if i < len(bars) and j < len(bars):
                                        bar1_height = bars[i].get_height()
                                        bar2_height = bars[j].get_height()
                                        max_height = max(bar1_height, bar2_height)
                                        
                                        # Calculate vertical offset based on number of existing markers
                                        marker_offset = len(subgoal_significance_markers) * 0.08 * max_height
                                        
                                        # Position the bracket above the bars with vertical offset
                                        base_height = max_height + 0.05 * max_height
                                        bracket_height = base_height + marker_offset
                                        line_height = bracket_height + 0.02 * max_height
                                        
                                        # Get the x positions of the bars
                                        x1 = bars[i].get_x() + bars[i].get_width() / 2
                                        x2 = bars[j].get_x() + bars[j].get_width() / 2
                                        
                                        # Draw the bracket
                                        ax2.plot([x1, x1, x2, x2], [bracket_height, line_height, line_height, bracket_height], 
                                                'k-', linewidth=1.5)
                                        
                                        # Add significance marker
                                        if p_val < 0.001:
                                            sig_text = '***'
                                        elif p_val < 0.01:
                                            sig_text = '**'
                                        else:
                                            sig_text = '*'
                                        
                                        ax2.text((x1 + x2) / 2, line_height + 0.01 * max_height, sig_text, 
                                                ha='center', va='bottom', fontsize=14, fontweight='bold')
                                        
                                        # Track this marker for future positioning
                                        subgoal_significance_markers.append({
                                            'x1': x1, 'x2': x2, 'height': line_height,
                                            'pattern1': subgoal_patterns[i], 'pattern2': subgoal_patterns[j]
                                        })
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(FIGURES_TOPICS + f"/goal_and_subgoal_rt_by_switching_pattern_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return both results if subgoal data is available
        if subgoal_results_data:
            return results_df, subgoal_results_df
        else:
            return results_df, None

    def plot_goal_rt_by_switching_pattern_by_block_half(self):
        """
        Plot reaction time bar plots comparing switching patterns, split by first half vs second half of blocks.
        Compares:
        1. Both goal and subgoal repeated from previous choice
        2. Goal repeated but subgoal switched from previous choice  
        3. Goal switched from previous choice
        
        Returns:
            Tuple of (goal_results_dict, subgoal_results_dict) where each dict has 'first_half' and 'second_half' keys
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Determine block half for each trial
        df['max_trial_in_block'] = df.groupby(['subject_id', 'block_num'])['trial_num'].transform('max')
        df['block_half'] = df.apply(
            lambda row: 'first_half' if row['trial_num'] <= row['max_trial_in_block'] / 2 else 'second_half',
            axis=1
        )
        
        # Create previous selections for comparison
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df['prev_goal'] = df['prev_goal'].replace('BR', 'OB')
        df['prev_subgoal'] = df.groupby(['subject_id', 'block_num'])['subgoal_selected'].shift(1)
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal', 'prev_subgoal', 'goal_rt'])
        
        # Define switching patterns
        def classify_switching_pattern(row):
            current_goal = row['goal_selected']
            prev_goal = row['prev_goal']
            current_subgoal = row['subgoal_selected']
            prev_subgoal = row['prev_subgoal']
            
            if current_goal == prev_goal and current_subgoal == prev_subgoal:
                return 'Both Repeated'
            elif current_goal == prev_goal and current_subgoal != prev_subgoal:
                return 'Goal Repeated, Subgoal Switched'
            elif current_goal != prev_goal:
                return 'Goal Switched'
            else:
                return 'Other'
        
        df['switching_pattern'] = df.apply(classify_switching_pattern, axis=1)
        
        # Filter to only the three main patterns we want to compare
        df_filtered = df[df['switching_pattern'].isin([
            'Both Repeated', 
            'Goal Repeated, Subgoal Switched', 
            'Goal Switched'
        ])].copy()
        
        if len(df_filtered) == 0:
            print("No valid data found for RT analysis")
            return None, None
        
        # Determine experiment number
        if self.data_type == "online":
            experiment = 'H2'
        elif self.data_type == "fmri":
            experiment = 'H1'
        else:
            experiment = ""
        
        # Process each block half
        goal_results_dict = {}
        subgoal_results_dict = {}
        
        for block_half in ['first_half', 'second_half']:
            half_df = df_filtered[df_filtered['block_half'] == block_half].copy()
            
            if len(half_df) == 0:
                print(f"No valid data found for {block_half}")
                goal_results_dict[block_half] = None
                subgoal_results_dict[block_half] = None
                continue
            
            # Calculate RT statistics for each switching pattern
            results_data = []
            subjects = half_df['subject_id'].unique()
            
            for pattern in ['Both Repeated', 'Goal Repeated, Subgoal Switched', 'Goal Switched']:
                pattern_data = half_df[half_df['switching_pattern'] == pattern]
                
                if len(pattern_data) == 0:
                    continue
                
                # Calculate RT statistics for each subject
                subject_rts = []
                total_rt_all_subjects = 0
                total_trials_all_subjects = 0
                
                for subject in subjects:
                    subject_pattern_data = pattern_data[pattern_data['subject_id'] == subject]
                    
                    if len(subject_pattern_data) > 0:
                        # Calculate mean RT for this subject and pattern
                        subject_mean_rt = subject_pattern_data['goal_rt'].mean()
                        subject_rts.append(subject_mean_rt)
                        total_rt_all_subjects += subject_pattern_data['goal_rt'].sum()
                        total_trials_all_subjects += len(subject_pattern_data)
                
                # Calculate mean RT across subjects
                if len(subject_rts) > 0:
                    mean_rt = np.mean(subject_rts)
                    sem_rt = np.std(subject_rts) / np.sqrt(len(subject_rts))
                    overall_mean_rt = total_rt_all_subjects / total_trials_all_subjects if total_trials_all_subjects > 0 else np.nan
                else:
                    mean_rt = np.nan
                    sem_rt = np.nan
                    overall_mean_rt = np.nan
                
                results_data.append({
                    'switching_pattern': pattern,
                    'mean_rt': mean_rt,
                    'sem_rt': sem_rt,
                    'n_subjects': len(subject_rts),
                    'total_trials': total_trials_all_subjects,
                    'overall_mean_rt': overall_mean_rt,
                    'subject_rts': subject_rts
                })
            
            if not results_data:
                print(f"No valid data found for {block_half}")
                goal_results_dict[block_half] = None
                subgoal_results_dict[block_half] = None
                continue
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results_data)
            goal_results_dict[block_half] = results_df
            
            # Print summary
            print(f"\nGoal RT by switching pattern ({block_half}):")
            for _, row in results_df.iterrows():
                print(f"  {row['switching_pattern']}: "
                      f"{row['mean_rt']:.3f} ± {row['sem_rt']:.3f} ms "
                      f"({row['total_trials']} trials, {row['n_subjects']} subjects)")
            
            # Now calculate subgoal RT statistics for the same switching patterns
            subgoal_results_data = []
            
            for pattern in ['Both Repeated', 'Goal Repeated, Subgoal Switched', 'Goal Switched']:
                pattern_data = half_df[half_df['switching_pattern'] == pattern]
                
                if len(pattern_data) == 0:
                    continue
                
                # Calculate subgoal RT statistics for each subject
                subject_subgoal_rts = []
                total_subgoal_rt_all_subjects = 0
                total_trials_all_subjects = 0
                
                for subject in subjects:
                    subject_pattern_data = pattern_data[pattern_data['subject_id'] == subject]
                    
                    if len(subject_pattern_data) > 0:
                        # Calculate mean subgoal RT for this subject and pattern
                        subject_mean_subgoal_rt = subject_pattern_data['subgoal_rt'].mean()
                        subject_subgoal_rts.append(subject_mean_subgoal_rt)
                        total_subgoal_rt_all_subjects += subject_pattern_data['subgoal_rt'].sum()
                        total_trials_all_subjects += len(subject_pattern_data)
                
                # Calculate mean subgoal RT across subjects
                if len(subject_subgoal_rts) > 0:
                    mean_subgoal_rt = np.mean(subject_subgoal_rts)
                    sem_subgoal_rt = np.std(subject_subgoal_rts) / np.sqrt(len(subject_subgoal_rts))
                    overall_mean_subgoal_rt = total_subgoal_rt_all_subjects / total_trials_all_subjects if total_trials_all_subjects > 0 else np.nan
                else:
                    mean_subgoal_rt = np.nan
                    sem_subgoal_rt = np.nan
                    overall_mean_subgoal_rt = np.nan
                
                subgoal_results_data.append({
                    'switching_pattern': pattern,
                    'mean_rt': mean_subgoal_rt,
                    'sem_rt': sem_subgoal_rt,
                    'n_subjects': len(subject_subgoal_rts),
                    'total_trials': total_trials_all_subjects,
                    'overall_mean_rt': overall_mean_subgoal_rt,
                    'subject_rts': subject_subgoal_rts
                })
            
            if subgoal_results_data:
                # Convert to DataFrame
                subgoal_results_df = pd.DataFrame(subgoal_results_data)
                subgoal_results_dict[block_half] = subgoal_results_df
                
                # Print subgoal RT summary
                print(f"\nSubgoal RT by switching pattern ({block_half}):")
                for _, row in subgoal_results_df.iterrows():
                    print(f"  {row['switching_pattern']}: "
                          f"{row['mean_rt']:.3f} ± {row['sem_rt']:.3f} ms "
                          f"({row['total_trials']} trials, {row['n_subjects']} subjects)")
            else:
                subgoal_results_dict[block_half] = None
        
        # Create the plot with four subplots in one row: goal RT (first half, second half) and subgoal RT (first half, second half)
        fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
        ax1_first = axes[0]  # Goal RT - First Half
        ax1_second = axes[1]  # Goal RT - Second Half
        ax2_first = axes[2]  # Subgoal RT - First Half
        ax2_second = axes[3]  # Subgoal RT - Second Half
        
        import textwrap
        from scipy import stats
        
        # Plot goal RT for first half
        if goal_results_dict.get('first_half') is not None:
            first_half_df = goal_results_dict['first_half']
            sns.barplot(
                data=first_half_df,
                x='switching_pattern',
                y='mean_rt',
                palette=['#000000', '#FFFFFF', '#808080'],
                edgecolor='black',
                width=0.4,
                ax=ax1_first
            )
            x_positions = range(len(first_half_df))
            ax1_first.errorbar(x_positions, first_half_df['mean_rt'], 
                            yerr=first_half_df['sem_rt'], 
                            fmt='none', 
                            color='black', 
                            capsize=5, 
                            capthick=2, 
                            linewidth=2)
            
            # Add significance markers
            significance_markers = []
            patterns = first_half_df['switching_pattern'].tolist()
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    pattern1_data = first_half_df.iloc[i]['subject_rts']
                    pattern2_data = first_half_df.iloc[j]['subject_rts']
                    if len(pattern1_data) > 1 and len(pattern2_data) > 1:
                        t_stat, p_val = stats.ttest_ind(pattern1_data, pattern2_data)
                        if p_val < 0.05:
                            bars = ax1_first.patches
                            if i < len(bars) and j < len(bars):
                                bar1_height = bars[i].get_height()
                                bar2_height = bars[j].get_height()
                                max_height = max(bar1_height, bar2_height)
                                marker_offset = len(significance_markers) * 0.08 * max_height
                                base_height = max_height + 0.05 * max_height
                                bracket_height = base_height + marker_offset
                                line_height = bracket_height + 0.02 * max_height
                                x1 = bars[i].get_x() + bars[i].get_width() / 2
                                x2 = bars[j].get_x() + bars[j].get_width() / 2
                                ax1_first.plot([x1, x1, x2, x2], [bracket_height, line_height, line_height, bracket_height], 
                                            'k-', linewidth=1.5)
                                sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                                ax1_first.text((x1 + x2) / 2, line_height + 0.01 * max_height, sig_text, 
                                            ha='center', va='bottom', fontsize=14, fontweight='bold')
                                significance_markers.append({'x1': x1, 'x2': x2, 'height': line_height})
            
            labels = ax1_first.get_xticklabels()
            wrapped_labels = [textwrap.fill(label.get_text(), width=20) for label in labels]
            ax1_first.set_xticklabels(wrapped_labels)
            ax1_first.set_title('Goal RT - First Half', fontsize=16, fontweight='bold')
            ax1_first.set_xlabel('Switching Pattern', fontsize=16, fontweight='bold')
            ax1_first.set_ylabel('Goal Reaction Time (ms)', fontsize=16, fontweight='bold')
            ax1_first.tick_params(axis='x', labelsize=14)
            ax1_first.tick_params(axis='y', labelsize=14)
            for label in ax1_first.get_xticklabels():
                label.set_weight('bold')
            for label in ax1_first.get_yticklabels():
                label.set_weight('bold')
            ax1_first.set_facecolor('white')
            ax1_first.set_ylim(0, 1200)
        
        # Plot goal RT for second half
        if goal_results_dict.get('second_half') is not None:
            second_half_df = goal_results_dict['second_half']
            sns.barplot(
                data=second_half_df,
                x='switching_pattern',
                y='mean_rt',
                palette=['#000000', '#FFFFFF', '#808080'],
                edgecolor='black',
                width=0.4,
                ax=ax1_second
            )
            x_positions = range(len(second_half_df))
            ax1_second.errorbar(x_positions, second_half_df['mean_rt'], 
                            yerr=second_half_df['sem_rt'], 
                            fmt='none', 
                            color='black', 
                            capsize=5, 
                            capthick=2, 
                            linewidth=2)
            
            # Add significance markers
            significance_markers = []
            patterns = second_half_df['switching_pattern'].tolist()
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    pattern1_data = second_half_df.iloc[i]['subject_rts']
                    pattern2_data = second_half_df.iloc[j]['subject_rts']
                    if len(pattern1_data) > 1 and len(pattern2_data) > 1:
                        t_stat, p_val = stats.ttest_ind(pattern1_data, pattern2_data)
                        if p_val < 0.05:
                            bars = ax1_second.patches
                            if i < len(bars) and j < len(bars):
                                bar1_height = bars[i].get_height()
                                bar2_height = bars[j].get_height()
                                max_height = max(bar1_height, bar2_height)
                                marker_offset = len(significance_markers) * 0.08 * max_height
                                base_height = max_height + 0.05 * max_height
                                bracket_height = base_height + marker_offset
                                line_height = bracket_height + 0.02 * max_height
                                x1 = bars[i].get_x() + bars[i].get_width() / 2
                                x2 = bars[j].get_x() + bars[j].get_width() / 2
                                ax1_second.plot([x1, x1, x2, x2], [bracket_height, line_height, line_height, bracket_height], 
                                            'k-', linewidth=1.5)
                                sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                                ax1_second.text((x1 + x2) / 2, line_height + 0.01 * max_height, sig_text, 
                                            ha='center', va='bottom', fontsize=14, fontweight='bold')
                                significance_markers.append({'x1': x1, 'x2': x2, 'height': line_height})
            
            labels = ax1_second.get_xticklabels()
            wrapped_labels = [textwrap.fill(label.get_text(), width=20) for label in labels]
            ax1_second.set_xticklabels(wrapped_labels)
            ax1_second.set_title('Goal RT - Second Half', fontsize=16, fontweight='bold')
            ax1_second.set_xlabel('Switching Pattern', fontsize=16, fontweight='bold')
            ax1_second.tick_params(axis='x', labelsize=14)
            ax1_second.tick_params(axis='y', labelsize=14)
            for label in ax1_second.get_xticklabels():
                label.set_weight('bold')
            for label in ax1_second.get_yticklabels():
                label.set_weight('bold')
            ax1_second.set_facecolor('white')
            ax1_second.set_ylim(0, 1200)
        
        # Plot subgoal RT for first half
        if subgoal_results_dict.get('first_half') is not None:
            first_half_subgoal_df = subgoal_results_dict['first_half']
            sns.barplot(
                data=first_half_subgoal_df,
                x='switching_pattern',
                y='mean_rt',
                palette=['#000000', '#FFFFFF', '#808080'],
                edgecolor='black',
                width=0.4,
                ax=ax2_first
            )
            x_positions = range(len(first_half_subgoal_df))
            ax2_first.errorbar(x_positions, first_half_subgoal_df['mean_rt'], 
                            yerr=first_half_subgoal_df['sem_rt'], 
                            fmt='none', 
                            color='black', 
                            capsize=5, 
                            capthick=2, 
                            linewidth=2)
            
            # Add significance markers
            significance_markers = []
            patterns = first_half_subgoal_df['switching_pattern'].tolist()
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    pattern1_data = first_half_subgoal_df.iloc[i]['subject_rts']
                    pattern2_data = first_half_subgoal_df.iloc[j]['subject_rts']
                    if len(pattern1_data) > 1 and len(pattern2_data) > 1:
                        t_stat, p_val = stats.ttest_ind(pattern1_data, pattern2_data)
                        if p_val < 0.05:
                            bars = ax2_first.patches
                            if i < len(bars) and j < len(bars):
                                bar1_height = bars[i].get_height()
                                bar2_height = bars[j].get_height()
                                max_height = max(bar1_height, bar2_height)
                                marker_offset = len(significance_markers) * 0.08 * max_height
                                base_height = max_height + 0.05 * max_height
                                bracket_height = base_height + marker_offset
                                line_height = bracket_height + 0.02 * max_height
                                x1 = bars[i].get_x() + bars[i].get_width() / 2
                                x2 = bars[j].get_x() + bars[j].get_width() / 2
                                ax2_first.plot([x1, x1, x2, x2], [bracket_height, line_height, line_height, bracket_height], 
                                            'k-', linewidth=1.5)
                                sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                                ax2_first.text((x1 + x2) / 2, line_height + 0.01 * max_height, sig_text, 
                                            ha='center', va='bottom', fontsize=14, fontweight='bold')
                                significance_markers.append({'x1': x1, 'x2': x2, 'height': line_height})
            
            labels = ax2_first.get_xticklabels()
            wrapped_labels = [textwrap.fill(label.get_text(), width=20) for label in labels]
            ax2_first.set_xticklabels(wrapped_labels)
            ax2_first.set_title('Subgoal RT - First Half', fontsize=16, fontweight='bold')
            ax2_first.set_xlabel('Switching Pattern', fontsize=16, fontweight='bold')
            ax2_first.set_ylabel('Subgoal Reaction Time (ms)', fontsize=16, fontweight='bold')
            ax2_first.tick_params(axis='x', labelsize=14)
            ax2_first.tick_params(axis='y', labelsize=14)
            for label in ax2_first.get_xticklabels():
                label.set_weight('bold')
            for label in ax2_first.get_yticklabels():
                label.set_weight('bold')
            ax2_first.set_facecolor('white')
            ax2_first.set_ylim(0, 1200)
        else:
            ax2_first.set_title('No Subgoal RT Data Available', fontsize=16, weight='bold')
            ax2_first.text(0.5, 0.5, 'No subgoal RT data found', ha='center', va='center', 
                        transform=ax2_first.transAxes, fontsize=14)
        
        # Plot subgoal RT for second half
        if subgoal_results_dict.get('second_half') is not None:
            second_half_subgoal_df = subgoal_results_dict['second_half']
            sns.barplot(
                data=second_half_subgoal_df,
                x='switching_pattern',
                y='mean_rt',
                palette=['#000000', '#FFFFFF', '#808080'],
                edgecolor='black',
                width=0.4,
                ax=ax2_second
            )
            x_positions = range(len(second_half_subgoal_df))
            ax2_second.errorbar(x_positions, second_half_subgoal_df['mean_rt'], 
                            yerr=second_half_subgoal_df['sem_rt'], 
                            fmt='none', 
                            color='black', 
                            capsize=5, 
                            capthick=2, 
                            linewidth=2)
            
            # Add significance markers
            significance_markers = []
            patterns = second_half_subgoal_df['switching_pattern'].tolist()
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    pattern1_data = second_half_subgoal_df.iloc[i]['subject_rts']
                    pattern2_data = second_half_subgoal_df.iloc[j]['subject_rts']
                    if len(pattern1_data) > 1 and len(pattern2_data) > 1:
                        t_stat, p_val = stats.ttest_ind(pattern1_data, pattern2_data)
                        if p_val < 0.05:
                            bars = ax2_second.patches
                            if i < len(bars) and j < len(bars):
                                bar1_height = bars[i].get_height()
                                bar2_height = bars[j].get_height()
                                max_height = max(bar1_height, bar2_height)
                                marker_offset = len(significance_markers) * 0.08 * max_height
                                base_height = max_height + 0.05 * max_height
                                bracket_height = base_height + marker_offset
                                line_height = bracket_height + 0.02 * max_height
                                x1 = bars[i].get_x() + bars[i].get_width() / 2
                                x2 = bars[j].get_x() + bars[j].get_width() / 2
                                ax2_second.plot([x1, x1, x2, x2], [bracket_height, line_height, line_height, bracket_height], 
                                            'k-', linewidth=1.5)
                                sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                                ax2_second.text((x1 + x2) / 2, line_height + 0.01 * max_height, sig_text, 
                                            ha='center', va='bottom', fontsize=14, fontweight='bold')
                                significance_markers.append({'x1': x1, 'x2': x2, 'height': line_height})
            
            labels = ax2_second.get_xticklabels()
            wrapped_labels = [textwrap.fill(label.get_text(), width=20) for label in labels]
            ax2_second.set_xticklabels(wrapped_labels)
            ax2_second.set_title('Subgoal RT - Second Half', fontsize=16, fontweight='bold')
            ax2_second.set_xlabel('Switching Pattern', fontsize=16, fontweight='bold')
            ax2_second.tick_params(axis='x', labelsize=14)
            ax2_second.tick_params(axis='y', labelsize=14)
            for label in ax2_second.get_xticklabels():
                label.set_weight('bold')
            for label in ax2_second.get_yticklabels():
                label.set_weight('bold')
            ax2_second.set_facecolor('white')
            ax2_second.set_ylim(0, 1200)
        else:
            ax2_second.set_title('No Subgoal RT Data Available', fontsize=16, weight='bold')
            ax2_second.text(0.5, 0.5, 'No subgoal RT data found', ha='center', va='center', 
                        transform=ax2_second.transAxes, fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(FIGURES_TOPICS + f"/goal_and_subgoal_rt_by_switching_pattern_by_block_half_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return goal_results_dict, subgoal_results_dict

    def plot_rt_increase_vs_goal_switches(self):
        """
        Plot the proportion of reaction time increase during goal switching 
        against the total number of goal switches made during the task.
        
        Returns:
            DataFrame with summary statistics for each subject
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create previous goal selections for comparison
        df['prev_goal'] = df.groupby(['subject_id', 'block_num'])['goal_selected'].shift(1)
        df['prev_goal'] = df['prev_goal'].replace('BR', 'OB')
        
        # Remove first trial of each block (no previous selection)
        df = df.dropna(subset=['prev_goal', 'goal_rt'])
        
        # Calculate metrics for each subject
        subject_metrics = []
        
        for subject in df['subject_id'].unique():
            subject_data = df[df['subject_id'] == subject].copy()
            
            if len(subject_data) == 0:
                continue
            
            # Identify goal switches (where current goal != previous goal)
            subject_data['is_goal_switch'] = (subject_data['goal_selected'] != subject_data['prev_goal'])
            
            # Calculate RT for goal switches vs non-switches
            switch_trials = subject_data[subject_data['is_goal_switch'] == True]
            stay_trials = subject_data[subject_data['is_goal_switch'] == False]
            
            if len(switch_trials) == 0 or len(stay_trials) == 0:
                continue
            
            # Calculate mean RT for switches and stays
            mean_rt_switch = switch_trials['goal_rt'].mean()
            mean_rt_stay = stay_trials['goal_rt'].mean()
            
            # Calculate proportion of RT increase
            if mean_rt_stay > 0:
                rt_increase_proportion = (mean_rt_switch - mean_rt_stay) / mean_rt_stay
            else:
                rt_increase_proportion = np.nan
            
            # Count total goal switches
            total_goal_switches = len(switch_trials)
            total_trials = len(subject_data)
            switch_proportion = total_goal_switches / total_trials
            
            subject_metrics.append({
                'subject_id': subject,
                'rt_increase_proportion': rt_increase_proportion,
                'total_goal_switches': total_goal_switches,
                'switch_proportion': switch_proportion,
                'mean_rt_switch': mean_rt_switch,
                'mean_rt_stay': mean_rt_stay,
                'total_trials': total_trials
            })
        
        if not subject_metrics:
            print("No valid data found for RT increase vs goal switches analysis")
            return None
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(subject_metrics)
        
        # Remove subjects with invalid RT increase calculations
        metrics_df = metrics_df.dropna(subset=['rt_increase_proportion'])
        
        if len(metrics_df) == 0:
            print("No valid RT increase data found")
            return None
        
        # Print summary statistics
        print(f"\nRT Increase vs Goal Switches Analysis (n={len(metrics_df)} subjects):")
        print(f"Mean RT increase proportion: {metrics_df['rt_increase_proportion'].mean():.3f} ± {metrics_df['rt_increase_proportion'].std():.3f}")
        print(f"Mean total goal switches: {metrics_df['total_goal_switches'].mean():.1f} ± {metrics_df['total_goal_switches'].std():.1f}")
        print(f"Mean switch proportion: {metrics_df['switch_proportion'].mean():.3f} ± {metrics_df['switch_proportion'].std():.3f}")
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot: RT increase proportion vs total goal switches
        ax.scatter(metrics_df['rt_increase_proportion'], metrics_df['total_goal_switches'], 
                   alpha=0.7, s=60, color='#2E86AB', edgecolors='black', linewidth=0.5)
        
        # Add trend line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            metrics_df['rt_increase_proportion'], metrics_df['total_goal_switches']
        )
        
        x_trend = np.linspace(metrics_df['rt_increase_proportion'].min(), 
                             metrics_df['rt_increase_proportion'].max(), 100)
        y_trend = slope * x_trend + intercept
        ax.plot(x_trend, y_trend, 'r--', linewidth=2, alpha=0.8)
        
        # Add correlation text
        ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('RT Increase Proportion', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Goal Switches', fontsize=14, fontweight='bold')
        ax.set_title('Total Goal Switches vs RT Increase Proportion', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add experiment label
        experiment = 'H2' if self.data_type == 'online' else 'H1'
        fig.suptitle(f'Goal Switching RT Analysis: Experiment {experiment}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(FIGURES_TOPICS + f"/rt_increase_vs_goal_switches_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical analysis
        print(f"\nStatistical Analysis:")
        print(f"Correlation between RT increase proportion and total goal switches: r={r_value:.3f}, p={p_value:.4f}")
        
        # Additional analysis: categorize subjects by switching behavior
        metrics_df['switching_category'] = pd.cut(metrics_df['total_goal_switches'], 
                                                 bins=3, labels=['Low', 'Medium', 'High'])
        
        print(f"\nRT Increase by Switching Category:")
        for category in ['Low', 'Medium', 'High']:
            cat_data = metrics_df[metrics_df['switching_category'] == category]
            if len(cat_data) > 0:
                print(f"{category} switchers (n={len(cat_data)}): "
                      f"RT increase = {cat_data['rt_increase_proportion'].mean():.3f} ± {cat_data['rt_increase_proportion'].std():.3f}")
        
        return metrics_df

    def plot_goal_rt_by_repetition(self):
        """
        Plot goal reaction time as a function of consecutive repetitions of the same goal-subgoal pair.
        Shows learning/speed-up effect with each repetition.
        
        Returns:
            DataFrame with summary statistics for each repetition number
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create goal-subgoal pair identifier
        df['goal_subgoal_pair'] = df['goal_selected'].astype(str) + '_' + df['subgoal_selected'].astype(str)
        
        # Calculate consecutive repetition number for each goal-subgoal pair
        df['repetition_number'] = 1  # Initialize all as 1
        
        # For each subject and block, calculate repetition numbers
        for subject in df['subject_id'].unique():
            subject_data = df[df['subject_id'] == subject]
            for block in subject_data['block_num'].unique():
                block_data = subject_data[subject_data['block_num'] == block]
                
                # Reset repetition counter for each new goal-subgoal pair
                current_pair = None
                repetition_count = 0
                
                for idx in block_data.index:
                    pair = block_data.loc[idx, 'goal_subgoal_pair']
                    
                    if pair == current_pair:
                        repetition_count += 1
                    else:
                        repetition_count = 1
                        current_pair = pair
                    
                    df.loc[idx, 'repetition_number'] = repetition_count
        
        # Filter to only include trials with valid goal_rt
        df_filtered = df.dropna(subset=['goal_rt', 'repetition_number']).copy()
        
        if len(df_filtered) == 0:
            print("No valid data found for repetition analysis")
            return None
        
        # Calculate statistics for each repetition number
        repetition_stats = []
        max_repetitions = min(df_filtered['repetition_number'].max(), 9)  # Limit to repetitions 1-9
        
        for rep_num in range(1, max_repetitions + 1):
            rep_data = df_filtered[df_filtered['repetition_number'] == rep_num]
            
            if len(rep_data) == 0:
                continue
            
            # Calculate mean RT for each subject, then average across subjects
            subject_means = []
            total_rt_all_subjects = 0
            total_trials_all_subjects = 0
            
            for subject in df_filtered['subject_id'].unique():
                subject_rep_data = rep_data[rep_data['subject_id'] == subject]
                
                if len(subject_rep_data) > 0:
                    subject_mean_rt = subject_rep_data['goal_rt'].mean()
                    subject_means.append(subject_mean_rt)
                    total_rt_all_subjects += subject_rep_data['goal_rt'].sum()
                    total_trials_all_subjects += len(subject_rep_data)
            
            if len(subject_means) > 0:
                mean_rt = np.mean(subject_means)
                sem_rt = np.std(subject_means) / np.sqrt(len(subject_means))
                overall_mean_rt = total_rt_all_subjects / total_trials_all_subjects if total_trials_all_subjects > 0 else np.nan
            else:
                mean_rt = np.nan
                sem_rt = np.nan
                overall_mean_rt = np.nan
            
            repetition_stats.append({
                'repetition_number': rep_num,
                'mean_rt': mean_rt,
                'sem_rt': sem_rt,
                'n_subjects': len(subject_means),
                'total_trials': total_trials_all_subjects,
                'overall_mean_rt': overall_mean_rt,
                'subject_rts': subject_means
            })
        
        if not repetition_stats:
            print("No valid repetition data found")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(repetition_stats)
        
        # Print summary
        print("\nGoal RT by repetition number:")
        for _, row in results_df.iterrows():
            print(f"Repetition {row['repetition_number']}: "
                  f"{row['mean_rt']:.3f} ± {row['sem_rt']:.3f} ms "
                  f"({row['total_trials']} trials, {row['n_subjects']} subjects)")
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot line with shaded SEM confidence interval
        ax.fill_between(results_df['repetition_number'], 
                       results_df['mean_rt'] - results_df['sem_rt'],
                       results_df['mean_rt'] + results_df['sem_rt'],
                       alpha=0.3, color='#B0B0B0', label='SEM')
        
        # Plot the main line with markers
        ax.plot(results_df['repetition_number'], results_df['mean_rt'], 
               marker='o', markersize=8, linewidth=3,
               color='black', markerfacecolor='black', markeredgecolor='black', 
               markeredgewidth=1, alpha=0.9, label='Mean RT')
        
        # Customize the plot
        ax.set_xlabel('Consecutive Repetition Number', fontsize=16, fontweight='bold')
        ax.set_ylabel('Goal Reaction Time (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Goal-Subgoal Repetition Effect on Goal RT', fontsize=18, fontweight='bold')
        
        # Set x-axis to show integer values
        ax.set_xticks(results_df['repetition_number'])
        ax.set_xlim(0.5, max_repetitions + 0.5)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Make tick labels bold
        ax.tick_params(axis='both', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_weight('bold')
        for label in ax.get_yticklabels():
            label.set_weight('bold')
        
        # Set clean white background
        ax.set_facecolor('white')
        
        # Add trend line
        from scipy import stats
        if len(results_df) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                results_df['repetition_number'], results_df['mean_rt']
            )
            
            x_trend = np.linspace(1, max_repetitions, 100)
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'k:', linewidth=2, alpha=0.7, label=f'Trend (r={r_value:.3f})')
            
            # Add correlation text
            ax.text(0.95, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                    transform=ax.transAxes, fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    ha='right', va='top')
        
        # Add title
        # fig.suptitle('Goal-Subgoal Repetition Effect', 
        #             fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(FIGURES_TOPICS + f"/goal_rt_by_repetition_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical analysis
        if len(results_df) > 1:
            print(f"\nStatistical Analysis:")
            print(f"Correlation between repetition number and goal RT: r={r_value:.3f}, p={p_value:.4f}")
            
            # Compare first vs last repetition
            if len(results_df) >= 2:
                first_rep = results_df.iloc[0]
                last_rep = results_df.iloc[-1]
                rt_improvement = first_rep['mean_rt'] - last_rep['mean_rt']
                improvement_pct = (rt_improvement / first_rep['mean_rt']) * 100
                
                print(f"\nRepetition Effect:")
                print(f"First repetition: {first_rep['mean_rt']:.1f} ± {first_rep['sem_rt']:.1f} ms")
                print(f"Last repetition: {last_rep['mean_rt']:.1f} ± {last_rep['sem_rt']:.1f} ms")
                print(f"Improvement: {rt_improvement:.1f} ms ({improvement_pct:.1f}% faster)")
        
        return results_df

    def plot_subgoal_rt_by_repetition(self):
        """
        Plot subgoal reaction time as a function of consecutive repetitions of the same goal-subgoal pair.
        Shows learning/speed-up effect with each repetition.
        
        Returns:
            DataFrame with summary statistics for each repetition number
        """
        df = self.df.copy()
        
        # Replace 'BR' with 'OB' for consistency
        df['goal_selected'] = df['goal_selected'].replace('BR', 'OB')
        
        # Ensure proper sorting
        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        
        # Create goal-subgoal pair identifier
        df['goal_subgoal_pair'] = df['goal_selected'].astype(str) + '_' + df['subgoal_selected'].astype(str)
        
        # Calculate consecutive repetition number for each goal-subgoal pair
        df['repetition_number'] = 1  # Initialize all as 1
        
        # For each subject and block, calculate repetition numbers
        for subject in df['subject_id'].unique():
            subject_data = df[df['subject_id'] == subject]
            for block in subject_data['block_num'].unique():
                block_data = subject_data[subject_data['block_num'] == block]
                
                # Reset repetition counter for each new goal-subgoal pair
                current_pair = None
                repetition_count = 0
                
                for idx in block_data.index:
                    pair = block_data.loc[idx, 'goal_subgoal_pair']
                    
                    if pair == current_pair:
                        repetition_count += 1
                    else:
                        repetition_count = 1
                        current_pair = pair
                    
                    df.loc[idx, 'repetition_number'] = repetition_count
        
        # Filter to only include trials with valid subgoal_rt
        df_filtered = df.dropna(subset=['subgoal_rt', 'repetition_number']).copy()
        
        if len(df_filtered) == 0:
            print("No valid data found for subgoal repetition analysis")
            return None
        
        # Calculate statistics for each repetition number
        repetition_stats = []
        max_repetitions = min(df_filtered['repetition_number'].max(), 6)  # Limit to repetitions 1-9
        
        for rep_num in range(1, max_repetitions + 1):
            rep_data = df_filtered[df_filtered['repetition_number'] == rep_num]
            
            if len(rep_data) == 0:
                continue
            
            # Calculate mean RT for each subject, then average across subjects
            subject_means = []
            total_rt_all_subjects = 0
            total_trials_all_subjects = 0
            
            for subject in df_filtered['subject_id'].unique():
                subject_rep_data = rep_data[rep_data['subject_id'] == subject]
                
                if len(subject_rep_data) > 0:
                    subject_mean_rt = subject_rep_data['subgoal_rt'].mean()
                    subject_means.append(subject_mean_rt)
                    total_rt_all_subjects += subject_rep_data['subgoal_rt'].sum()
                    total_trials_all_subjects += len(subject_rep_data)
            
            if len(subject_means) > 0:
                mean_rt = np.mean(subject_means)
                sem_rt = np.std(subject_means) / np.sqrt(len(subject_means))
                overall_mean_rt = total_rt_all_subjects / total_trials_all_subjects if total_trials_all_subjects > 0 else np.nan
            else:
                mean_rt = np.nan
                sem_rt = np.nan
                overall_mean_rt = np.nan
            
            repetition_stats.append({
                'repetition_number': rep_num,
                'mean_rt': mean_rt,
                'sem_rt': sem_rt,
                'n_subjects': len(subject_means),
                'total_trials': total_trials_all_subjects,
                'overall_mean_rt': overall_mean_rt,
                'subject_rts': subject_means
            })
        
        if not repetition_stats:
            print("No valid repetition data found")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(repetition_stats)
        
        # Print summary
        print("\nSubgoal RT by repetition number:")
        for _, row in results_df.iterrows():
            print(f"Repetition {row['repetition_number']}: "
                  f"{row['mean_rt']:.3f} ± {row['sem_rt']:.3f} ms "
                  f"({row['total_trials']} trials, {row['n_subjects']} subjects)")
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot line with shaded SEM confidence interval
        ax.fill_between(results_df['repetition_number'], 
                       results_df['mean_rt'] - results_df['sem_rt'],
                       results_df['mean_rt'] + results_df['sem_rt'],
                       alpha=0.3, color='#B0B0B0', label='SEM')
        
        # Plot the main line with markers
        ax.plot(results_df['repetition_number'], results_df['mean_rt'], 
               marker='o', markersize=8, linewidth=3,
               color='black', markerfacecolor='black', markeredgecolor='black', 
               markeredgewidth=1, alpha=0.9, label='Mean RT')
        
        # Customize the plot
        ax.set_xlabel('Consecutive Repetition Number', fontsize=16, fontweight='bold')
        ax.set_ylabel('Subgoal Reaction Time (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Goal-Subgoal Repetition Effect on Subgoal RT', fontsize=18, fontweight='bold')
        
        # Set x-axis to show integer values
        ax.set_xticks(results_df['repetition_number'])
        ax.set_xlim(0.5, max_repetitions + 0.5)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Make tick labels bold
        ax.tick_params(axis='both', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_weight('bold')
        for label in ax.get_yticklabels():
            label.set_weight('bold')
        
        # Set clean white background
        ax.set_facecolor('white')
        
        # Add trend line
        from scipy import stats
        if len(results_df) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                results_df['repetition_number'], results_df['mean_rt']
            )
            
            x_trend = np.linspace(1, max_repetitions, 100)
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'k:', linewidth=2, alpha=0.7, label=f'Trend (r={r_value:.3f})')
            
            # Add correlation text
            ax.text(0.95, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                    transform=ax.transAxes, fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    ha='right', va='top')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(FIGURES_TOPICS + f"/subgoal_rt_by_repetition_experiment_{self.experiment}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical analysis
        if len(results_df) > 1:
            print(f"\nStatistical Analysis:")
            print(f"Correlation between repetition number and subgoal RT: r={r_value:.3f}, p={p_value:.4f}")
            
            # Compare first vs last repetition
            if len(results_df) >= 2:
                first_rep = results_df.iloc[0]
                last_rep = results_df.iloc[-1]
                rt_improvement = first_rep['mean_rt'] - last_rep['mean_rt']
                improvement_pct = (rt_improvement / first_rep['mean_rt']) * 100
                
                print(f"\nRepetition Effect:")
                print(f"First repetition: {first_rep['mean_rt']:.1f} ± {first_rep['sem_rt']:.1f} ms")
                print(f"Last repetition: {last_rep['mean_rt']:.1f} ± {last_rep['sem_rt']:.1f} ms")
                print(f"Improvement: {rt_improvement:.1f} ms ({improvement_pct:.1f}% faster)")
        
        return results_df


if __name__ == "__main__":
    # Example usage
    plot_rt = PlotRTMeasures(data_type="online", read_from_csv=True)
    #goal_results, subgoal_results = plot_rt.plot_goal_rt_by_switching_pattern()
    
    # New method: RT by switching pattern split by block half
    #goal_results, subgoal_results = plot_rt.plot_goal_rt_by_switching_pattern_by_block_half()
    
    # New method: RT increase vs goal switches
    #rt_metrics = plot_rt.plot_rt_increase_vs_goal_switches()
    
    # New method: Goal RT by repetition
    #repetition_results = plot_rt.plot_goal_rt_by_repetition()
    
    # New method: Subgoal RT by repetition
    subgoal_repetition_results = plot_rt.plot_subgoal_rt_by_repetition()
