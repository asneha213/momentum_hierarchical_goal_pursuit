
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/")
from behavior_utils import *

from datautils import *


def get_behavioral_measure(subject_measure, measure_name):
    if measure_name == "optimal_goal":
        return subject_measure.get_optimal_goal_condition().T
    elif measure_name == "optimal_subgoal":
        return subject_measure.get_optimal_subgoal_condition().T
    elif measure_name == "characterize_subgoal_switches":
        return subject_measure.characterize_subgoal_switches()
    elif measure_name == "retro_valuation_factor":
        return subject_measure.get_retro_valuation_factor()
    elif measure_name == "action_goal_congruence":
        return subject_measure.get_goal_action_congruence()
    elif measure_name == "retro_goal_valuation_with_progress":
        return subject_measure.retro_goal_valuation_with_progress()
    elif measure_name == "goal_selection_in_relation_to_goal_progress":
        return subject_measure.goal_selection_in_relation_to_goal_progress()
    elif measure_name == "conditional_subgoal_selection_given_goal_in_relation_to_subgoal_progress":
        return subject_measure.subgoal_selection_in_relation_to_subgoal_progress(conditional=True)
    elif measure_name == "subgoal_selection_in_relation_to_subgoal_progress":
        return subject_measure.subgoal_selection_in_relation_to_subgoal_progress(conditional=False)
    elif measure_name == "characterize_subgoal_switches":
        return subject_measure.characterize_subgoal_switches()
    elif measure_name == "characterize_goal_subgoal_switches":
        return subject_measure.characterize_goal_and_subgoal_switches()



def get_aggregate_measure(experiment, measure_name, cache=False, data_type="fmri", np_array=True):
    cache_file_name = CACHE_DIR + measure_name + "_" + str(experiment) + ".npy"
    if cache:
        return np.load(cache_file_name)
    sub_measures = []
    if data_type == "fmri":
        subject_ids = ACTIVE_SUBJECT_IDS
    else:
        subject_ids = ACTIVE_SUBJECT_IDS_ONLINE
    for subject_id in subject_ids:
        #print(subject_id)
        subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=None, data_type=data_type)
        measure = get_behavioral_measure(subject_measures, measure_name)
        print(subject_id, measure)
        sub_measures.append(measure)
    if not np_array:
        return sub_measures
    sub_measures = np.array(sub_measures)
    np.save(cache_file_name, sub_measures)
    return sub_measures


class SubjectMeasure:
    def __init__(self, subject_id, experiment, model_res=None, data_type='fmri'):
        self.subject_id = subject_id
        self.experiment = experiment

        self.details = {
            "num_episodes": 9,
            "num_trials": 33,
            "num_conditions": 6
        }
        self.num_episodes = self.details["num_episodes"]
        self.num_trials = self.details["num_trials"]
        self.num_conditions = self.details["num_conditions"]
        self.data_type = data_type

        if model_res:
            self.data = model_res
        else:
            self.data = get_subject_data_from_id(experiment, subject_id, data_type=data_type)
        self.blocks = self.data['GoalSwitching']

    def get_task_performance(self):
        total_rewards = 0

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]

            if 'reward_trials' not in block:
                continue
            num_rewards = len(block['reward_trials'])
            total_rewards += num_rewards

        return total_rewards

    def get_task_performance_fmri(self):
        score = 0

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            if self.data_type == "fmri":
                trial_list = [key for key in block.keys() if "trial_" in key]
                for trial_str in trial_list:
                    trial = block[trial_str]
                    if self.data_type == 'fmri':
                        score = trial['score']
            else:
                if 'reward_trials' not in block:
                    continue
                num_rewards = len(block['reward_trials'])
                score += num_rewards

        return score

    def get_optimal_goal_condition(self):
        goalselects = np.zeros((self.num_conditions, 3))
        counts = np.zeros(self.num_conditions)

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                if "probe_selected" not in trial and 'goal_selected' not in trial:
                    continue
                counts[condition] += 1
                if 'probe_selected' in trial:
                    goalselect = trial["probe_selected"]
                elif 'goal_selected' in trial:
                    goalselect = trial["goal_selected"]

                if trial["goal_selected"] is None:
                    continue
                goalindex = ['SH', 'BR', 'HO'].index(goalselect)
                goalselects[condition][goalindex] += 1

        goalselects = goalselects / counts[:, None]

        return goalselects

    def get_optimal_subgoal_condition(self):
        goalselects = np.zeros((self.num_conditions, 4))
        counts = np.zeros(self.num_conditions)

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]

                if trial['action_selected'] is not None:
                    subgoalselect = trial["action_selected"]
                else:
                    continue
                counts[condition] += 1

                goalindex = ['G', 'C', 'M', 'S'].index(subgoalselect)
                goalselects[condition][goalindex] += 1

        goalselects = goalselects / counts[:, None]

        return goalselects

    def get_optimal_goal_condition_cond(self):
        goalselects = np.zeros((2, 3))

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                if condition < 3:
                    cond = 0
                else:
                    cond = 1
                goalselect = trial["goalindex"]
                goalselects[cond][goalselect] += 1

        return goalselects

    def get_goal_switches_condition(self):
        goal_switches = np.zeros(self.num_conditions)
        counts = np.zeros(self.num_conditions)
        prev_goal = None
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                if "probe_selected" not in trial and 'goal_selected' not in trial:
                    continue
                counts[condition] += 1
                if 'probe_selected' in trial:
                    goalselect = trial["probe_selected"]
                elif 'goal_selected' in trial:
                    goalselect = trial["goal_selected"]
                if prev_goal:
                    if prev_goal != goalselect:
                        goal_switches[condition] += 1
                prev_goal = goalselect

        return goal_switches

    def get_subgoal_switches_condition(self):
        goal_switches = np.zeros(self.num_conditions)
        counts = np.zeros(self.num_conditions)
        prev_goal = None
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                if "action_selected" not in trial:
                    continue
                counts[condition] += 1
                goalselect = trial["action_selected"]
                if prev_goal:
                    if prev_goal != goalselect:
                        goal_switches[condition] += 1
                prev_goal = goalselect
        return goal_switches


    def characterize_goal_and_subgoal_switches(self):
        """
        Character 0: Staying with the same goal and subgoal
        Character 1: Staying with the same goal and switching subgoal
        Character 2: Switching goal and staying with the same subgoal
        Character 3: Switching goal and switching subgoal to a relevant alternative goal
        Character 4: Switching goal and switching subgoal to an irrelevant alternative goal
        Character 5: Goal-incongruent action
        """

        switching_char = np.zeros(7)
        prev_goal = None
        prev_subgoal = None
        action_outcome = None

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                if "goal_selected" not in trial or "action_selected" not in trial:
                    continue
                if trial["goal_selected"] is None or trial["action_selected"] is None:
                    continue

                goal = trial["goal_selected"]
                subgoal = trial["action_selected"]
                condition = trial["block_type"]
                goal_progress = trial["goal_progress_pre"]

                if prev_goal is None:
                    prev_goal = goal
                    prev_subgoal = subgoal
                    action_outcome = trial["action_outcome"]
                    continue


                if subgoal not in goal_progress[goal]:
                    rel_subgoal = False
                elif goal_progress[goal][subgoal][0] == goal_progress[goal][subgoal][1]:
                    rel_subgoal = False
                else:
                    rel_subgoal = True

                # Check if staying with the same goal and subgoal and rel_subgoal True
                if prev_goal == goal and prev_subgoal == subgoal and rel_subgoal:
                    switching_char[0] += 1

                elif prev_goal == goal and prev_subgoal != subgoal and rel_subgoal:
                    switching_char[1] += 1

                elif prev_goal == goal and not rel_subgoal:
                    switching_char[2] += 1

                elif prev_goal != goal and prev_subgoal == subgoal and rel_subgoal:
                    # Check if the subgoal is relevant to the new goal
                    switching_char[3] += 1

                elif prev_goal != goal and prev_subgoal != subgoal and rel_subgoal:
                    relevant_alt_subgoals = get_relevant_alternative_subgoals_for_subgoals()
                    if (prev_goal, prev_subgoal) in relevant_alt_subgoals:
                        if subgoal in relevant_alt_subgoals[(prev_goal, prev_subgoal)]:
                            switching_char[4] += 1
                        else:
                            switching_char[5] += 1

                elif prev_goal != goal and prev_subgoal != subgoal and not rel_subgoal:
                    switching_char[6] += 1


                prev_goal = goal
                prev_subgoal = subgoal

        return switching_char




    def characterize_subgoal_switches(self):
        """
        Character 0: Staying with the subgoal
        Character 1: Switching subgoal within a goal (no goal switching)
            Switching to subgoal that is congruent with the goal specified in the probe

        Character 2: Switching subgoal to relevant alternative goal
            Pursuing a subgoal that is congruent with two goals.
            Rate of switching over to a subgoal that is alternative to the goal specified in the probe

        Character 3: Switching subgoal to irrelevant alternative goal
        """

        def get_consecutive_subgoal_relations():
            # Checking for Character 1
            goal_congruent_alt_subgoal = get_relevant_congruent_subgoals_for_subgoal()
            if (prev_goal, prev_subgoal) not in goal_congruent_alt_subgoal:
                return -1
            if subgoal in goal_congruent_alt_subgoal[(prev_goal, prev_subgoal)]:
                if goal_progress[prev_goal][subgoal][0] < goal_progress[prev_goal][subgoal][1]:
                    return 1
            # Checking for Character 2 and 3
            relevant_alt_goal_subgoal = get_relevant_alternative_subgoals_for_subgoals()
            if (prev_goal, prev_subgoal) not in relevant_alt_goal_subgoal:
                return -1
            if subgoal in relevant_alt_goal_subgoal[(prev_goal, prev_subgoal)]:
                if subgoal in goal_progress[prev_goal]:
                    if goal_progress[prev_goal][subgoal][0] < goal_progress[prev_goal][subgoal][1]:
                        # condition not encountered
                        return 1
                    else:
                        return 2
                else:
                    return 2
            else:
                # If switched to irrelevant goal in perspective of the current subgoal
                return 3

        subgoal_char = np.zeros(4)
        prev_goal = None
        prev_subgoal = None
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]

                if "goal_selected" not in trial or "action_selected" not in trial:
                    continue

                if trial["goal_selected"] is None or trial["action_selected"] is None:
                    continue

                goal = trial["goal_selected"]
                subgoal = trial["action_selected"]
                condition = trial["block_type"]
                goal_progress = trial["goal_progress_pre"]

                if prev_goal is None:
                    prev_goal = goal
                    prev_subgoal = subgoal
                    continue
                action_rel = get_action_relevance_in_goal(prev_goal, prev_subgoal, trial["resources_pre"])
                # if action_rel == 0:
                #     continue

                #print(subgoal, prev_subgoal)

                if subgoal != prev_subgoal:
                    char = get_consecutive_subgoal_relations()
                    if char != -1:
                        subgoal_char[char] += 1
                else:
                    if subgoal in goal_progress[prev_goal]:
                        if goal_progress[prev_goal][subgoal][0] < goal_progress[prev_goal][subgoal][1]:
                            subgoal_char[0]+= 1
                        else:
                            subgoal_char[3] += 1

                prev_goal = goal
                prev_subgoal = subgoal

        return subgoal_char



    def get_goal_selections_condition(self):
        goal_selects = np.zeros((self.num_conditions, 3))
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                if "probe_selected" not in trial and "goal_selected" not in trial:
                    continue
                if "probe_selected" in trial:
                    goalselect = ["SH", "HO", "BR"].index(trial["probe_selected"])
                elif "goal_selected" in trial:
                    goalselect = ["SH", "HO", "BR"].index(trial["goal_selected"])
                goal_selects[condition][goalselect] += 1
        return goal_selects

    def get_action_selections_condition(self):
        action_selects = np.zeros((self.num_conditions, 4))
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                if "action_selected" not in trial and 'resource_selected' not in trial:
                    continue
                if "resource_selected" in trial:
                    actionselect = ['G', 'C', 'M', 'S'].index(trial["resource_selected"])
                elif "action_selected" in trial:
                    actionselect = ['G', 'C', 'M', 'S'].index(trial["action_selected"])
                action_selects[condition][actionselect] += 1
        return action_selects

    """
    Goal and subgoal selection statistics   
    """

    def retro_goal_valuation_with_progress(self):
        goal_selections = np.zeros((self.num_conditions, 6))
        counts = np.zeros((self.num_conditions, 6))

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                goal_progress = trial["goal_progress_pre"]
                progresses = {'SH': 0, 'BR': 0, 'HO': 0}
                for gkey in goal_progress.keys():
                    progresses[gkey] = np.sum([goal_progress[gkey][key][0] for key in goal_progress[gkey].keys()])
                goals = list(progresses.keys())

                goal_selected = trial["goal_selected"]
                if goal_selected is None:
                    continue

                max_progress_goal = np.argmax([progresses[goal] for goal in goals])
                max_progress = progresses[goals[max_progress_goal]]

                counts[condition][max_progress] += 1
                if goal_selected == goals[max_progress_goal]:
                    goal_selections[condition][max_progress] += 1

        return goal_selections / counts


    def goal_selection_in_relation_to_goal_progress(self):

        """
        Goal selection in relation to goal progress
            
        return: Propotion of goal selections in relation to goal progress as a 3D array
        ((num_goals, num_conditions, num_progresses(7)))

        """
        
        goal_selections = np.zeros((3, self.num_conditions, 6))
        counts = np.zeros((3, self.num_conditions, 6))

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                goal_progress = trial["goal_progress_pre"]
                progresses = {'SH': 0, 'BR': 0, 'HO': 0}
                for gkey in goal_progress.keys():
                    progresses[gkey] = np.sum([goal_progress[gkey][key][0] for key in goal_progress[gkey].keys()])
                goals = list(progresses.keys())

                if "goal_selected" not in trial:
                    continue

                goal_selected = trial["goal_selected"]
                if goal_selected is None:
                    continue

                for goal in goals:
                    goal_index = goals.index(goal)
                    if progresses[goal] > 5:
                        continue
                    counts[goal_index][condition][progresses[goal]] += 1
                    if goal_selected == goal:
                        goal_selections[goal_index][condition][progresses[goal]] += 1

        return goal_selections/ counts


    def subgoal_selection_in_relation_to_subgoal_progress(self, conditional=False):
        """
        Get subgoal selection in relation to subgoal progress
        return: Propotion of goal selections in relation to goal progress as a 3D array
        ((num_goals, num_conditions, num_progresses(3)))

        conditional: If True, get conditional subgoal selection given goal in relation to subgoal progress
                    If False, get subgoal selection in relation to subgoal resource progress (out of 3 max resources allowed)

        """
        goal_selections = np.zeros((4, self.num_conditions, 3))
        counts = np.zeros((4, self.num_conditions, 3))
        goals = ['G', 'C', 'M', 'S']

        def get_relevant_subgoals(goal_selected):
            if goal_selected == "SH":
                return ['M', 'G']
            elif goal_selected == "BR":
                return ['S']
            elif goal_selected == "HO":
                return ['C']

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                goal_progress = trial["goal_progress_pre"]
                # Nested dict of subgoals with keys as goals and values as subgoal progress in each goal
                subgoal_progress_dict = get_subgoal_progress_given_goal_progress(goal_progress)
                if "goal_selected" not in trial:
                    continue
                goal_selected = trial["goal_selected"]
                if goal_selected is None:
                    continue

                if trial['action_selected'] not in goal_progress[goal_selected]:
                    continue

                # Get the relevant subgoals for the selected goal
                relevant_subgoals = get_relevant_subgoals(goal_selected)
                subgoal_selected = trial["action_selected"]

                if subgoal_selected is None:
                    continue

                if conditional:
                    progress_keys = {
                    0: 0, 0.3: 1, 0.7: 2
                    }
                    for goal in relevant_subgoals:
                        goal_index = goals.index(goal)
                        subgoal_progress = round(subgoal_progress_dict[goal_selected][goal], 1)
                        if subgoal_progress == 1:
                            continue
                        counts[goal_index][condition][progress_keys[subgoal_progress]] += 1
                        if subgoal_selected == goal:
                            goal_selections[goal_index][condition][progress_keys[subgoal_progress]] += 1
                else:
                    progress_keys = {
                        0: 0, 0.3: 1, 0.7: 2
                    }
                    for goal in goals:
                        goal_index = goals.index(goal)
                        subgoal_progress = round(trial["resources_pre"][goal] / 3, 1)
                        if subgoal_progress == 1:
                            continue
                        counts[goal_index][condition][progress_keys[subgoal_progress]] += 1
                        if subgoal_selected == goal:
                            goal_selections[goal_index][condition][progress_keys[subgoal_progress]] += 1

        return goal_selections / counts

    def get_retro_valuation_factor(self, mode="slot", half=None, diverge=True):
        retro_choices = np.zeros((3, self.num_conditions))
        retro_counts = np.zeros(self.num_conditions)

        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                condition = trial["block_type"]
                if condition == 0 or condition == 3:
                    pros_choice = "SH"
                elif condition == 1 or condition == 4:
                    pros_choice = "BR"
                elif condition == 2 or condition == 5:
                    pros_choice = "HO"

                goal_progress = trial["goal_progress_pre"]

                pfs = {'SH': 0, 'BR': 0, 'HO': 0}
                for gkey in goal_progress.keys():
                    progress_fraction = np.sum([goal_progress[gkey][key][0] for key in goal_progress[gkey].keys()]) / 6
                    pfs[gkey] = progress_fraction

                # get key of max pf
                retro_choice = max(pfs, key=pfs.get)
                if pros_choice == retro_choice:
                    continue

                goal_selected = trial["goal_selected"]
                if goal_selected is None:
                    continue

                retro_counts[condition] += 1

                if goal_selected == retro_choice:
                    retro_choices[0][condition] += 1
                elif goal_selected == pros_choice:
                    retro_choices[1][condition] += 1
                else:
                    retro_choices[2][condition] += 1

        return retro_choices/retro_counts

    def get_rts(self, rt_type="goal"):
        rts = []
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            for trial_num in range(self.num_trials):
                trial = block["trial_" + str(trial_num)]
                if rt_type == "goal":
                    if trial["probe_rt"] is not None:
                        rts.append(trial["probe_rt"])
                    else:
                        rts.append(np.nan)
                else:
                    if trial["action_rt"] is not None:
                        rts.append(trial["action_rt"])
                    else:
                        rts.append(np.nan)
        return rts

    def get_goal_action_values_timeseries(self):
        goal_values = []
        goal_selects = []
        action_values = []
        action_selects = []
        goal_switches = []
        switch_probs = []
        conditions = []
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            for trial_num in range(self.num_trials):
                trial = block["trial_" + str(trial_num)]
                condition = trial["block_type"]
                actionselect = ['g', 'c', 'm', 's'].index(trial["action_selected"].lower())
                action_selects.append(actionselect)
                action_values.append(trial["action_vals"])
                gvals = trial["goal_vals"]
                goal_values.append(gvals)
                goal_selects.append(trial["goalindex"])
                goal_switches.append(trial["switch_goal"])
                switch_probs.append(trial["switch_probs"])
                conditions.append(condition)

        return np.array(goal_values), np.array(goal_selects), \
            np.array(action_values), np.array(action_selects), \
            np.array(goal_switches), np.array(switch_probs), \
            np.array(conditions)


    def get_goal_action_congruence(self, chance=False):
        counts = 0
        relevances = 0
        chance_relevance = []
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                # if 'resources' in trial:
                #     resources = trial["resources"]
                # else:
                goal_progress = trial["goal_progress_pre"]
                if "probe_selected" not in trial and 'goal_selected' not in trial:
                    continue

                if "probe_selected" in trial:
                    probe_selected = trial["probe_selected"]
                elif "goal_selected" in trial:
                    probe_selected = trial["goal_selected"]


                if self.data_type == "fmri":
                    if trial["action_keypress"] is None:
                        continue

                    action_keypress = trial['action_keypress']
                    key = {'1': '37', '2': '38', '3': '39', '4': '40'}

                    action_selected = trial[key[action_keypress[0]]]
                elif self.data_type == "online":
                    action_selected = trial["resource_selected"]


                relevance = get_action_relevance_from_goal_progress(probe_selected, action_selected, goal_progress)
                num_relevances = get_num_relevant_actions_goal_progress(probe_selected, goal_progress)
                chance_relevance.append(num_relevances / 4.0)
                counts += 1
                if relevance:
                    relevances += 1
        if chance:
            return np.mean(chance_relevance)
        else:
            return relevances / counts


    def characterize_goal_action_incongruence(self):
        """
        Characterize goal-action incongruence
        Returns a 2D array with the following columns:
        0: Goal selected
        1: Action selected
        2: Action relevance (0 - incongruent, 1 - congruent)
        3: Action outcome (0 - failure, 1 - success)
        """
        counts_incongruences = 0
        resource_excess = 0
        
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            trial_list = [key for key in block.keys() if "trial_" in key]
            for trial_str in trial_list:
                trial = block[trial_str]
                if "probe_selected" not in trial and 'goal_selected' not in trial:
                    continue

                if "probe_selected" in trial:
                    probe_selected = trial["probe_selected"]
                elif "goal_selected" in trial:
                    probe_selected = trial["goal_selected"]

                if trial["action_keypress"] is None:
                    continue
                
                resource_selected = trial['resource_selected']
                
                if resource_selected in trial["goal_progress_pre"][probe_selected]:
                    if trial["resources_pre"][resource_selected] == 3:
                        counts_incongruences += 1
                        resource_excess += 1
        #         else:
                    
                    
                        
                
                
                
                

        #         action_keypress = trial['action_keypress']
        #         key = {'1': '37', '2': '38', '3': '39', '4': '40'}

        #         action_selected = trial[key[action_keypress[0]]]

        #         relevance = get_action_relevance_from_goal_progress(probe_selected, action_selected, trial["resources_pre"])
        #         action_outcome = int(trial["action_outcome"] == "success")
                
        #         incongruence.append([probe_selected, action_selected, relevance, action_outcome])

        # return np.array(incongruence)
        


