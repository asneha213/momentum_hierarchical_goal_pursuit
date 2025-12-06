from scipy.special import softmax

import numpy as np
import copy



class Model:
    def __init__(self, params):
        self.params = params
        self.reward = 10
        self.resource_limit = 3

        self.prev_goal = ''
        self.prev_subgoal = ''

        # Goal-selection parameters
        self.alpha_c = None
        self.beta_g = None
        self.beta_0 = None
        
        # subgoal-selection parameters
        self.alpha_ca = None
        self.beta_a = None
        self.beta_0a = None


        self.current_time = 0
        self.goal_progress = {
            "SH": {"G": [0, 3], "M": [0, 3]},
            "BR": {"G": [0, 2], "C": [0, 1], "S": [0, 3]},
            "HO": {"C": [0, 3], "M": [0, 2], "S": [0, 1]}
        }

        self.reset_slots()
        #self.reset_card_probs()

    def reset_slots(self):

        self.goal_progress = {
            "SH": {"G": [0, 3], "M": [0, 3]},
            "BR": {"G": [0, 2], "C": [0, 1], "S": [0, 3]},
            "HO": {"C": [0, 3], "M": [0, 2], "S": [0, 1]}
        }
        self.resources = {
            'G': 0,
            'C': 0,
            'M': 0,
            'S': 0,
        }
    def reset_card_probs(self):
        self.C = {'SH': 0.5, 'HO': 0.5, 'BR': 0.5}
        self.C_sub = {'G': 0.25, 'C': 0.25, 'S': 0.25, 'M': 0.25}
        self.Cp = {'SH': 0.5, 'HO': 0.5, 'BR': 0.5}
        self.Cp_sub = {'G': 0.5, 'C': 0.5, 'S': 0.5, 'M': 0.5}


    def calculate_qvals_goals(self):
        return {'SH': 0.5, 'HO': 0.5, 'BR': 0.5}

    def calculate_qvals_subgoals(self, goal):
        return {'G': 0.25, 'C': 0.25, 'S': 0.25, 'M': 0.25}


    """
        Update functions after flipping a card
        1. Update card probabilities based on the outcome of the flip.
        2. Update goal and subgoal persevation based on the outcome of the flip.
        3. Update resources based on the outcome of the flip.
        4. Update goal progress based on the outcome of the flip.
        5. Check if the goal is completed based on the outcome of the flip.
    """

    def update_card_probs(self, goal, card, flip):
        return [None, None]

    def update_goal_perseveration(self, goal):

        self.C[goal] += self.alpha_c * (1 - self.C[goal])
        for poss_goal in ['SH', 'HO', 'BR']:
            if poss_goal != goal:
                self.C[poss_goal] += self.alpha_c * (0 - self.C[poss_goal])

        # self.C[goal] += 0.1 * (1 - self.C[goal])
        # for poss_goal in ['SH', 'HO', 'BR']:
        #     if poss_goal != goal:
        #         self.C[poss_goal] += 0.1 * (0 - self.C[poss_goal])

    def update_subgoal_perseveration(self, subgoal):

        self.Cp_sub[subgoal] += 0.1 * (1 - self.Cp_sub[subgoal])
        for poss_goal in ["G", "C", "S", "M"]:
            if poss_goal != subgoal:
                self.Cp_sub[poss_goal] += 0.1 * (0 - self.Cp_sub[poss_goal])

    def update_resources(self, resources, trial, resource):
        outcome = trial[resource + "_flip"]
        if (outcome == 1) and (resources[resource] < 3):
            resources[resource] += 1
        return resources

    def update_goal_progress(self, goal_progress, resources):
        for goal in goal_progress:
            for key in goal_progress[goal]:
                goal_progress[goal][key][0] = min(resources[key],
                                                  goal_progress[goal][key][1])
        return goal_progress

    def get_goal_completion_status(self, goal_progress, goal, resources):
        if goal is None:
            return False, goal_progress, resources
        gSH = goal_progress[goal]
        for key in gSH:
            if gSH[key][0] < gSH[key][1]:
                return False, goal_progress, resources

        for key in gSH:
            resources[key] -= gSH[key][1]

        goal_progress = self.update_goal_progress(goal_progress, resources)
        return True, goal_progress, resources

    def get_goal_progress(self, goal):
        goal_progress = self.goal_progress[goal]
        progress = 0
        for key in goal_progress.keys():
            progress += goal_progress[key][0]
        progress = progress / 6
        return progress

        """Goal and subgoal selection methods.
            1. Choose a goal based on the Q-values of the goals.
            2. Choose a subgoal based on the Q-values of the subgoals.
            3. Choose a subgoal of an active goal at random if no subgoal is selected.
        """

    def get_subgoal_progress(self, goal):
        goal_progress = self.goal_progress[goal]
        subgoal_progress = {}
        for subgoal in goal_progress.keys():
            subgoal_count = goal_progress[subgoal][0]
            subgoal_max = goal_progress[subgoal][1]
            if subgoal_count == subgoal_max:
                continue
            progress = subgoal_count / subgoal_max
            subgoal_progress[subgoal] = progress
        return subgoal_progress

    def choose_goal(self, beta_g, prev_goal, return_switch_probs=False):
        q_vals = self.calculate_qvals_goals()
        q_probs = softmax(beta_g * np.array([q_vals['SH'], q_vals['HO'], q_vals['BR']]))
        goal_select_probs = {'SH':0, 'HO':0, 'BR':0}

        # If previous goal is not set, or is 'u', choose a goal at random
        if (prev_goal == '') or (prev_goal == 'u'):
            goal = np.random.choice(['SH', 'HO', 'BR'], p=q_probs)
            return goal, {'SH': q_probs[0], 'HO': q_probs[1], 'BR': q_probs[2]}, q_vals

        cons = [token for token in ['SH', 'HO', 'BR'] if token != prev_goal]
        q_cons = np.array([q_vals[token] for token in cons])
        pstay = 1. / (1 + np.exp(self.beta_0 - self.C[prev_goal] - beta_g * (q_vals[prev_goal] - np.max(q_cons))))
        goal_select_probs[prev_goal] = pstay
        
        if return_switch_probs:
            return [pstay, 1 - pstay]

        q_cons_probs = softmax(beta_g * q_cons)

        for i, token in enumerate(cons):
            goal_select_probs[token] = q_cons_probs[i] * (1 - pstay)

        goal = np.random.choice(['SH', 'HO', 'BR'], p=[goal_select_probs['SH'], goal_select_probs['HO'], goal_select_probs['BR']])

        return goal, goal_select_probs, q_vals

    def choose_subgoal(self, goal):
        q_vals = self.calculate_qvals_subgoals(goal)

        active_subgoals = list(q_vals.keys())
        subgoal_select_probs = {'G': 0.01, 'C': 0.01, 'S': 0.01, 'M': 0.01}
     
        if len(active_subgoals) == 0:
            active_subgoals = []
            for subgoal in ['G', 'C', 'S', 'M']:
                if self.resources[subgoal] < self.resource_limit:
                    active_subgoals.append(subgoal)
            return np.random.choice(active_subgoals), subgoal_select_probs, q_vals


        # If no previous subgoal, choose one at random
        if self.prev_subgoal == '':
            subgoal = np.random.choice(['G', 'C', 'M', 'S'])
            return subgoal, {'G': 0.25, 'C': 0.25, 'S': 0.25, 'M': 0.25}, q_vals

        # You haven't switched goals, so you can choose to stay with the previous subgoal or switch
        if goal == self.prev_goal:
            # If there is only one active subgoal in the goal, select it with high chance
            if len(active_subgoals) == 1:
                subgoal_select_probs[active_subgoals[0]] = 0.97
                subgoal = active_subgoals[0]
                return subgoal, subgoal_select_probs, q_vals

            # If there are multiple active subgoals in the goal
            # If the previous subgoal is still active, we can choose to stay or switch
            if self.prev_subgoal in active_subgoals:
                cons = [token for token in active_subgoals if token != self.prev_subgoal]
                q_cons = np.array([q_vals[token] for token in cons])

                pstay = 1. / (1 + np.exp(
                    self.beta_0a  - self.beta_a * (q_vals[self.prev_subgoal] - np.max(q_cons))))

                subgoal_select_probs[self.prev_subgoal] = pstay
                q_cons_probs = softmax(self.beta_a * q_cons)
                for i, token in enumerate(cons):
                    subgoal_select_probs[token] = q_cons_probs[i] * (1 - pstay)

                subgoal = np.random.choice(active_subgoals, p=[subgoal_select_probs[token] for token in active_subgoals])

                return subgoal, subgoal_select_probs, q_vals
        
        # If the previous subgoal is no longer active
        # Or if you switched goals
        # then it's not a choice between staying or switching, but rather a choice among active subgoals
        q_active = np.array([q_vals[token] for token in active_subgoals])

        subgoal_probs = softmax(self.beta_a * q_active)
        subgoal = np.random.choice(active_subgoals, p=[subgoal_probs[i] for i in range(len(active_subgoals))])

        for i, token in enumerate(active_subgoals):
            subgoal_select_probs[token] = subgoal_probs[i]

        #print("Subgoal selection probs for goal:", subgoal_select_probs, subgoal)
        return subgoal, subgoal_select_probs, q_vals
                

    def choose_active_subgoal_at_random(self, goal):
        active_subgoals = []

        for subgoal in self.goal_progress[goal].keys():
            if self.goal_progress[goal][subgoal][0] < self.goal_progress[goal][subgoal][1]:
                active_subgoals.append(subgoal)

        if len(active_subgoals) == 0:
            subgoal_choice = np.random.choice(['G', 'S', 'C', 'M'])
        else:
            subgoal_choice = np.random.choice(active_subgoals)
        return subgoal_choice, -1, -1, -1


        """Likelihood functions for the model.
        These functions calculate the likelihood of the model's choices based on the trial data.
        """


    def run_subject_action_trial(self, trial_data, goal=True, subgoal=True, trial_dict=None):
        """
        Get subject choice probabiltiies based on the trial data.
        :param trial_data: Dictionary containing the trial data.
        :param subgoal: Boolean indicating whether to run subgoal selection.
        :return: Updated trial data, goal completion status, and selected goal probability."""
        
        trial = trial_data.copy()

        # Update resources and goal progress status from the previous trial
        self.resources = copy.deepcopy(trial["resources_pre"])
        self.goal_progress = copy.deepcopy(trial["goal_progress_pre"])

        self.block_type = trial['block_type']

        if "goal_selected" not in trial:
            return -1, -1, -1, -1

        if trial["goal_selected"] is None or trial["action_selected"] is None:
            if trial_dict is not None:
                return trial, -1, -1, -1, trial_dict
            return -1, -1, -1, -1

        # Choose goal, estimate goal stay probability, and update goal preservation

        goal_sub = trial["goal_selected"]
        if goal:
        # Choose goal, estimate goal stay probability, and update goal preservation
            goal_m, goal_select_probs, q_vals = self.choose_goal(self.beta_g, self.prev_goal)
            trial["goal_selected_model"] = goal_m
            #print("Goal selection probs for goal:", goal_select_probs)

            goal_select_prob = goal_select_probs[goal_sub]

            ## Record goal selection probabilities in the trial dictionary
            if trial_dict:
                trial_dict["chosen_goal_value"] = q_vals[goal_sub]
                trial_dict["chosen_goal_progress"] = self.get_goal_progress(goal_sub)
                trial_dict["chosen_goal_prob"] = goal_select_probs[goal_sub]
                trial_dict["chosen_goal_perseveration"] = self.C[goal_sub]

                trial_dict['SH_prob'] = goal_select_probs['SH']
                trial_dict['HO_prob'] = goal_select_probs['HO']
                trial_dict['BR_prob'] = goal_select_probs['BR']

                trial_dict['SH_value'] = q_vals['SH']
                trial_dict['HO_value'] = q_vals['HO']
                trial_dict['BR_value'] = q_vals['BR']

                trial_dict['SH_pers'] = self.Cp['SH']
                trial_dict['HO_pers'] = self.Cp['HO']
                trial_dict['BR_pers'] = self.Cp['BR']

                cons = [token for token in ['SH', 'HO', 'BR'] if token != goal_sub]
                q_cons = np.array([q_vals[token] for token in ['SH', 'HO', 'BR'] if token != goal_sub])
                trial_dict["alt_goal_value"] = np.max(q_cons)
                trial_dict['alt_goal_selected'] = cons[np.argmax(q_cons)]
                trial_dict['alt_goal_progress'] = self.get_goal_progress(trial_dict['alt_goal_selected'])
                if self.prev_goal != '':
                    trial_dict["switching_costs"] = - self.beta_0 + self.C[self.prev_goal] 
                    #trial_dict["momentum_advantage"] = self.beta_g * (q_vals[self.prev_goal] - np.max(q_cons))
                    trial_dict["momentum_advantage"] = (q_vals[self.prev_goal] - np.max(q_cons))

                if self.model_name == "momentum":
                    trial_dict['chosen_goal_progress_rate'] = self.w_g[goal_sub]
                    trial_dict['alt_goal_progress_rate'] = self.w_g[trial_dict['alt_goal_selected']]

                q_cons_prob = np.array([goal_select_probs[token] for token in ['SH', 'HO', 'BR'] if token != goal_sub])
                trial_dict["alt_goal_prob"] = np.max(q_cons_prob)

                if goal_sub == self.prev_goal:
                    trial_dict["goal_switch"] = 0
                    trial_dict["goal_stay"] = 1
                else:
                    trial_dict["goal_switch"] = 1
                    trial_dict["goal_stay"] = 0

            self.update_goal_perseveration(goal_sub)
        else:
            # Assign goal based on the trial data
            goal_select_prob = -1

    
        subgoal_sub = trial["resource_selected"]
        
        
        if subgoal:
            
            subgoal_select_probs_dict = {'G': 0, 'C': 0, 'S': 0, 'M': 0}
            subgoal_select_values_dict = {'G': 0, 'C': 0, 'S': 0, 'M': 0}

            if trial_dict is not None:
                subgoals_to_iterate = ['G', 'C', 'S', 'M']
            else:
                subgoals_to_iterate = [subgoal_sub]
            goal_m, goal_select_probs_subgoal, q_vals = self.choose_goal(self.beta_g, goal_sub, return_switch_probs=False)
            for subgoal_key in subgoals_to_iterate:
                subgoal_select_prob_key = 0
                # Should I assume goal switching for exploration happens at a different rate?
                # Choose subgoal, estimate subgoal stay probability, and update subgoal preservation
                for goal_given in ['SH', 'BR', 'HO']:
                    subgoal_m, subgoal_select_probs, subgoal_values = self.choose_subgoal(goal_given)
                    #print(goal_given, subgoal_select_probs, self.resources)
                    subgoal_select_prob_key += subgoal_select_probs[subgoal_key] * goal_select_probs_subgoal[goal_given]
                subgoal_select_probs_dict[subgoal_key] = subgoal_select_prob_key
            subgoal_select_prob = subgoal_select_probs_dict[subgoal_sub]


            ## Record latent-goal integrated subgoal selection probabilities in the trial dictionary

            # check relevance of subgoal_sub in goal_sub
            if trial_dict is not None:
                if subgoal_sub not in self.goal_progress[goal_sub].keys():
                    alt_goal_0 = trial_dict["alt_goal_selected"]
                    alt_goal_1 = [token for token in ['SH', 'BR', 'HO'] if token != goal_sub and token != alt_goal_0][0]
                    if self.goal_progress[alt_goal_0][subgoal_sub][0] < self.goal_progress[alt_goal_0][subgoal_sub][1]:
                        goal_rel = alt_goal_0
                    else:
                        goal_rel = alt_goal_1
                else:
                    goal_rel = goal_sub
                
                subgoal_m, subgoal_select_probs, subgoal_values = self.choose_subgoal(goal_rel)

                subgoal_modular_prob = subgoal_select_probs[subgoal_sub]
                if subgoal_sub in subgoal_values.keys():
                    subgoal_modular_value = subgoal_values[subgoal_sub]
                    subgoal_progress_rel = self.get_subgoal_progress(goal_rel)[subgoal_sub]
                else:
                    subgoal_modular_value = np.nan
                    subgoal_progress_rel = np.nan
                subgoal_modular_cons = [token for token in self.goal_progress[goal_rel].keys() if token != subgoal_sub]
                q_modular_cons = np.array([subgoal_select_probs[token] for token in subgoal_modular_cons])
                subgoal_modular_cons_prob = np.max(q_modular_cons)

                subgoal_modular_con_val_array = np.array([subgoal_values[token] for token in subgoal_modular_cons if token in subgoal_values.keys()])
                if len(subgoal_modular_con_val_array) == 0:
                    subgoal_modular_cons_value = np.nan
                else:
                    subgoal_modular_cons_value = np.max(subgoal_modular_con_val_array)
                
            
            self.update_subgoal_perseveration(subgoal_sub)
            if trial_dict:
                    trial_dict['G_prob'] = subgoal_select_probs_dict['G']
                    trial_dict['C_prob'] = subgoal_select_probs_dict['C']
                    trial_dict['S_prob'] = subgoal_select_probs_dict['S']
                    trial_dict['M_prob'] = subgoal_select_probs_dict['M']
                    subgoal_m, subgoal_select_probs, subgoal_values = self.choose_subgoal(goal_sub)

                    for subgoal_key in ['G', 'C', 'S', 'M']:
                        if subgoal_key in subgoal_values.keys():
                            trial_dict[subgoal_key + "_value"] = subgoal_values[subgoal_key]
                        else:
                            trial_dict[subgoal_key + "_value"] = np.nan

                    # if subgoal_sub in self.goal_progress[goal_sub].keys():
                    #     chosen_subgoal_progress = self.goal_progress[goal_sub][subgoal_sub][0] / self.goal_progress[goal_sub][subgoal_sub][1]
                    # elif subgoal_sub in self.goal_progress[trial_dict["alt_goal_selected"]].keys():
                    #     chosen_subgoal_progress = self.goal_progress[trial_dict["alt_goal_selected"]][subgoal_sub][0] / self.goal_progress[trial_dict["alt_goal_selected"]][subgoal_sub][1]
                    # else:
                    #     # get goal in ['SH', 'BR', 'HO'] that is not goal_sub or alt_goal_selected
                    #     goal_inf = [token for token in ['SH', 'BR', 'HO'] if token != goal_sub and token != trial_dict["alt_goal_selected"]][0]
                    #     chosen_subgoal_progress = self.goal_progress[goal_inf][subgoal_sub][0] / self.goal_progress[goal_inf][subgoal_sub][1]        
            
                    trial_dict["chosen_subgoal_progress"] = subgoal_progress_rel
                    trial_dict["chosen_subgoal_prob"] = subgoal_modular_prob
                    trial_dict["alt_subgoal_prob"] = subgoal_modular_cons_prob
                    trial_dict["chosen_subgoal_value"] = subgoal_modular_value
                    trial_dict["alt_subgoal_value"] = subgoal_modular_cons_value
                    cons = [token for token in ['G', 'C', 'S', 'M'] if token != subgoal_sub]
                    q_cons = np.array([subgoal_select_probs_dict[token] for token in ['G', 'C', 'S', 'M']  if token != subgoal_sub])
                    trial_dict['alt_subgoal_selected'] = cons[np.argmax(q_cons)]
                    trial_dict['chosen_subgoal_perseveration'] = self.Cp_sub[subgoal_sub]
                    
        else:
            # Assign subgoal based on the trial data
            subgoal_select_prob = -1

        flip = trial["action_outcome"]
        [goal_delta, subgoal_delta] = self.update_card_probs(goal_sub, subgoal_sub, flip)
        if trial_dict is not None:
            trial_dict["chosen_goal_delta"] = goal_delta
            trial_dict["chosen_subgoal_delta"] = subgoal_delta
        goal_completion = trial["goal_completion"]

        self.prev_goal = goal_sub
        self.prev_subgoal = subgoal_sub

        if trial_dict is not None:
            return trial, goal_completion, goal_select_prob, subgoal_select_prob, trial_dict
        return trial, goal_completion, goal_select_prob, subgoal_select_prob


        """Simulate subject action trial based on the provided trial data.
        This function simulates the subject's action trial by choosing a goal and subgoal,
        updating the card probabilities, resources, and goal progress, and checking for goal completion.
        """

    def run_sim_subject_action_trial(self, trial_data, trial_dict=None, subgoal=False):
        """
        Run a simulated subject action trial based on the provided trial data.
        :param trial_data: Dictionary containing the trial data.
        :return: Updated trial data, goal completion status, and selected goal.
        """
        if trial_dict is None:
           trial_dict = {}

        trial = trial_data.copy()

        trial["resources_pre"] = copy.deepcopy(self.resources)
        trial["goal_progress_pre"] = copy.deepcopy(self.goal_progress)

        self.block_type = trial['block_type']

        goal, goal_select_probs, goal_values = self.choose_goal(self.beta_g, self.prev_goal)

        trial["goal_selected"] = goal
        trial["probe_selected"] = goal
        if trial_dict is not None:
            trial_dict["goal_selected"] = goal

        self.update_goal_perseveration(goal)

        subgoal, subgoal_select_probs, subgoal_values = self.choose_subgoal(goal)
        #subgoal, switch_subgoal, subgoal_switch_probs, subgoal_values = self.choose_active_subgoal_at_random(goal)

        trial["resource_selected"] = subgoal
        trial["action_selected"] = subgoal
        if trial_dict is not None:
            trial_dict["resource_selected"] = subgoal
            trial_dict["action_selected"] = subgoal

        self.update_subgoal_perseveration(subgoal)

        if subgoal == -1:
            return trial, -1, -1

        prob = trial[subgoal + '_prob']
        if np.random.rand() < prob:
            flip = 1
        else:
            flip = 0

        trial[subgoal + "_flip"] = flip

        self.update_card_probs(goal, subgoal, flip)

        self.resources = self.update_resources(self.resources, trial, subgoal)
        self.goal_progress = self.update_goal_progress(self.goal_progress, self.resources)

        trial["resources_post"] = copy.deepcopy(self.resources)
        trial["goal_progress_post"] = copy.deepcopy(self.goal_progress)
        if trial_dict is not None:
            trial_dict["goal_progress_post"] = copy.deepcopy(self.goal_progress)
            trial_dict["resources_post"] = copy.deepcopy(self.resources)

        goal_completion, self.goal_progress, self.resources = \
            self.get_goal_completion_status(self.goal_progress, goal, self.resources)

        self.prev_goal = goal
        self.prev_subgoal = subgoal

        return trial, goal_completion, goal, trial_dict


    def run_experiment_block(self, trials, model=False):
        reward_trials = []
        goals = []
        trials_res = []

        trial_list = [int(i.split("_")[1]) for i in trials.keys() if "trial_" in i]
        trial_list = sorted(trial_list)

        for trial_num in trial_list:
            trial_data = trials["trial_" + str(trial_num)]

            if model == True:
                trial, goal_complete, _ = self.run_subject_action_trial(trial_data)
            else:
                trial, goal_complete, _, _ = self.run_sim_subject_action_trial(trial_data)

            if goal_complete == True:
                goals.append(goal_complete)
                reward_trials.append(trial_num)

            trials_res.append(trial)

        return trials_res, goals, reward_trials


        """Run model for the given experiment data.
        This function processes the experiment data, runs the model on each block of trials,
        and returns the results in a structured format.
        """

    def run_model(self, data, model=False):
        """
        Run model on the given data.
        :param data: Dictionary containing the experiment data.
        :param model: Boolean indicating whether to run the simulation or not.
        """

        # with open('generate/json/experiment.json', 'r') as file:
        #     episodes = json.load(file)

        block_trials_res = {"GoalSwitching": {}}
        blocks = block_trials_res["GoalSwitching"]
        self.reset_slots()
        self.reset_card_probs()
        self.current_time = 0.0

        block_nums = data['GoalSwitching'].keys()


        for block_num in block_nums:

            self.reset_card_probs()
            trials = data['GoalSwitching'][str(block_num)]
            blocks[str(block_num)] = {}

            trials_res, targets, reward_trials = self.run_experiment_block(trials, model)

            for trial_num in range(len(trials_res)):
                blocks[str(block_num)]["trial_" + str(trial_num)] = trials_res[trial_num]

            blocks[str(block_num)]["targets"] = targets
            blocks[str(block_num)]["reward_trials"] = reward_trials

        return block_trials_res

    
