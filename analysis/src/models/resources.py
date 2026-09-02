from .model import Model
import numpy as np

from scipy.special import softmax

class Resources(Model):
    def __init__(self, params):
        super().__init__(params)
        self.alpha = params['alpha']
        self.alpha_c = params['alpha_c']
        self.alpha_ca = params['alpha_ca']
        self.beta_0 = params['beta_0']
        self.beta_0a = params['beta_0a']
        self.beta_g = params['beta_g']
        self.beta_a = params['beta_a']
        self.gamma_g = params['gamma']
        self.model_name = 'resources'

    def reset_card_probs(self):
        super().reset_card_probs()
        self.M = {
            'G': 0.25,
            'M': 0.25,
            'C': 0.25,
            'S': 0.25,
        }

    def update_subgoal_perseveration(self, subgoal):

        self.Cp_sub[subgoal] += self.alpha_ca * (1 - self.Cp_sub[subgoal])
        for poss_goal in ["G", "C", "S", "M"]:
            if poss_goal != subgoal:
                self.Cp_sub[poss_goal] += self.alpha_ca * (0 - self.Cp_sub[poss_goal])

    def update_card_probs(self, goal, card, flip):
        super().update_card_probs(goal, card, flip)

        if flip == 1:
            self.M[card] +=  self.alpha * (1 - self.M[card])
        else:
            self.M[card] +=  self.alpha * (0 - self.M[card])

        return [None, None]

    # def calculate_qvals_goals(self):
    #     qvals = {}

    #     for goal in ['SH', 'HO', 'BR']:
    #         subgoal_values = []
    #         q_vals_sub = self.calculate_qvals_subgoals(goal)
    #         active_subgoals = list(q_vals_sub.keys())
    #         for subgoal in active_subgoals:
    #             subgoal_values.append(q_vals_sub[subgoal])
    #         if len(subgoal_values) == 0:
    #             qvals[goal] = 0
    #         else:
    #             qvals[goal] = np.mean(subgoal_values)

    #     return qvals
    def calculate_goal_value(self, goal):

        count = 6 - 6 * self.get_goal_progress(goal)

        return 10.0 / (1 + self.gamma_g * count)

    def calculate_qvals_goals(self):
        qvals = {}

        #calculate goal values
        for goal in ['SH', 'HO', 'BR']:
            qvals[goal] = self.calculate_goal_value(goal)

        return qvals

    def calculate_qvals_subgoals(self, goal):
        qvals = {}
        goal_progress = self.goal_progress[goal]
        for subgoal in goal_progress.keys():
            if goal_progress[subgoal][0] < goal_progress[subgoal][1]:
                qvals[subgoal] = self.M[subgoal]

        return qvals


    def choose_goal(self, beta_g, prev_goal, return_switch_probs=False):
        q_vals = self.calculate_qvals_goals()
        q_probs = softmax(beta_g * np.array([q_vals['SH'], q_vals['HO'], q_vals['BR']]) + self.beta_0 * np.array([self.C['SH'], self.C['HO'], self.C['BR']]))
        goal = np.random.choice(['SH', 'HO', 'BR'], p=q_probs)
        return goal, {'SH': q_probs[0], 'HO': q_probs[1], 'BR': q_probs[2]}, q_vals


    def choose_subgoal_copy(self, goal):
        q_vals = self.calculate_qvals_subgoals(goal)
        active_subgoals = list(q_vals.keys())
        subgoal_select_probs = {'G': 0.01, 'C': 0.01, 'S': 0.01, 'M': 0.01}
        if len(active_subgoals) == 0:
            active_subgoals = []
            for subgoal in ['G', 'C', 'S', 'M']:
                if self.resources[subgoal] < self.resource_limit:
                    active_subgoals.append(subgoal)
            return np.random.choice(active_subgoals), subgoal_select_probs, q_vals
        q_probs = softmax(self.beta_a * np.array([q_vals[token] for token in active_subgoals]) + self.beta_0a * np.array([self.Cp_sub[token] for token in active_subgoals]))
        subgoal = np.random.choice(active_subgoals, p=q_probs)
        for i, token in enumerate(active_subgoals):
            subgoal_select_probs[token] = q_probs[i]
        return subgoal, subgoal_select_probs, q_vals



    def choose_subgoal(self, goal):
        subgoal_select_probs = {'G': 0.01, 'C': 0.01, 'S': 0.01, 'M': 0.01}
        active_subgoals = []
        for subgoal in ['G', 'C', 'S', 'M']:
            if self.resources[subgoal] < self.resource_limit:
                active_subgoals.append(subgoal)

        q_probs = softmax(self.beta_a * np.array([self.M[token] for token in active_subgoals]) + self.beta_0a * np.array([self.Cp_sub[token] for token in active_subgoals]))
        subgoal = np.random.choice(active_subgoals, p=q_probs)
        for i, token in enumerate(active_subgoals):
            subgoal_select_probs[token] = q_probs[i]
        return subgoal, subgoal_select_probs, self.M