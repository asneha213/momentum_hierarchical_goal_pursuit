from .model import Model
import numpy as np

from scipy.special import softmax

class TDPersistence(Model):
    def __init__(self, params):
        super().__init__(params)
        self.alpha = params['alpha']
        self.alpha_c = params['alpha_c']
        self.beta_0 = params['beta_0']
        self.beta_0a = params['beta_0a']
        self.beta_g = params['beta_g']
        self.beta_a = params['beta_a']
        self.model_name = 'td_persistence'

    def reset_card_probs(self):
        super().reset_card_probs()
        self.M = {
            'G': 0.25,
            'M': 0.25,
            'C': 0.25,
            'S': 0.25,
        }

    def update_card_probs(self, goal, card, flip):
        super().update_card_probs(goal, card, flip)

        if flip == 1:
            self.M[card] +=  self.alpha * (1 - self.M[card])
        else:
            self.M[card] +=  self.alpha * (0 - self.M[card])

        return [None, None]

    def calculate_qvals_goals(self):
        qvals = {}

        # calculate goal values
        # qvals['SH'] = np.mean([3* self.M['G'], 3*self.M['M']])
        # qvals['HO'] = np.mean([3*self.M['C'], self.M['S'],  2*self.M['M']])
        # qvals['BR'] = np.mean([3*self.M['S'] ,2*self.M['G'] , self.M['C']])

        qvals['SH'] = min([self.M['G'], self.M['M']])
        qvals['HO'] = min([self.M['C'], self.M['M']])
        qvals['BR'] = min([self.M['S'] ,self.M['G']])

        return qvals


    def calculate_qvals_subgoals(self, goal):
        qvals = {}
        goal_progress = self.goal_progress[goal]
        for subgoal in goal_progress.keys():
            if goal_progress[subgoal][0] < goal_progress[subgoal][1]:
                qvals[subgoal] = self.M[subgoal]

        return qvals

