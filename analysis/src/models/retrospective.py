from .model import Model
import numpy as np
import copy


class Retrospective(Model):
    def __init__(self, params):
        super().__init__(params)
        self.alpha_c = params['alpha_c']
        self.beta_0 = params['beta_0']
        self.beta_0a = params['beta_0a']
        self.beta_g = params['beta_g']
        self.beta_a = params['beta_a']
        self.gamma_g = params['gamma']
        self.gamma_a = params['gamma']
        self.model_name = 'retrospective'

    def reset_card_probs(self):
        super().reset_card_probs()

    def update_card_probs(self, goal, card, flip):
        super().update_card_probs(goal, card, flip)
        return [None, None]



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
        for subgoal in self.goal_progress[goal].keys():
            if self.goal_progress[goal][subgoal][0] == self.goal_progress[goal][subgoal][1]:
                continue
            count = self.goal_progress[goal][subgoal][1] - self.goal_progress[goal][subgoal][0]
            qvals[subgoal] = 10.0 / (1 + self.gamma_a * count)
        return qvals



