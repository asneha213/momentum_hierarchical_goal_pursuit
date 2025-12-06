from .momentum import Momentum
import copy
import numpy as np


class MomentumWithPersSubgoal(Momentum):
    def __init__(self, params, learn_alt_goal=True):
        super().__init__(params, learn_alt_goal)
        self.model_name = 'momentum_with_pers_subgoal'

    def reset_card_probs(self):
        super().reset_card_probs()
        self.w_g = {'SH': 0.5, 'HO': 0.5, 'BR': 0.5}
        self.b_g = {'SH': 0.2, 'HO': 0.2, 'BR': 0.2}
        self.w_s = {'G': 0.5, 'S': 0.5, 'C': 0.5, 'M': 0.5}
        self.b_s = {'G': 0.2, 'S': 0.2, 'C': 0.2, 'M': 0.2}
        self.M = {
            'G': 0.25,
            'M': 0.25,
            'C': 0.25,
            'S': 0.25,
        }

    def update_card_probs(self, goal, card, flip):
        super().update_card_probs(goal, card, flip)

        goal_delta = None
        subgoal_delta = None
        #for goal in ['SH', 'HO', 'BR']:
        if card in self.goal_progress[goal]:
            if self.goal_progress[goal][card][0] < self.goal_progress[goal][card][1]:
                goal_delta = self.update_goal_value(goal, flip)

        ## update alternative-relavent goal value

        if self.learn_alt_goal:
            for alt_goal in ['SH', 'HO', 'BR']:
                if alt_goal != goal:
                    if card in self.goal_progress[alt_goal]:
                        if self.goal_progress[alt_goal][card][0] < self.goal_progress[alt_goal][card][1]:
                            self.update_goal_value(alt_goal, flip)

        if flip == 1:
            self.M[card] +=  self.alpha * (1 - self.M[card])
        else:
            self.M[card] +=  self.alpha * (0 - self.M[card])

        return [goal_delta, subgoal_delta] 

    # Subgoal methods

    def calculate_qvals_subgoals(self, goal):
        qvals = {}
        goal_progress = self.goal_progress[goal]
        for subgoal in goal_progress.keys():
            if goal_progress[subgoal][0] < goal_progress[subgoal][1]:
                qvals[subgoal] = self.M[subgoal]
        return qvals
