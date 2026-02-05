from .momentum import Momentum
import numpy as np
from scipy.special import softmax


class MomentumWithSoftmax(Momentum):
    def __init__(self, params, learn_alt_goal=True):
        super().__init__(params, learn_alt_goal)
        self.model_name = 'momentum_with_softmax'

    def choose_goal(self, beta_g, prev_goal, return_switch_probs=False):
        q_vals = self.calculate_qvals_goals()
        q_probs = softmax(beta_g * np.array([q_vals['SH'], q_vals['HO'], q_vals['BR']]) + self.beta_0 * np.array([self.C['SH'], self.C['HO'], self.C['BR']]))
        goal = np.random.choice(['SH', 'HO', 'BR'], p=q_probs)
        return goal, {'SH': q_probs[0], 'HO': q_probs[1], 'BR': q_probs[2]}, q_vals

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
        q_probs = softmax(self.beta_a * np.array([q_vals[token] for token in active_subgoals]) + self.beta_0a * np.array([self.Cp_sub[token] for token in active_subgoals]))
        subgoal = np.random.choice(active_subgoals, p=q_probs)
        for i, token in enumerate(active_subgoals):
            subgoal_select_probs[token] = q_probs[i]
        return subgoal, subgoal_select_probs, q_vals