from .model import Model
import numpy as np
import copy


class ProspectiveExp(Model):
    def __init__(self, params):
        super().__init__(params)
        self.alpha = params['alpha']
        self.alpha_c = params['alpha_c']
        self.beta_0 = params['beta_0']
        self.beta_0a = params['beta_0a']
        self.beta_g = params['beta_g']
        self.beta_a = params['beta_a']
        self.gamma_g = params['gamma']
        self.gamma_a = params['gamma']
        self.model_name = 'prospective'

    def reset_card_probs(self):
        super().reset_card_probs()
        self.M = {
            'G': 0.5,
            'M': 0.5,
            'C': 0.5,
            'S': 0.5,
        }

    def update_card_probs(self, goal, card, flip):
        super().update_card_probs(goal, card, flip)

        if flip == 1:
            self.M[card] +=  self.alpha * (1 - self.M[card])
        else:
            self.M[card] +=  self.alpha * (0 - self.M[card])

        return [None, None]

    def finish_subgoal(self, goal, subgoal, goal_progress, count):
        if count > 15:
            return 25, goal_progress
        if goal_progress[goal][subgoal][0] == goal_progress[goal][subgoal][1]:
            return count, goal_progress
        else:
            if np.random.random() < self.M[subgoal]:
                goal_progress[goal][subgoal][0] += 1
            return self.finish_subgoal(goal, subgoal, goal_progress, count + 1)


    def calculate_goal_value(self, goal):
        subgoals = self.goal_progress[goal].keys()
        subgoal_vals = {subgoal: self.M[subgoal] for subgoal in subgoals}
        sorted_subgoals = sorted(subgoal_vals, key=subgoal_vals.get, reverse=True)

        goal_progress = copy.deepcopy(self.goal_progress)
        count = 0
        for subgoal in sorted_subgoals:
            count, goal_progress = self.finish_subgoal(goal, subgoal, goal_progress, count)

        return 10 * (self.gamma_g**(count - 1))


    def calculate_qvals_goals(self):
        qvals = {}

        #calculate goal values
        for goal in ['SH', 'HO', 'BR']:
            qvals[goal] = self.calculate_goal_value(goal)

        #print(qvals)
        return qvals


    def calculate_qvals_subgoals(self, goal):
        qvals = {}
        #goal_progress = self.goal_progress[goal]
        for subgoal in self.goal_progress[goal].keys():
            goal_progress = copy.deepcopy(self.goal_progress)
            if goal_progress[goal][subgoal][0] == goal_progress[goal][subgoal][1]:
                continue
            count = 0
            count, goal_progress = self.finish_subgoal(goal, subgoal, goal_progress, count)
            qvals[subgoal] = 10 * (self.gamma_a ** (count - 1))
        #print(self.goal_progress[goal], qvals)
        return qvals
        # for subgoal in goal_progress.keys():
        #     if goal_progress[subgoal][0] < goal_progress[subgoal][1]:
        #         qvals[subgoal] = self.M[subgoal]
        # return qvals





