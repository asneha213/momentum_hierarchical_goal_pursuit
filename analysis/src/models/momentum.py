from .model import Model
import numpy as np


class Momentum(Model):
    def __init__(self, params, learn_alt_goal=True):
        super().__init__(params)
        self.params = params
        self.alpha = params['alpha']
        self.alpha_c = params['alpha_c']
        self.beta_0 = params['beta_0']
        self.beta_0a = params['beta_0a']
        self.beta_g = params['beta_g']
        self.beta_a = params['beta_a']
        self.gamma_g = params['gamma']
        self.gamma_a = params['gamma']
        self.learn_alt_goal = learn_alt_goal
        self.model_name = 'momentum'

    def reset_card_probs(self):
        super().reset_card_probs()
        self.w_g = {'SH': 0.5, 'HO': 0.5, 'BR': 0.5}
        self.b_g = {'SH': 0.2, 'HO': 0.2, 'BR': 0.2}

        self.w_s = {'G': 0.5, 'S': 0.5, 'C': 0.5, 'M': 0.5}
        self.b_s = {'G': 0.2, 'S': 0.2, 'C': 0.2, 'M': 0.2}

    def get_goal_progress(self, goal):
        goal_progress = self.goal_progress[goal]
        progress = 0
        for key in goal_progress.keys():
            progress += goal_progress[key][0]
        progress = progress / 6
        return progress

    def goal_momentum(self, progress, rate, bias):
        """
        Calculate goal momentum using the formula:
        momentum = rate * progress + bias
        """
        return rate * progress + bias

    def update_goal_value(self, goal, flip):
        "Update values of goals based on the card flipped and its effect on momentum"
        progress = self.get_goal_progress(goal)

        if flip == 0:
            progress_new = progress
        else:
            progress_new = progress + 1.0/6


        delta = self.gamma_g * self.goal_momentum(progress_new, self.w_g[goal], self.b_g[goal]) \
                - self.goal_momentum(progress, self.w_g[goal], self.b_g[goal])

        self.w_g[goal] += self.alpha * delta * progress_new
        self.b_g[goal] += self.alpha * delta
        #print(goal, flip, self.w_g[goal], self.b_g[goal])

        return delta
        #print(goal, flip, self.w_g[goal], self.b_g[goal])


    def calculate_goal_value(self, goal):
        progress = self.get_goal_progress(goal)
        q = self.w_g[goal] * progress + self.b_g[goal]
        return q

    def calculate_qvals_goals(self):
        qvals = {}
        # calculate goal values
        for goal in ['SH', 'HO', 'BR']:
            qvals[goal] = self.calculate_goal_value(goal)
        #print(qvals)
        return qvals

    ### Subgoal methods

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

    def subgoal_momentum(self, progress, rate, bias):
        """
        Calculate subgoal momentum using the formula:
        momentum = rate * progress + bias
        """
        return rate * progress + bias
    
    def update_subgoal_value(self, goal, card, flip):
        """
        Update subgoal value based on the card flip outcome using momentum prediction errors. """

        def get_subgoal_progress_fraction(goal, subgoal):
            goal_progress = self.goal_progress[goal]
            subgoal_max = goal_progress[subgoal][1]
            progress = 1 / subgoal_max
            return progress

        # Check if subgoal is active in the current goal
        subgoal_progress = self.get_subgoal_progress(goal)
        if card not in subgoal_progress:
            return
        
        # Get progress and incremental progress for subgoal
        progress = subgoal_progress[card]
        progress_fraction = get_subgoal_progress_fraction(goal, card)
        if flip == 0:
            progress_new = progress
        else:
            progress_new = progress + progress_fraction

        # Update subgoal rate of progress and bias using momentum prediction error

        delta = self.gamma_a * self.subgoal_momentum(progress_new, self.w_s[card], self.b_s[card]) \
                - self.subgoal_momentum(progress, self.w_s[card], self.b_s[card])

        self.w_s[card] += self.alpha * delta * progress_new
        self.b_s[card] += self.alpha * delta

        return delta


    def calculate_qvals_subgoals(self, goal):
        """
        Calculate Q-values for each active subgoal in a goal."""
        qvals = {}

        # Get progress for each subgoal in a goal
        subgoal_progress = self.get_subgoal_progress(goal)
        for subgoal in subgoal_progress.keys():
            # only active subgoals have keys in subgoal_progress
            qvals[subgoal] = self.subgoal_momentum(
                subgoal_progress[subgoal],
                self.w_s[subgoal],
                self.b_s[subgoal]
            )
        return qvals


    def update_card_probs(self, goal, card, flip):
        super().update_card_probs(goal, card, flip)

        goal_delta = None
        subgoal_delta = None
        #for goal in ['SH', 'HO', 'BR']:
        if card in self.goal_progress[goal]:
            if self.goal_progress[goal][card][0] < self.goal_progress[goal][card][1]:
                goal_delta = self.update_goal_value(goal, flip)
                subgoal_delta = self.update_subgoal_value(goal, card, flip)

        ## update alternative-relavent goal value

        if self.learn_alt_goal:
            for alt_goal in ['SH', 'HO', 'BR']:
                if alt_goal != goal:
                    if card in self.goal_progress[alt_goal]:
                        if self.goal_progress[alt_goal][card][0] < self.goal_progress[alt_goal][card][1]:
                            self.update_goal_value(alt_goal, flip)


        return [goal_delta, subgoal_delta] 
        
        