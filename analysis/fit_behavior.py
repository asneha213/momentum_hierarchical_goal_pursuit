import optuna
import pickle

import sys
sys.path.append("src/")

from src.models import *
from src.datautils import *
from src.run_model import get_model

import sys
import platform

print(platform.node())


def get_optuna_params(trial, model_name):

    alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
    alpha_c = trial.suggest_uniform('alpha_c', 0.0, 1.0)
    beta_0 = trial.suggest_uniform('beta_0', -1, 1)
    beta_0a = trial.suggest_uniform('beta_0a', -1, 1)
    beta_g = trial.suggest_uniform('beta_g', 0.0, 10.0)
    beta_a = trial.suggest_uniform('beta_a', 0.0, 10.0)

    params = { 'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g, 'beta_a': beta_a }

    if model_name == "prospective":
        gamma = trial.suggest_uniform('gamma', 0, 1.0)
        params = { 'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g,\
            'beta_a': beta_a, 'gamma': gamma}

    elif model_name == "momentum" or model_name == "momentum_learn_alt_goal" or \
        model_name == "momentum_with_pers_subgoal" or model_name == "prospective_exp":
        gamma = trial.suggest_uniform('gamma', 0.6, 1.0)
        params = { 'alpha': alpha, 'alpha_c': alpha_c,  'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g,\
            'beta_a': beta_a, 'gamma': gamma}

    elif model_name == "momentum_with_pros_subgoal":
        gamma = trial.suggest_uniform('gamma', 0.6, 1.0)
        k = trial.suggest_uniform('k', 0.0, 1.0)
        params = { 'alpha': alpha, 'alpha_c': alpha_c,  'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g,\
            'beta_a': beta_a, 'gamma': gamma, 'k': k}

    return params


class BehaviorFits:
    def __init__(self, experiment, subject_id, model_name, model_res=None, num_sims=50, data_type='fmri', normalize=False):
        self.experiment = experiment
        self.subject_id = subject_id
        self.model_name = model_name
        if not model_res:
            self.subject_data = get_subject_data_from_id(experiment, subject_id, data_type=data_type)
            if data_type == 'fmri':
                self.subject_data = fill_in_goal_completion_trials(self.subject_id, self.subject_data)
        else:
            self.subject_data = model_res
        self.data_type = data_type
        self.test_blocks = None
        self.trt =None
        self.num_sims = num_sims
        self.normalize = normalize

    def get_cv_blocks(self, num_blocks, num_folds):
        blocks = np.arange(num_blocks)
        np.random.shuffle(blocks)
        blocks = np.array_split(blocks, num_folds)
        return blocks

    def get_choice_likelihood(self, params, test=False, likelihood_split=False):

        blocks_data = self.subject_data['GoalSwitching']
        loglikelihoods_goal = []
        loglikelihoods_subgoal = []
        stay_loglikelihoods = []
        switch_loglikelihoods = []
        subgoal_stay_loglikelihoods = []
        subgoal_switch_loglikelihoods = []
        num_goal_stays = 0
        num_goal_switches = 0
        num_subgoal_stays = 0
        num_subgoal_switches = 0

        model = get_model(self.model_name, params)
        model.reset_slots()
        model.reset_card_probs()
        
        prev_goal = ''
        prev_subgoal = ''

        for block_num in range(len(blocks_data.keys())):

            if test == False and self.test_blocks is not None:
                if block_num in self.test_blocks:
                    continue
            if test:
                if block_num not in self.test_blocks:
                    continue

            if self.trt == 0:
                if block_num < len(blocks_data.keys()) / 2:
                    continue
            elif self.trt == 1:
                if block_num >= len(blocks_data.keys()) / 2:
                    continue

            block = blocks_data[str(block_num)]
            model.reset_card_probs()

            trial_list = [int(i.split("_")[1]) for i in block.keys() if "trial_" in i]
            trial_list = sorted(trial_list)

            for trial_num in trial_list:
                model.trial_num = trial_num
                trial = block['trial_' + str(trial_num)]

                if "goal_selected" not in trial or "resource_selected" not in trial:
                    continue
                trial_m, goal_complete, goal_select_prob, subgoal_select_prob= model.run_subject_action_trial(trial)
                #print(goal_select_prob, subgoal_select_prob, trial['goal_selected'], trial['resource_selected'], trial['resources_pre'])
                if goal_select_prob != -1:
                    loglikelihoods_goal.append(np.log(goal_select_prob))
                    #print(goal_select_prob, np.log(goal_select_prob))
                if subgoal_select_prob != -1:
                    loglikelihoods_subgoal.append(np.log(subgoal_select_prob))
                    
                if goal_select_prob != -1:
                    if trial['goal_selected'] == prev_goal:
                        stay_loglikelihoods.append(np.log(goal_select_prob))
                        num_goal_stays += 1
                    else:
                        switch_loglikelihoods.append(np.log(goal_select_prob))
                        num_goal_switches += 1
                        
                if subgoal_select_prob != -1:
                    if trial['resource_selected'] == prev_subgoal:
                        subgoal_stay_loglikelihoods.append(np.log(subgoal_select_prob))
                        num_subgoal_stays += 1
                    else:
                        subgoal_switch_loglikelihoods.append(np.log(subgoal_select_prob))
                        num_subgoal_switches += 1
                prev_goal = trial['goal_selected']
                prev_subgoal = trial['resource_selected']

        if self.data_type == 'online':
            loglikelihood = -1 * np.sum(loglikelihoods_goal) - 1 * np.sum(loglikelihoods_subgoal)
            print("Goal Loglikelihood: ", -1 * np.sum(loglikelihoods_goal), "Subgoal Loglikelihood: ", -1 * np.sum(loglikelihoods_subgoal), "Total Loglikelihood: ", loglikelihood)
            if likelihood_split:
                likelihood_dict = {
                    'goal_loglikelihood': -1 * np.sum(loglikelihoods_goal),
                    'subgoal_loglikelihood': -1 * np.sum(loglikelihoods_subgoal),
                    'total_loglikelihood': loglikelihood
                }
                return likelihood_dict
            return loglikelihood

    def get_optuna_objective(self, trial):
        params = get_optuna_params(trial, self.model_name)
        err = self.get_choice_likelihood(params)
        return err

    def fit_optuna(self, trt=None):
        best_params = []
        best_vals = []
        self.trt = trt

        for i in range(self.num_sims):
            study = optuna.create_study()
            study.optimize(self.get_optuna_objective, n_trials=100)
            best_params.append(study.best_params)
            best_vals.append(study.best_value)

        best_params = np.array(best_params)
        best_vals = np.array(best_vals)
        best_params = best_params[np.argmin(best_vals)]
        best_vals = np.min(best_vals)
        print(best_vals, best_params)
        return best_params, best_vals

    def fit_behavior(self, write_results=False):
        if write_results:
            outdir = MODEL_RESULTS + self.model_name + "_" + str(self.experiment) + "/"
            file_path = outdir + str(self.subject_id) + ".pkl"

            if os.path.exists(file_path):
                return 

        params, fits = self.fit_optuna()
        
        # Get likelihood breakdown for best fitting parameters
        likelihood_dict = self.get_choice_likelihood(params, likelihood_split=True)
        print(likelihood_dict)

        if write_results:
            model_fits = {}
            model_fits['fits'] = fits
            model_fits['params'] = params
            model_fits['likelihood_breakdown'] = likelihood_dict
            if self.data_type == 'fmri':
                if not self.normalize:
                    outdir = MODEL_RESULTS + self.model_name + "_" + str(self.experiment) + "/"
                else:
                    outdir = MODEL_RESULTS + self.model_name + "_normalized_" + str(self.experiment) + "/"
            elif self.data_type == 'online':
                outdir = 'results_online/' + self.model_name + "_" + str(self.experiment) + "/"
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            file = open(outdir + str(self.subject_id) + ".pkl", "wb")
            pickle.dump(model_fits, file)
            file.close()

        return params, fits

    def get_cv_score(self):
        if self.experiment == 1:
            num_folds = 3
            num_blocks = 18
        elif self.experiment == 2:
            num_folds = 3
            num_blocks = 12

        cv_blocks = self.get_cv_blocks(num_blocks, num_folds)
        self.test_blocks = cv_blocks[0]
        print(self.test_blocks)
        params, fits = self.fit_optuna()
        params = list(params.values())
        cv_score, _ = self.get_choice_likelihood(params, test=True)
        return cv_score

    def get_cv_results(self):
        cv_scores = []
        for i in range(5):
            cv_score = self.get_cv_score()
            cv_scores.append(cv_score)
        cv_scores = np.array(cv_scores)
        file = open(MODEL_RESULTS + self.model_name + "_" + str(self.experiment) + "/" + "cv_" + str(self.subject_id) + ".pkl", "wb")
        pickle.dump(cv_scores, file)
        file.close()
        print(np.mean(cv_scores), np.std(cv_scores))

    def get_test_retest_results(self, trt=0):
        params, fits = self.fit_optuna(trt=trt)
        
        # Get likelihood breakdown for best fitting parameters
        likelihood_dict = self.get_choice_likelihood(params, likelihood_split=True)
        
        model_fits = {}
        model_fits['fits'] = fits
        model_fits['params'] = params
        model_fits['likelihood_breakdown'] = likelihood_dict
        file = open('results_fmri/' + self.model_name + "_trt_" + str(self.experiment) + "/" + str(trt) + "/" + str(self.subject_id) + ".pkl", "wb")
        pickle.dump(model_fits, file)
        file.close()


if __name__ == "__main__":


    experiment = 0
    if platform.node() == "Snehas-MacBook-Air-2.local":
        model_name = "momentum"
        model_name = "momentum_learn_alt_goal"
        model_name = "prospective"
        #model_name = "td_persistence"
        #model_name = "prospective_exp"
        #model_name = "prospective"
        #model_name = "momentum_w_slowdown"
        #model_name = "momentum_with_pers_subgoal"
        #model_name = "momentum_with_pros_subgoal"
        subject_id = 39
        write_results = False
        num_sims = 1
        normalize = 0
        modelopt_t = BehaviorFits(experiment=experiment, subject_id=subject_id, \
                                model_name=model_name, num_sims=num_sims, data_type='fmri', normalize=normalize)

        #modelopt_t.get_choice_likelihood(params={'alpha': 0.69, 'beta_0': 0.84, 'beta_0a': -1.2, 'beta_g': 7.0, 'beta_a': 5.297, 'gamma': 0.765})
        #modelopt_t.get_choice_likelihood(params={'alpha': 0.01, 'alpha_a': 0.5, 'beta_0': 5.0, 'beta_g': 8.0, 'beta_a': 4.0, 'gamma': 0.9, 'gamma_s': 0.3})
        print(model_name)
        params, fits = modelopt_t.fit_behavior(write_results=write_results)
    else:
        subject_id = int(sys.argv[1])
        model_name = sys.argv[2]
        data_type = sys.argv[3]
        write_results = True
        normalize = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        num_sims = 100

        modelopt_t = BehaviorFits(experiment=experiment, subject_id=subject_id, \
                                model_name=model_name, num_sims=num_sims, data_type=data_type, normalize=normalize)

        params, fits = modelopt_t.fit_behavior(write_results=write_results)

