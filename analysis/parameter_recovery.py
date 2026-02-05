import sys
import os

from fit_behavior import *
from src.models import *

import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


def get_sample_random(model_name):
    params = None
    
    # Retrospective model (no alpha)
    if model_name == "retrospective":
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_0a = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        gamma = np.random.uniform(0, 1.0)
        params = {'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma}
    
    elif model_name == "progress":
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_0a = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        params = {'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g, 'beta_a': beta_a}
    # Resources models (no gamma, has alpha_ca)
    elif model_name == "resources" or model_name == "resources_depth":
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        alpha_ca = np.random.uniform(0, 1)
        if model_name == "resources_depth":
            beta_0 = np.random.uniform(-1, 1)
        else:
            beta_0 = np.random.uniform(0, 10)
        beta_0a = np.random.uniform(0, 10)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        gamma = np.random.uniform(0, 1.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'alpha_ca': alpha_ca, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma}
    
    # Prospective model (gamma 0-1)
    elif model_name == "prospective":
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_0a = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        gamma = np.random.uniform(0, 1.0)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma}
    
    # Momentum models (gamma 0.6-1)
    elif (model_name == "momentum" or model_name == "momentum_learn_alt_goal" or 
          model_name == "momentum_with_pers_subgoal" or
          model_name == "td_persistence"):
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_0a = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        gamma = np.random.uniform(0.6, 1)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma}
    
    elif model_name == "momentum_with_pros_subgoal":
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_0a = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        gamma = np.random.uniform(0.6, 1)
        k = np.random.uniform(0, 1)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma, 'k': k}
    return params


def get_sample_random_from_model_fits(model_name, experiment=0, results_path=None):
    """
    Sample random parameters from the range of fitted parameters across participants.

    Loads all participant fits for the given model from the model results folder,
    computes the min/max range for each parameter, and returns one random sample
    drawn uniformly from each parameter's range.

    Args:
        model_name: Name of the model (e.g. 'momentum_learn_alt_goal', 'retrospective').
        experiment: Experiment index used in the results folder name (default 0).
        results_path: Base path for results (default MODEL_RESULTS). Use MODEL_RESULTS_ONLINE
            for online experiment fits.

    Returns:
        dict: Parameter name -> value, same format as get_sample_random(model_name).
        None if no participant fits are found.
    """
    if results_path is None:
        results_path = MODEL_RESULTS
    folder = os.path.join(results_path, model_name + "_" + str(experiment))
    if not os.path.isdir(folder):
        print(f"No results found for {model_name} in {results_path}")
        return None
    all_params = []
    for fname in os.listdir(folder):
        if not fname.endswith(".pkl") or fname.startswith("cv_"):
            continue
        pkl_path = os.path.join(folder, fname)
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "params" in data:
                all_params.append(data["params"])
        except Exception:
            continue
    if not all_params:
        print(f"No parameters found for {model_name} in {results_path}")
        return None
    param_names = set()
    for p in all_params:
        param_names.update(p.keys())
    ranges = {}
    for pname in param_names:
        vals = [d[pname] for d in all_params if pname in d]
        if vals:
            ranges[pname] = (float(np.min(vals)), float(np.max(vals)))
    params = {
        pname: np.random.uniform(ranges[pname][0], ranges[pname][1])
        for pname in param_names
    }
    return params


def simulate_and_recover(experiment, model_name, params, seed):
    model = get_model(model_name, params)
    experiment = int(experiment)
    subject_id = 11  # Use a fixed subject ID for parameter recovery
    data = get_subject_data_from_id(experiment, subject_id, data_type='fmri')
    model_res = model.run_model(data)
    fits = BehaviorFits(experiment, seed, model_name, model_res, num_sims=25)
    r_params, vals = fits.fit_optuna()
    r_params = [r_params['alpha'], r_params['alpha_c'], r_params['beta_0'], r_params['beta_0a'], r_params['beta_g'], r_params['beta_a'], r_params['gamma']]

    precovery = {}
    o_params = [params['alpha'], params['alpha_c'], params['beta_0'], params['beta_0a'], params['beta_g'], params['beta_a'], params['gamma']]
    precovery['original'] = o_params
    precovery['recovered'] = r_params

    os.makedirs(MODEL_RESULTS + 'param_recovery', exist_ok=True)

    file = open(MODEL_RESULTS + 'param_recovery/' + model_name + "_" + str(experiment) + "_" + str(seed) + "_parameter_recovery.pkl", "wb")

    pickle.dump(precovery, file)
    return r_params


def recover_params(experiment, model_name, seed):
    np.random.seed(seed)
    params = get_sample_random(model_name)
    simulate_and_recover(experiment, model_name, params, seed)


def recover_model(experiment, model_name, seed):
    # All models from model_comparisons.py
    models = [
        'momentum_learn_alt_goal',
        'momentum', 
        'momentum_with_pros_subgoal',
        'momentum_with_pers_subgoal',
        'prospective',
        'td_persistence',
        'retrospective',
        'resources',
        'progress',
    ]
    
    modelfits = np.zeros(len(models))
    np.random.seed(seed)
    params = get_sample_random_from_model_fits(model_name, experiment=experiment, results_path=MODEL_RESULTS_ONLINE)
    #if params is None:
    #    params = get_sample_random_from_model_fits(model_name, experiment=experiment, results_path=MODEL_RESULTS)

    print(params)
    model = get_model(model_name, params)
    data = get_subject_data_from_id(experiment=0, subject_id=11, data_type='online')
    
    model_res = model.run_model(data)

    for i in range(len(models)):
        model_fname = models[i]
        print("Fitting model: ", model_fname)
        fits = BehaviorFits(experiment, subject_id=seed, model_name=model_fname, model_res=model_res, num_sims=20, data_type='online')
        r_params, vals = fits.fit_optuna()
        modelfits[i] = vals

    os.makedirs(MODEL_RESULTS_ONLINE + 'model_recovery', exist_ok=True)

    file = open(MODEL_RESULTS_ONLINE + 'model_recovery/' + model_name + "_" + str(seed) + "_" + str(experiment) +  "_model_recovery.pkl",
                "wb")

    pickle.dump(modelfits, file)


def plot_model_recovery(experiment):
    # All models from recover_model
    models = [
        'momentum_learn_alt_goal',
        'momentum', 
        'momentum_with_pros_subgoal',
        'momentum_with_pers_subgoal',
        'prospective',
        'td_persistence',
        'retrospective',
        'resources',
        'progress',
    ]
    
    # Model display names
    model_display_names = {
        'momentum_learn_alt_goal': 'TD-momentum-alt-goal',
        'momentum': 'TD-momentum',
        'momentum_with_pros_subgoal': 'TD-momentum(G)-Prospective(SG)',
        'momentum_with_pers_subgoal': 'TD-momentum(G)-Persistence(SG)', 
        'prospective': 'Prospective',
        'td_persistence': 'TD-persistence',
        'retrospective': 'Retrospective',
        'resources': 'Resources',
        'progress': 'Progress',
    }
    
    # Parameter counts for each model (for BIC calculation)
    # These are the number of free parameters in each model
    param_counts = {
        'momentum_learn_alt_goal': 7,  # alpha, alpha_c, beta_0, beta_0a, beta_g, beta_a, gamma
        'momentum': 7,
        'momentum_with_pros_subgoal': 7,
        'momentum_with_pers_subgoal': 7,
        'prospective': 7,
        'td_persistence': 7,
        'retrospective': 6,  # alpha_c, beta_0, beta_0a, beta_g, beta_a, gamma (no alpha)
        'resources': 7,  # alpha, alpha_c, alpha_ca, beta_0, beta_0a, beta_g, beta_a (no gamma)
        'progress': 6,  # alpha_c, beta_0, beta_0a, beta_g, beta_a (no gamma)
    }

    confusion = np.zeros((len(models), len(models)))
    samples = 360  # Approximate number of samples

    for i in range(len(models)):
        model_fname = models[i]
        for seed in range(100):
            try:
                file_path = MODEL_RESULTS_ONLINE + 'model_recovery/' + model_fname + "_" + str(seed) + "_" + str(experiment) + "_model_recovery.pkl"
                with open(file_path, "rb") as f:
                    modelfit = pickle.load(f)
                
                # Convert fit values to BIC for each model
                # modelfit contains the fit values from fit_optuna
                # get_choice_likelihood returns: -1 * sum(loglikelihoods) where loglikelihoods are negative
                # So the returned value is positive (negative of sum of negatives)
                # In model_comparisons, BIC = 2 * ll + k * ln(n) where ll is this positive value
                # So modelfit[j] is already in the right form: BIC = 2 * modelfit[j] + k * ln(n)
                bic_values = np.zeros(len(models))
                for j in range(len(models)):
                    if j < len(modelfit) and not np.isinf(modelfit[j]):
                        k = param_counts[models[j]]
                        # modelfit[j] is the fit value from get_choice_likelihood (already positive)
                        # BIC = 2 * modelfit[j] + k * ln(n)
                        bic_values[j] = 2 * modelfit[j] + k * np.log(samples)
                        if model_fname == "prospective":
                            print(bic_values[j], models[j])
                    else:
                        bic_values[j] = np.inf
                
                # Find which model has the minimum BIC
                best_model_idx = np.argmin(bic_values)
                confusion[i][best_model_idx] += 1
            except FileNotFoundError:
                # File doesn't exist, skip this seed
                continue
            except Exception as e:
                print(f"Warning: Error processing {model_fname} seed {seed}: {e}")
                continue

    # Normalize confusion matrix
    row_sums = confusion.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm = confusion.astype('float') / row_sums[:, np.newaxis]
    
    col_sums = cm.sum(axis=0)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    cm_inv = cm.astype('float') / col_sums[np.newaxis, :]

    # Create display names list
    display_names = [model_display_names[m] for m in models]

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    sns.set(font_scale=0.8)

    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=display_names, yticklabels=display_names, ax=axs[0])
    axs[0].set_xlabel('Fit', fontsize=13, fontweight='bold')
    axs[0].set_ylabel('Simulated', fontsize=13, fontweight='bold')
    axs[0].set_title('Pr(Fit | Simulated)', fontsize=13, fontweight='bold')
    axs[0].tick_params(axis='both', labelsize=11)

    sns.heatmap(cm_inv, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=display_names, yticklabels=False, ax=axs[1])
    axs[1].set_xlabel('Fit', fontsize=13, fontweight='bold')
    axs[1].set_title('Pr(Simulated | Fit)', fontsize=13, fontweight='bold')
    axs[1].tick_params(axis='x', labelsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_TOPICS + 'model_recovery.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_parameter_recovery():

    original = []
    recovered = []

    num_sims = 500
    for i in range(num_sims):
        try:
            with open(MODEL_RESULTS + 'param_recovery/momentum_learn_alt_goal_0_' + str(i) + "_parameter_recovery.pkl", "rb") as f:
                precovery = pickle.load(f)
        except:
            continue

        porig = precovery['original']
        precovery_recovered = precovery['recovered']
        precovery_original = porig


        original.append(precovery_original)
        recovered.append(precovery_recovered)

    original = np.array(original)
    recovered = np.array(recovered)

    print(original.shape)
    param_names = ["Learning rate", "Goal switching cost ", \
                   "Action switching cost", "Goal softmax", "Action softmax", "Discount factor"]

    fig, axs = plt.subplots(1, len(param_names), figsize=(18, 4))

    

    for k in range(len(param_names)):
        print(pg.corr(original[:, k], recovered[:, k], method="spearman"))
        data = original[:, k]
        axs[k].scatter(original[:, k] , recovered[:, k], s=5)
        axs[k].set_xlabel(param_names[k] + " - Original", fontweight='bold')
        axs[k].set_ylabel(param_names[k] + " - Recovered", fontweight='bold')
        axs[k].set_ylim(np.min(data)-0.1, np.max(data)+0.1)
        x = np.linspace(np.min(data), np.max(data), 100)
        y = x

        r = pg.corr(original[:, k], recovered[:, k], method="spearman")['r'].to_numpy()[0]

        # Create a plot
        axs[k].plot(x, y, color='red', label=f'R={r:.2f}')
        axs[k].legend(fontsize=10, loc='upper left', prop={'weight': 'bold'})

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    import platform 
    machine = 'local'
    #machine = 'server'

    if platform.node() == "Snehas-MacBook-Air-5.local":
        seed = 4228
        model_name = "momentum_learn_alt_goal"
        # model_name = "hybrid"
        # model_name = "prospective"
        # model_name = "td_persistence"
        #model_name = "resources"
        param = 0
        plot = True
        experiment = 0
    else:
        seed = int(sys.argv[1])
        experiment = 0
        model_name = sys.argv[2]
        param = int(sys.argv[3])
        plot = False

    if not plot:
        if param == 1:
           recover_params(experiment=experiment, model_name=model_name, seed=seed)
        else:
           recover_model(experiment=experiment, model_name=model_name, seed=seed)
    else:
        if param == 1:
            plot_parameter_recovery()
        else:
            plot_model_recovery(experiment=experiment)

