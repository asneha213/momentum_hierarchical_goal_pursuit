import sys

from fit_behavior import *
from src.models import *

import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


def get_sample_random(model_name):

    if (model_name == "prospective") or (model_name == "momentum") or (model_name == "momentum_learn_alt_goal"):
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_0a = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        gamma = np.random.uniform(0.6, 1)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_0a': beta_0a, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma}

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
    modelfits = np.zeros(4)
    np.random.seed(seed)
    params = get_sample_random(model_name)
    model = get_model(model_name, params)
    data = get_subject_data_from_id(experiment=0, subject_id=11, data_type='fmri')
    
    model_res = model.run_model(data)

    models = ['td_persistence', 'momentum', 'prospective', 'hybrid']

    for i in range(len(models)):
        model_fname = models[i]
        fits = BehaviorFits(experiment, seed, model_fname, model_res, num_sims=20)
        r_params, vals = fits.fit_optuna()
        modelfits[i] = vals

    file = open('results/model_recovery/' + model_name + "_" + str(seed) + "_" + str(experiment) +  "_model_recovery.pkl",
                "wb")

    pickle.dump(modelfits, file)


def plot_model_recovery(experiment):
    models = ['td_persistence', 'momentum', 'prospective', 'hybrid']

    confusion = np.zeros((len(models), len(models)))

    for i in range(len(models)):
        model_fname = models[i]
        for seed in range(100):

            with open('results_latent/model_recovery/' + model_fname + "_" + str(seed) + "_" + str(
                    experiment) + "_model_recovery.pkl",
                      "rb") as f:

                modelfit = pickle.load(f)
                samples = 540

                modelfit[0] = 2 * modelfit[0] + 3 * np.log(samples)
                modelfit[1] = 2 * modelfit[1] + 4 * np.log(samples)
                modelfit[2] = 2 * modelfit[2] + 4 * np.log(samples)
                modelfit[3] = 2 * modelfit[3] + 5 * np.log(samples)

                confusion[i][np.argmin(modelfit)] += 1

    cm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    cm_inv = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]

    models = ['TD-Persistence', 'TD-Momentum', 'Prospective', 'Hybrid']

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.set(font_scale=0.8)

    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=models, yticklabels=models, ax=axs[0])
    axs[0].set_xlabel('Fit')
    axs[0].set_ylabel('Simulated')
    axs[0].set_title('Pr(Fit | Simulated)', fontsize=10)

    sns.heatmap(cm_inv, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=models, yticklabels=False, ax=axs[1])
    axs[1].set_xlabel('Fit')
    axs[1].set_title('Pr(Simulated | Fit)', fontsize=10)

    plt.savefig('figures/model_recovery.png')

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

    if platform.node() == "Snehas-MacBook-Air-2.local":
        seed = 4128
        model_name = "momentum_learn_alt_goal"
        # model_name = "hybrid"
        # model_name = "prospective"
        # model_name = "td_persistence"
        param = 1
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

