"""Generate task designs; run from this directory to write into its json/ folder."""

import os
import numpy as np
import pandas as pd
from itertools import combinations


def get_isi(mean_isi=2, jitter=0.5):
    mean_interval = mean_isi  # Mean interval in seconds
    min_jitter = -1 * jitter  # Minimum jitter in seconds
    max_jitter = jitter
    jitter = np.random.uniform(min_jitter, max_jitter)
    isi = mean_interval + jitter
    return isi


def get_fixation_jitters(num_trials, min_isi=0.25, max_isi=3.5):
    isis = np.random.uniform(min_isi, max_isi, size=num_trials)
    return isis


A_G = {
    "SH" : {"G": 3, "M": 3, "S": 0, "C": 0},
    "HO" : {"M": 2, "C": 3, "S": 1, "G": 0},
    "BR" : {"G": 2, "S": 3, "C": 1, "M": 0},
}


CONDITIONS = {

    0: {"M": 0.8, "C": 0.2, "S": 0.2, "G": 0.8},
    1: {"M": 0.2, "C": 0.5, "S": 0.8, "G": 0.8},
    2:  {"M": 0.8, "C": 0.8, "S": 0.5, "G": 0.2},

    3: {"M": 0.6, "C": 0.4, "S": 0.4, "G": 0.6},
    4: {"M": 0.4, "C": 0.5, "S": 0.6, "G": 0.6},
    5: {"M": 0.6, "C": 0.6, "S": 0.5, "G": 0.4},

    'SH': {"M": 1.0, "C": 0.0, "S": 0.0, "G": 1.0},
    'HO': {"M": 1.0, "C": 1.0, "S": 1.0, "G": 0},
    'BR': {"M": 0, "C": 1.0, "S": 1.0, "G": 1.0},

}


def get_card_flip(prob):
    if np.random.rand() < prob:
        return 1
    else:
        return 0


def get_card_combinations(num_trials):
    cards = ["G", "C", "M", "S"]
    combns = list(combinations(cards, 3))
    combns = [list(c) for c in combns] * int(num_trials / 4)
    #combns = [["W", "M"], ["C", "S"], ["W", "S"], ["C", "M"]] * int(num_trials / 4)
    np.random.shuffle(combns)
    return combns


def get_card_set_blocks(num_blocks, practice=False):
    card_blocks = []
    for block in range(num_blocks):
        if practice == False:
            if block in [0, 1, 4, 5, 8, 9]:
                card_blocks.append(0)
            else:
                card_blocks.append(1)
        else:
            if block in [0,1,2, 6,7,8]:
                card_blocks.append(0)
            else:
                card_blocks.append(1)
    return card_blocks


def get_alien_card_maps():
    aliens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    #aliens = [i + '.png' for i in aliens]

    np.random.shuffle(aliens)

    aliens_0 = aliens[0:4]
    aliens_1 = aliens[4:8]

    cards = ["G", "C", "M", "S"]
    card_maps_0 = {}
    for i in range(4):
        card_maps_0[cards[i]] = aliens_0[i]

    card_maps_1 = {}
    for i in range(4):
        card_maps_1[cards[i]] = aliens_1[i]

    return card_maps_0, card_maps_1


def get_experiment_conditions(practice=False, seed=100):
    np.random.seed(seed)
    if practice:
        block_types = []
        for i in range(4):
            blocks_practice = ['SH', 'HO', 'BR']
            np.random.shuffle(blocks_practice)
            block_types += blocks_practice
    else:
        block_types = [0, 1, 2, 3, 4, 5] + [0, 4, 5]

    np.random.shuffle(block_types)
    return block_types


def get_action_practice_blocks(alien_maps, num_trials):

    blocks = []
    for block_num in range(1):

        trial_conditions = (['G'] * 5 + ['M'] * 5 + ['C'] * 5 + ['S'] * 5) * 2 + ['G', 'M', 'C', 'S'] * int((num_trials - 40) / 4)

        block = []
        for trial_num in range(num_trials):
            trial = {}
            trial["block_num"] = block_num

            trial["trial_num"] = trial_num

            trial_condition = trial_conditions[trial_num]

            trial["resource"] = trial_condition

            cards = ["G", "M", "C", "S"]

            np.random.shuffle(cards)

            trial["37"] = cards[0]
            trial["38"] = cards[1]
            trial["39"] = cards[2]
            trial["40"] = cards[3]

            trial[cards[0] + "_key"] = "37"
            trial[cards[1] + "_key"] = "38"
            trial[cards[2] + "_key"] = "39"
            trial[cards[3] + "_key"] = "40"

            trial["G"] = alien_maps["G"]
            trial["C"] = alien_maps["C"]
            trial["M"] = alien_maps["M"]
            trial["S"] = alien_maps["S"]

            trial["correct_key"] = trial[trial_condition + "_key"]

            block.append(trial)

        blocks.append(block)

    return blocks


def generate_action_practice_json(blocks, file_name='json/action_practice.json'):
    df = pd.DataFrame()
    count = 0
    for block in blocks:
        for trial in block:
            trial['block'] = count
            trial_num = trial['trial_num']
            df = pd.concat([df, pd.DataFrame(trial, index=[trial_num])])
        count += 1

    df.groupby(['block']).apply(lambda x: x[df.columns[0:]].to_dict('records')).reset_index().rename(
        columns={0: 'trials'}).to_json(file_name, orient='records', indent=4)

    return df


def get_experiment_block_data(run_num, block_num, condition, alien_maps, num_trials, seed=100):
    goal_onset_jitters = get_fixation_jitters(num_trials, min_isi=0.25, max_isi=3.5)
    action_onset_jitters = get_fixation_jitters(num_trials, min_isi=0.25, max_isi=3.5)
    outcome_onset_jitters = get_fixation_jitters(num_trials, min_isi=0.25, max_isi=3.5)
    block = []

    goals = ['SH', 'HO', 'BR', 'NA']



    for trial_num in range(num_trials):
        trial = {}
        trial["block_num"] = block_num
        trial["block_type"] = condition
        trial["trial_num"] = trial_num

        trial["run_num"] = run_num

        cards = ["G", "M", "C", "S"]

        np.random.shuffle(cards)

        trial["37"] = cards[0]
        trial["38"] = cards[1]
        trial["39"] = cards[2]
        trial["40"] = cards[3]

        trial["G"] = alien_maps["G"]
        trial["C"] = alien_maps["C"]
        trial["M"] = alien_maps["M"]
        trial["S"] = alien_maps["S"]

        trial["goal_onset_jitter"] = goal_onset_jitters[trial_num]
        trial["action_onset_jitter"] = action_onset_jitters[trial_num]
        trial["outcome_onset_jitter"] = outcome_onset_jitters[trial_num]

        trial[cards[0][0] + "_key"] = "37"
        trial[cards[1][0] + "_key"] = "38"
        trial[cards[2][0] + "_key"] = "39"
        trial[cards[3][0] + "_key"] = "40"

        trial[cards[0][0] + "_prob"] = CONDITIONS[condition][cards[0][0]]
        trial[cards[1][0] + "_prob"] = CONDITIONS[condition][cards[1][0]]
        trial[cards[2][0] + "_prob"] = CONDITIONS[condition][cards[2][0]]
        trial[cards[3][0] + "_prob"] = CONDITIONS[condition][cards[3][0]]

        trial[cards[0][0] + "_flip"] = get_card_flip(CONDITIONS[condition][cards[0][0]])
        trial[cards[1][0] + "_flip"] = get_card_flip(CONDITIONS[condition][cards[1][0]])
        trial[cards[2][0] + "_flip"] = get_card_flip(CONDITIONS[condition][cards[2][0]])
        trial[cards[3][0] + "_flip"] = get_card_flip(CONDITIONS[condition][cards[3][0]])

        np.random.shuffle(goals)

        trial["37_goal"] = goals[0]
        trial["38_goal"] = goals[1]
        trial["39_goal"] = goals[2]
        trial["40_goal"] = goals[3]

        trial[goals[0]] = "37"
        trial[goals[1]] = "38"
        trial[goals[2]] = "39"
        trial[goals[3]] = "40"

        block.append(trial)

    return block


def get_block_types_two_runs(seed=100):
    np.random.seed(seed)
    blocks_low_d = [3, 4, 5] * 2
    np.random.shuffle(blocks_low_d)
    run_1_blocks = [0, 1, 2] + blocks_low_d[0:3]
    run_2_blocks = [0, 1, 2] + blocks_low_d[3:6]
    np.random.shuffle(run_1_blocks)
    np.random.shuffle(run_2_blocks)
    return run_1_blocks, run_2_blocks


def get_two_runs_experiment_blocks(alien_maps, num_trials, seed=100):
    np.random.seed(seed)
    run_1_conditions, run_2_conditions = get_block_types_two_runs(seed=seed)
    alien_maps_1 = alien_maps[0]
    alien_maps_2 = alien_maps[1]

    run_num = 1
    run_1_blocks = []
    for block_num in range(len(run_1_conditions)):
        condition = run_1_conditions[block_num]
        block = get_experiment_block_data(run_num, block_num, condition, alien_maps_1, num_trials, seed=seed)
        run_1_blocks.append(block)

    run_num = 2
    run_2_blocks = []
    for block_num in range(len(run_2_conditions)):
        condition = run_2_conditions[block_num]
        block = get_experiment_block_data(run_num, block_num, condition, alien_maps_2, num_trials, seed=seed)
        run_2_blocks.append(block)

    return run_1_blocks, run_2_blocks



def get_goal_practice_blocks(alien_maps, run_num, num_blocks, num_trials, seed=100):
    np.random.seed(seed)
    block_types = []
    for i in range(int(num_blocks/3)):
        blocks_practice = ['SH', 'HO', 'BR']
        np.random.shuffle(blocks_practice)
        block_types += blocks_practice

    np.random.shuffle(block_types)
    blocks = []
    for block_num in range(num_blocks):
        condition = block_types[block_num]
        block = get_experiment_block_data(run_num, block_num, condition, alien_maps, num_trials, seed=seed)
        blocks.append(block)
    return blocks

def get_experiment_blocks(alien_maps, num_blocks, num_trials, practice=False, seed=100):
    np.random.seed(seed)
    block_types = get_experiment_conditions(seed=seed, practice=practice)
    card_blocks = get_card_set_blocks(num_blocks, practice=practice)
    blocks = []

    for block_num in range(num_blocks):
        condition = block_types[block_num]
        #card_set_type = card_blocks[block_num]
        block = get_experiment_block_data(block_num, condition, alien_maps, num_trials, seed=seed)
        blocks.append(block)
    return blocks


def generate_experiment_json(blocks, file_name='json/experiment.json'):
    df = pd.DataFrame()
    count = 0
    for block in blocks:
        for trial in block:
            trial['block'] = count
            trial_num = trial['trial_num']
            df = pd.concat([df, pd.DataFrame(trial, index=[trial_num])])
        count += 1

    df.groupby(['block']).apply(lambda x: x[df.columns[0:]].to_dict('records')).reset_index().rename(
        columns={0: 'trials'}).to_json(file_name, orient='records', indent=4)

    return df


def get_experiment_jsons(subject_id=None, data_dir="../../fmri/subject_data/"):
    directory = data_dir + "sub_" + str(subject_id) + "/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    seed = 10 * subject_id + 20

    alien_maps_0, alien_maps_1 = get_alien_card_maps()
    action_practice_0 = get_action_practice_blocks(alien_maps_0, 60)
    action_practice_1 = get_action_practice_blocks(alien_maps_1, 60)

    generate_action_practice_json(action_practice_0, file_name= directory + 'action_practice_session_1.json')
    generate_action_practice_json(action_practice_1, file_name= directory + 'action_practice_session_2.json')

    goal_practice_0 = get_goal_practice_blocks(alien_maps_0, run_num=1, num_blocks=12, num_trials=10, seed=seed)
    goal_practice_1 = get_goal_practice_blocks(alien_maps_1, run_num=2, num_blocks=12, num_trials=10, seed=seed)

    generate_experiment_json(goal_practice_0, file_name= directory + 'goal_practice_session_1.json')
    generate_experiment_json(goal_practice_1, file_name= directory + 'goal_practice_session_2.json')

    block = get_experiment_block_data(0, 0, 'BR', alien_maps_0, num_trials=12, seed=seed)
    generate_experiment_json([block], file_name= directory + 'main_practice_session_1.json')

    alien_maps = [alien_maps_0, alien_maps_1]
    run_1_blocks, run_2_blocks = get_two_runs_experiment_blocks(alien_maps, num_trials=40, seed= 10 * subject_id + 20)
    generate_experiment_json(run_1_blocks, file_name=directory + 'main_task_session_1.json')
    generate_experiment_json(run_2_blocks, file_name=directory + 'main_task_session_2.json')


if __name__ == "__main__":
    get_experiment_jsons(subject_id=0)
    #generate_experiment_json(num_blocks=12, num_rounds=10, practice=True)
    #generate_action_practice_json()