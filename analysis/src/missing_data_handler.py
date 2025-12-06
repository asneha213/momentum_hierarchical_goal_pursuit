from datautils import *
import copy


def check_for_goal_completion(pre_trial_resources, post_trial_resources):
    pre = pre_trial_resources
    post = post_trial_resources
    if pre['G'] == 3 and pre['M'] == 3 and post['G'] == 0 and post['M'] == 0:
        return "SH"
    elif post["G"] - pre["G"] == -2 and post["C"] - pre["C"] == -1 and post["S"] - pre["S"] == -3:
        return "BR"
    elif post["M"] - pre["M"] == -2 and post["S"] - pre["S"] == -1 and post["C"] - pre["C"] == -3:
        return "HO"


def fill_in_goal_completion_trials(subject_id, subject_data, run_num=None):
    blocks_data = subject_data['GoalSwitching']
    pre_trial_resources = None
    pre_goal_progress = None
    pre_trial_end_time = None

    subject_data_revised = copy.deepcopy(subject_data)
    blocks_data_revised = subject_data_revised['GoalSwitching']

    block_keys = list(blocks_data.keys())
    #sort block keys to ensure order
    block_keys.sort(key=lambda x: int(x))
    
    for block_key in block_keys:
    

        block = blocks_data[block_key]

        trial_list = [int(i.split("_")[1]) for i in block.keys() if "trial_" in i]
        trial_list = sorted(trial_list)
        max_trial_num = max(trial_list)


        for trial_num in range(max_trial_num + 1):
            if f'trial_{trial_num}' not in block:

                ## Goal completion trials are not recorded. So correcting that.
                
                block[f'trial_{trial_num}'] = {}
                post_trial_resources = block['trial_' + str(trial_num + 1)]['resources_pre'] if trial_num < max_trial_num else {'G': 0, 'M': 0, 'C': 0, 'S': 0}
                post_goal_progress = block['trial_' + str(trial_num + 1)]['goal_progress_pre'] if trial_num < max_trial_num else {'SH': {'G': [0, 0], 'M': [0, 0]}, 'BR': {'G': [0, 0], 'C': [0, 0], 'S': [0, 0]}, 'HO': {'C': [0, 0], 'M': [0, 0], 'S': [0, 0]}}
                goal_completed = check_for_goal_completion(pre_trial_resources, post_trial_resources)
                block['trial_' + str(trial_num)]['goal_completion'] = True
                #verify
                if run_num is not None:
                    if run_num < 4:
                        session_num = 1
                    else:
                        session_num = 2
                    block_num = int(block_key)
                else:
                    if int(block_key) < 6:
                        session_num = 1
                    else:
                        session_num = 2
                    block_num = int(block_key) % 6
                session_json_data = load_fmri_data_subject_input_json(subject_id, session_num)
                
                ### verify action onset jitter
                if trial_num > 0:
                    verify_trial_num = trial_num - 1
                else:
                    verify_trial_num = trial_num + 1

                if not session_json_data[block_num]['trials'][verify_trial_num]['action_onset_jitter'] == block['trial_' + str(verify_trial_num)]['action_onset_jitter']:
                    print("Onset jitter mismatch for trial", trial_num, "in block", block_key)

                trial_correct = {}
                trial_correct['goal_completion'] = True
                
                trial_correct['resources_pre'] = pre_trial_resources
                trial_correct['resources_post'] = post_trial_resources
                trial_correct['goal_progress_pre'] = pre_goal_progress
                trial_correct['goal_progress_post'] = post_goal_progress
                trial_correct['goal_selected'] = goal_completed
                trial_correct['goal_onset_jitter'] = session_json_data[block_num]['trials'][trial_num]['goal_onset_jitter']
                trial_correct['trial_start_tme'] = pre_trial_end_time
                trial_correct['resource_status_onset_time'] = pre_trial_end_time + 0.1
                trial_correct['goal_progress_onset_time'] = pre_trial_end_time + 0.1 + 0.9
                trial_correct['goal_onset_time'] = pre_trial_end_time + 0.15 + 0.75 + 1 + trial_correct['goal_onset_jitter'] 
                trial_correct['goal_probe_end_time'] = None
                trial_correct['block_type'] = session_json_data[block_num]['trials'][trial_num]['block_type']
                trial_correct['action_selected'] = None
                trial_correct['resource_selected'] = None
                trial_correct['action_onset_time'] = None
                trial_correct['action_end_time'] = None
                trial_correct['outcome_onset_time'] = None
                trial_correct['resource_stack_onset_time'] = None
                trial_correct['trial_end_time'] = None
                trial_correct['G_prob'] = session_json_data[block_num]['trials'][trial_num]['G_prob']
                trial_correct['M_prob'] = session_json_data[block_num]['trials'][trial_num]['M_prob']
                trial_correct['C_prob'] = session_json_data[block_num]['trials'][trial_num]['C_prob']
                trial_correct['S_prob'] = session_json_data[block_num]['trials'][trial_num]['S_prob']

                blocks_data_revised[str(block_key)]['trial_' + str(trial_num)] = trial_correct
                
                
            else:
                trial = block['trial_' + str(trial_num)]
                pre_trial_resources = trial['resources_post']
                pre_goal_progress = trial['goal_progress_post']
                pre_trial_end_time = trial['trial_end_time']

    print("Goal completion trials filled in for subject", subject_id)
    return subject_data_revised



def fill_in_goal_missing_trials(subject_data):
    blocks_data = subject_data['GoalSwitching']
    for block_num in range(len(blocks_data.keys())):
        block = blocks_data[str(block_num)]
        trial_list = [int(i.split("_")[1]) for i in block.keys() if "trial_" in i]
        trial_list = sorted(trial_list)
        max_trial_num = max(trial_list)

        for trial_num in range(max_trial_num + 1):

            trial = block.get('trial_' + str(trial_num), None)
            if trial['goal_selected'] is None:
                if trial_num > 0:
                    prev_trial = block['trial_' + str(trial_num - 1)]
                    if 'goal_selected' in prev_trial:
                        prev_goal = prev_trial['goal_selected']
                    else:
                        prev_goal = None
                    if 'resource_selected' in prev_trial:
                        prev_action = prev_trial['resource_selected']
                    else:
                        prev_action = None
                else:
                    prev_trial = None
                    prev_goal = None
                    prev_action = None
                if trial_num < max_trial_num:
                    next_trial = block['trial_' + str(trial_num + 1)]
                    if 'goal_selected' in next_trial:
                        next_goal = next_trial['goal_selected']
                    else:
                        next_goal = None
                    if 'resource_selected' in next_trial:
                        next_action = next_trial['resource_selected']
                    else:
                        next_action = None
                else:
                    next_trial = None
                    next_goal = None
                    next_action = None
                    
                    
                if prev_goal is not None and next_goal is not None:
                    if prev_goal == next_goal:
                        trial['goal_filled'] = prev_goal
                        
                    print("Sandwich missing goal trial found at block", block_num, "trial", trial_num)


                
    
    return blocks_data


if __name__ == "__main__":
    subject_id = 11
    experiment = 0
    data = get_subject_data_from_id(experiment, subject_id)
    data = fill_in_goal_completion_trials(subject_id, data)
    
    #data = fill_in_goal_missing_trials(data)
    
    


