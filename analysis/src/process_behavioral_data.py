"""Reconstruct inventory and goal progress from raw task records."""

from datautils import *



class ProcessBehavioralData:
    def __init__(self, data):
        self.data = data
    
    def init_goal_progress(self):

        goal_progress = {
            "SH": {"G": [0, 3], "M": [0, 3]},
            "BR": {"G": [0, 2], "C": [0, 1], "S": [0, 3]},
            "HO": {"C": [0, 3], "M": [0, 2], "S": [0, 1]}
        }
        return goal_progress

    def init_resources(self):
        resources = {
            "G": 0,
            "M": 0,
            "C": 0,
            "S": 0
        }
        return resources

    def update_resources(self, resources, resource_token):
        if resource_token in resources:
            if resources[resource_token] < 3:
                resources[resource_token] += 1
        return resources

    def update_goal_progress(self, resources):
        goal_progress = {
            "SH": {"G": [0, 3], "M": [0, 3]},
            "BR": {"G": [0, 2], "C": [0, 1], "S": [0, 3]},
            "HO": {"C": [0, 3], "M": [0, 2], "S": [0, 1]}
        }
        for goal in goal_progress:
            for key in goal_progress[goal]:
                goal_progress[goal][key][0] = min(resources[key],
                                                  goal_progress[goal][key][1])
        return goal_progress



    def get_goal_completion_status(self, goal_progress):
        goal_completion = False
        goal_completed = None
        for goal in goal_progress:
            goal_status = True
            for key in goal_progress[goal]:
                if goal_progress[goal][key][0] < goal_progress[goal][key][1]:
                    goal_status = False
            if goal_status:
                goal_completion = True
                goal_completed = goal
                break
        return goal_completion, goal_completed
        
    def process_data(self):
        """
        Process the behavioral data for the subject.
        
        Args:
            data (dict): The raw data for the subject.
        
        Returns:
            dict: Processed data.
        """
        def differ_in_more_than_one(dict1, dict2):
            # Find common keys
            common_keys = dict1.keys() & dict2.keys()
            
            # Count keys where values differ
            diff_count = sum(1 for k in common_keys if dict1[k] != dict2[k])
            
            # Also count keys that exist in one dict but not the other
            diff_count += len(dict1.keys() ^ dict2.keys())
            
            return diff_count > 1

        prev_trial_goal_progress = self.init_goal_progress()
        prev_trial_resources =  self.init_resources()

        block_type_data = {}
        block_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        blocks = self.data['GoalSwitching']
        for block_num in range(9):
            goal_completion_trials = []
            # Process each block of data as needed
            block_data = blocks[str(block_num)]
            # Do something with block_data
            trial_list = [key for key in block_data.keys() if 'trial_' in key]
            # Sort by the numeric ID after "trial_"
            trial_list = sorted(trial_list, key=lambda k: int(k.split("_")[1]))
            for trial_str in trial_list:
                trial_num = int(trial_str.split('_')[1])
                trial_data = block_data[trial_str]
                trial_data['resources_pre'] = trial_data['resources'].copy()
                trial_data['goal_progress_pre'] = self.update_goal_progress(trial_data['resources_pre'])
                if trial_data['block_type'] not in block_type_data:
                    block_type_data[trial_data['block_type']] = {
                        'G_prob': trial_data['G_prob'],
                        'S_prob': trial_data['S_prob'],
                        'M_prob': trial_data['M_prob'],
                        'C_prob': trial_data['C_prob'],
                    }

                block_counts[trial_data['block_type']] += 1

                if 'action_selected' not in trial_data:
                    trial_data['action_selected'] = None
                    trial_data['resources_post'] = trial_data['resources_pre'].copy()
                    trial_data['goal_progress_post'] = trial_data['goal_progress_pre'].copy()
                    continue

                trial_data['goal_progress_post'] = self.update_goal_progress(trial_data['resources_post'])

                if 'probe_selected' not in trial_data:
                    trial_data['probe_selected'] = None
                trial_data['goal_selected'] = trial_data['probe_selected']
                
                if differ_in_more_than_one(trial_data['resources'], prev_trial_resources):
                    goal_completion_trials.append(trial_num)
                    goal_completion, goal_completed = self.get_goal_completion_status(prev_trial_goal_progress)
                    trial_data['goal_completion'] = goal_completion
                    trial_data['goal_completed'] = goal_completed
                else:
                    trial_data['goal_completion'] = False
                    trial_data['goal_completed'] = None
                    

                if trial_data['resource_token'] != 'e':
                    trial_data['action_outcome'] = 1
                else:
                    trial_data['action_outcome'] = 0

                trial_data['goal_selected'] = trial_data['probe_selected']
                trial_data['resource_selected'] = trial_data['action_selected']

                prev_trial_resources = trial_data['resources_pre']
                prev_trial_goal_progress = trial_data['goal_progress_pre']


        self.data['block_type_data'] = block_type_data
        self.data['block_counts'] = block_counts

        return self.data



if __name__ == "__main__":
    # Example usage
    data_type = 'online'  # or 'online'
    #for subject_id in range(len(subject_names)):
    subject_id = 0
    process_data = ProcessBehavioralData(subject_id)
    process_data.process_data()