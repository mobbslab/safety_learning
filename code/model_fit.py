import pandas as pd
import numpy as np
import arviz as az
import os

from cmdstanpy import cmdstan_path, CmdStanModel

if __name__ == "__main__":
    
    # Get slurm run ID
    try:
        runID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    except:
        runID = 1    

    model_type = 'choice_only'

    # Set up output directory
    out_dir = 'data/model_fitting/model{0}/{1}'.format(runID, model_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get data
    df = pd.read_csv('data/all_confidence_estimate_combo_df.txt', sep='\t')

    # Extract data fpr Stam
    n_subs = len(df['sub'].unique())
    n_trials = df['trial_nr'].max() + 1

    choices = np.zeros((n_subs, 2, n_trials))
    points_received = np.zeros((n_subs, 2, n_trials))
    start_state = np.zeros((n_subs, 2, n_trials))
    stakes = np.zeros((n_subs, 2, n_trials))  # Use -999 to signify missing trials
    estimate = np.zeros((n_subs, 2, n_trials))
    rt = np.zeros((n_subs, 2, n_trials))
    sub_min_rt = np.zeros(n_subs)

    subject_ids = sorted(df['sub'].unique())

    for n, sub in enumerate(subject_ids):
        sub_df = df[df['sub'] == sub]
        
        min_rt = np.inf

        for task_type_id in [0, 1]:

            sub_task_type_df = sub_df[sub_df['task_type'] == task_type_id]
            sub_task_type_df = sub_task_type_df.sort_values('trial_nr')

            # Trials with responses
            trial_numbers = sub_task_type_df['trial_nr'].values

            choices[n, task_type_id, trial_numbers] = sub_task_type_df['choice1'].values[:n_trials]
            points_received[n, task_type_id, trial_numbers] = sub_task_type_df['points'].values[:n_trials]
            start_state[n, task_type_id, trial_numbers] = sub_task_type_df['state1'].values[:n_trials]
            stakes[n, task_type_id, trial_numbers] = sub_task_type_df['stake'].values[:n_trials]
            estimate[n, task_type_id, trial_numbers] = sub_task_type_df['estimate'].str.extract('([0-9])')[0].values[:n_trials].astype(float)
            rt[n, task_type_id, trial_numbers] = sub_task_type_df['rt1'].values[:n_trials]
            
            if np.min(sub_task_type_df['rt1'].values[:n_trials]) < min_rt:
                min_rt = np.min(sub_task_type_df['rt1'].values[:n_trials])
                
        sub_min_rt[n] = min_rt

            
    estimate[np.isnan(estimate)] = -999

    choices_binary = 1 - (choices % 2).astype(int)

    stakes[stakes == 4] = 2

    second_stage_state = np.zeros((n_subs, 2, n_trials), dtype=int) + 5
    second_stage_state[choices == 1] = 5
    second_stage_state[choices == 2] = 6
    second_stage_state[choices == 3] = 6
    second_stage_state[choices == 4] = 5

    data = {'nSubs': n_subs,
            'nTrials': n_trials,
            'X': np.maximum(start_state.astype(int), 1),
            'Y': choices_binary.astype(int) + 1,
            'second_stage_state': second_stage_state.astype(int) -4,
            "estimates": estimate,
            "RT": rt / 1000,
            "sub_min_RT": sub_min_rt / 1000,
            'R': points_received,
            'stakes': stakes.astype(int)}

    # Model
    model = CmdStanModel(stan_file='code/stan_models/{1}/MB_MF_Model{0}_{1}.stan'.format(runID, model_type),  cpp_options={'STAN_THREADS': False})

    # Fit the real model to the simulated data
    fit = model.sample(data=data, chains=4, seed=123, iter_sampling=4000, show_progress=True, refresh=2, output_dir=out_dir)

    # Convert to Arviz inference data
    fit_data = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik")

    param_names = [i for i in list(fit_data.posterior.keys()) if 'sub' in i and not 'offset' in i and not 'matrix' in i]
    summary = az.summary(fit_data, var_names=param_names, hdi_prob=0.95)

    # Create param and subject ID columns
    summary['param'] = summary.index.str.extract('(.+_sub)').values
    summary['sub'] = np.tile(subject_ids, len(param_names))

    # Add WAIC
    summary['WAIC'] = az.waic(fit_data).waic

    # Save
    summary.to_csv(os.path.join(out_dir, 'model{0}_{1}_parameter_estimates.csv'.format(runID, model_type)), index=False)

    # Save inference data
    az.to_netcdf(fit_data, os.path.join(out_dir, 'model{0}_{1}_fit_data'.format(runID, model_type)))
