// MODEL 1
// Same MB/MF weighting across both stakes conditions & task types
// Model adapted from Kool et al. (2017), Psychological Science

data {

    int nSubs;                                      // Number of subjects
    int nTrials;                                    // Number of trials 

    int X[nSubs, 2, nTrials];                       // First stage state (1 = options 1 & 2; 2 = options 3 & 4)
    int Y[nSubs, 2, nTrials];                       // Subject choice (binary)
    int second_stage_state[nSubs, 2, nTrials];      // Second stage state shown
    real estimates[nSubs, 2, nTrials];              // Expectancy estimates
    real R[nSubs, 2, nTrials];                      // Number of points received
    int stakes[nSubs, 2, nTrials];                  // Stakes

    real RT[nSubs, 2, nTrials];                     // Reaction times (in seconds)
    real sub_min_RT[nSubs];                         // Minimum RT per subject
}

parameters {

    // Group mean - one entry per parameter
    vector[6] mu;

    // Group sigma
    vector<lower=0>[6] sigma;

    // Subject offsets
    vector[nSubs] alpha_offset;         // Learning rate
    vector[nSubs] beta_offset;          // Inverse temperature
    vector[nSubs] lambda_offset;        // Eligibility trace
    vector[nSubs] W_offset;             // MB/MF weighting
    vector[nSubs] pi_offset;            // Choice stickiness

    // DDM parameters (beta is not estimated)
    vector[nSubs] DDM_alpha_offset;     // Decision threshold
    vector[nSubs] DDM_tau_offset;       // Non-decision time
    vector[nSubs] DDM_v_offset;         // V (Pederson & Frank, 2020) - exploration parameter

}

transformed parameters {

    // Subject parameters
    vector<lower=0, upper=1>[nSubs] alpha_sub;      // Learning rate
    vector<lower=0, upper=20>[nSubs] beta_sub;      // Inverse temperature
    vector<lower=0, upper=1>[nSubs] lambda_sub;     // Eligibility trace
    vector<lower=0, upper=1>[nSubs] W_sub;          // MB/MF weighting
    vector<lower=0, upper=1>[nSubs] pi_sub;         // Choice stickiness

    // DDM parameters 
    vector<lower=0, upper=2>[nSubs] DDM_alpha_sub;      // Decision threshold
    vector<lower=0, upper=max(sub_min_RT)>[nSubs] DDM_tau_sub;      // Non-decision time - lower bound at 0.1, upper bound = 1.5
    vector<lower=0, upper=2>[nSubs] DDM_v_sub;          // V (Pederson & Frank, 2020) - exploration parameter


    for (sub in 1:nSubs) {
        alpha_sub[sub]          = Phi_approx(mu[1] + sigma[1] * alpha_offset[sub]);
        beta_sub[sub]           = Phi_approx(mu[2] + sigma[2] * beta_offset[sub]) * 20;
        lambda_sub[sub]         = Phi_approx(mu[3] + sigma[3] * lambda_offset[sub]);
        W_sub[sub]              = Phi_approx(mu[4] + sigma[4] * W_offset[sub]);
        pi_sub[sub]             = Phi_approx(mu[5] + sigma[5] * pi_offset[sub]);

        // DDM
        DDM_alpha_sub[sub]    = Phi_approx(mu[6] + sigma[6] * DDM_alpha_offset[sub]) * 2;
        DDM_tau_sub[sub]     = Phi_approx(mu[7] + sigma[7] * DDM_tau_offset[sub]) * sub_min_RT[sub];
        DDM_v_sub[sub]   = Phi_approx(mu[8] + sigma[8] * DDM_v_offset[sub]) * 2;
    }

}
model {

    // Group mean
    mu                          ~ normal(0, 1);

    // Group sigma
    sigma                       ~ normal(0, 1);

    // Subject parameters
    alpha_offset                ~ normal(0, 1);
    beta_offset                 ~ normal(0, 1);
    lambda_offset               ~ normal(0, 1);
    W_offset                    ~ normal(0, 1);
    pi_offset                   ~ normal(0, 1);

    DDM_alpha_offset            ~ normal(0, 1);
    DDM_tau_offset              ~ normal(0, 1);
    DDM_v_offset                ~ normal(0, 1);

    // Used for PE
    real pe_second_stage;

    // Used to record choices
    int trial_start_state;
    int trial_choice;

    // Loop over subjects
    for (sub in 1:nSubs) {

        // Loop over task types
        for (task in 1:2) {

            // Initialise Q values      
            vector[2] mf_Q[2];
            vector[2] mf_Q_second_stage;
            vector[2] mb_Q[2];
            vector[2] hybrid_Q[2];
            
            // Give Q values a starting value of 0.5
            for (s in 1:2){ 
                mf_Q[s] = rep_vector(0.5, 2); 
            }

            for (s in 1:2){ 
                mb_Q[s] = rep_vector(0.5, 2); 
            }

            mf_Q_second_stage = rep_vector(0.5, 2);

            // Loop over trials
            for (trial in 1:nTrials) {

                if (stakes[sub, task, trial] > 0) {

                    // MB Q values for start options are equal to MF values of second stage states
                    mb_Q[1, 1] = mf_Q_second_stage[1];
                    mb_Q[1, 2] = mf_Q_second_stage[2];
                    mb_Q[2, 1] = mf_Q_second_stage[2];
                    mb_Q[2, 2] = mf_Q_second_stage[1];

                    // Calculate hybrid values
                    hybrid_Q[1] = W_sub[sub] * mb_Q[1] + (1 - W_sub[sub]) * mf_Q[1];
                    hybrid_Q[2] = W_sub[sub] * mb_Q[2] + (1 - W_sub[sub]) * mf_Q[2];

                    // Add choice stickiness
                    if (trial != 1)
                        hybrid_Q[trial_start_state, trial_choice] = hybrid_Q[trial_start_state, trial_choice] + pi_sub[sub];

                    // Get choice on this trial
                    trial_choice = Y[sub,task,trial];
                    trial_start_state = X[sub,task,trial];

                    // Likelihood of choice
                    (trial_choice - 1) ~ bernoulli_logit( beta_sub[sub] *  (hybrid_Q[trial_start_state, 2] - hybrid_Q[trial_start_state, 1]));

                    // Likelihood of RT
                    if (Y[sub,task,trial] == 1) {
                        RT[sub, task, trial] ~ wiener(DDM_alpha_sub[sub], DDM_tau_sub[sub], 1 - (0.5 * DDM_alpha_sub[sub]), -(DDM_v_sub[sub] * (hybrid_Q[X[sub,task,trial], 2] - hybrid_Q[X[sub,task,trial], 1])));
                    }
                    if (Y[sub,task,trial] == 2) {
                        RT[sub, task, trial] ~ wiener(DDM_alpha_sub[sub], DDM_tau_sub[sub], 0.5 * DDM_alpha_sub[sub], DDM_v_sub[sub] * (hybrid_Q[X[sub,task,trial], 2] - hybrid_Q[X[sub,task,trial], 1]));
                    }

                    // Update first level MF values
                    mf_Q[trial_start_state, trial_choice] += alpha_sub[sub] * (mf_Q_second_stage[second_stage_state[sub,task,trial]] - mf_Q[trial_start_state, trial_choice] );

                    // Update second level MF values
                    pe_second_stage = (R[sub,task,trial] - mf_Q_second_stage[second_stage_state[sub,task,trial]] );
                    mf_Q_second_stage[second_stage_state[sub,task,trial]]  += alpha_sub[sub] * pe_second_stage;

                    // Eligibility trace
                    mf_Q[trial_start_state, trial_choice]    += lambda_sub[sub] * alpha_sub[sub] * pe_second_stage;

                }


            }
        }
    }
}

generated quantities {

    // Log likelihood
    real log_lik[nSubs,2,nTrials];   

    // Used for PE
    real pe_second_stage;

    // Used to record choices
    int trial_start_state;
    int trial_choice;
    
    // Loop over subjects
    for (sub in 1:nSubs) {

        // Loop over task types
        for (task in 1:2) {

            // Initialise Q values      
            vector[2] mf_Q[2];
            vector[2] mf_Q_second_stage;
            vector[2] mb_Q[2];
            vector[2] hybrid_Q[2];
            
            // Give Q values a starting value of 0.5
            for (s in 1:2){ 
                mf_Q[s] = rep_vector(0.5, 2); 
            }

            for (s in 1:2){ 
                mb_Q[s] = rep_vector(0.5, 2); 
            }

            mf_Q_second_stage = rep_vector(0.5, 2);

            // Loop over trials
            for (trial in 1:nTrials) {

                if (stakes[sub, task, trial] > 0)  {
                    
                    // MB Q values for start options are equal to MF values of second stage states
                    mb_Q[1, 1] = mf_Q_second_stage[1];
                    mb_Q[1, 2] = mf_Q_second_stage[2];
                    mb_Q[2, 1] = mf_Q_second_stage[2];
                    mb_Q[2, 2] = mf_Q_second_stage[1];

                    // Calculate hybrid values
                    hybrid_Q[1] = W_sub[sub] * mb_Q[1] + (1 - W_sub[sub]) * mf_Q[1];
                    hybrid_Q[2] = W_sub[sub] * mb_Q[2] + (1 - W_sub[sub]) * mf_Q[2];

                    // Add choice stickiness
                    if (trial != 1)
                        hybrid_Q[trial_start_state, trial_choice] = hybrid_Q[trial_start_state, trial_choice] + pi_sub[sub];

                    // Get choice on this trial
                    trial_choice = Y[sub,task,trial];
                    trial_start_state = X[sub,task,trial];

                    // Likelihood of choice
                    log_lik[sub,task,trial] = bernoulli_logit_lpmf(trial_choice - 1 | beta_sub[sub] *  (hybrid_Q[trial_start_state, 2] - hybrid_Q[trial_start_state, 1]));

                    // Likelihood of RT -- NOT IMPEMENTED PROPERLY YET
                    // if (Y[sub,task,trial] == 1) {
                    //     RT[sub, task, trial] ~ wiener(DDM_alpha_sub[sub], DDM_tau_sub[sub], 1 - (0.5 * DDM_alpha_sub[sub]), -(DDM_v_sub[sub] * (hybrid_Q[X[sub,task,trial], 2] - hybrid_Q[X[sub,task,trial], 1])));
                    // }
                    // if (Y[sub,task,trial] == 2) {
                    //     RT[sub, task, trial] ~ wiener(DDM_alpha_sub[sub], DDM_tau_sub[sub], 0.5 * DDM_alpha_sub[sub], DDM_v_sub[sub] * (hybrid_Q[X[sub,task,trial], 2] - hybrid_Q[X[sub,task,trial], 1]));
                    // }


                    // Update first level MF values
                    mf_Q[trial_start_state, trial_choice] += alpha_sub[sub] * (mf_Q_second_stage[second_stage_state[sub,task,trial]] - mf_Q[trial_start_state, trial_choice] );

                    // Update second level MF values
                    pe_second_stage = (R[sub,task,trial] - mf_Q_second_stage[second_stage_state[sub,task,trial]] );
                    mf_Q_second_stage[second_stage_state[sub,task,trial]]  += alpha_sub[sub] * pe_second_stage;

                    // Eligibility trace
                    mf_Q[trial_start_state, trial_choice]    += lambda_sub[sub] * alpha_sub[sub] * pe_second_stage;
                }

                else {
                    log_lik[sub, task, trial]  = 0;
                }

            }
        }
    }
}