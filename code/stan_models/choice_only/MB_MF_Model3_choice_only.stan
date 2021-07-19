// MODEL 3
// Different learning rate and MB/MF weighting for safety & reward
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

}

parameters {

    // Group mean - one entry per parameter
    vector[8] mu;

    // Group sigma
    vector<lower=0>[8] sigma;

    // Subject offsets
    vector[nSubs] alpha_safety_offset;         // Learning rate
    vector[nSubs] alpha_reward_offset;         // Learning rate
    vector[nSubs] beta_offset;          // Inverse temperature
    vector[nSubs] lambda_offset;        // Eligibility trace
    vector[nSubs] W_safety_offset;             // MB/MF weighting
    vector[nSubs] W_reward_offset;             // MB/MF weighting
    vector[nSubs] pi_offset;            // Choice stickiness

}

transformed parameters {

    // Subject parameters
    vector<lower=0, upper=1>[nSubs] alpha_safety_sub;      // Learning rate
    vector<lower=0, upper=1>[nSubs] alpha_reward_sub;      // Learning rate
    vector<lower=0, upper=20>[nSubs] beta_sub;      // Inverse temperature
    vector<lower=0, upper=1>[nSubs] lambda_sub;     // Eligibility trace
    vector<lower=0, upper=1>[nSubs] W_safety_sub;          // MB/MF weighting
    vector<lower=0, upper=1>[nSubs] W_reward_sub;          // MB/MF weighting
    vector<lower=0, upper=1>[nSubs] pi_sub;         // Choice stickiness

    // Matrix of parameters that differ according to task type - easier to select later
    real<lower=0, upper=1> alpha_sub_matrix[nSubs, 2];
    real<lower=0, upper=1> W_sub_matrix[nSubs, 2];

    for (sub in 1:nSubs) {
        alpha_safety_sub[sub]           = Phi_approx(mu[1] + sigma[1] * alpha_safety_offset[sub]);
        alpha_reward_sub[sub]           = Phi_approx(mu[2] + sigma[2] * alpha_reward_offset[sub]);
        beta_sub[sub]               = Phi_approx(mu[3] + sigma[3] * beta_offset[sub]) * 20;
        lambda_sub[sub]             = Phi_approx(mu[4] + sigma[4] * lambda_offset[sub]);
        W_safety_sub[sub]               = Phi_approx(mu[5] + sigma[5] * W_safety_offset[sub]);
        W_reward_sub[sub]               = Phi_approx(mu[6] + sigma[6] * W_reward_offset[sub]);
        pi_sub[sub]                 = Phi_approx(mu[7] + sigma[7] * pi_offset[sub]);

        // Add to matrix for easier selection later
        alpha_sub_matrix[sub, 2]  = alpha_safety_sub[sub];
        alpha_sub_matrix[sub, 1]  = alpha_reward_sub[sub];
        W_sub_matrix[sub, 2]      = W_safety_sub[sub];
        W_sub_matrix[sub, 1]      = W_reward_sub[sub];

    }

}
model {

    // Group mean
    mu                              ~ normal(0, 1);

    // Group sigma
    sigma                           ~ normal(0, 1);

    // Subject parameters
    alpha_safety_offset                 ~ normal(0, 1);
    alpha_reward_offset                 ~ normal(0, 1);
    beta_offset                     ~ normal(0, 1);
    lambda_offset                   ~ normal(0, 1);
    W_safety_offset                     ~ normal(0, 1);
    W_reward_offset                     ~ normal(0, 1);
    pi_offset                       ~ normal(0, 1);

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
                    hybrid_Q[1] = W_sub_matrix[sub, task] * mb_Q[1] + (1 - W_sub_matrix[sub, task]) * mf_Q[1];
                    hybrid_Q[2] = W_sub_matrix[sub, task] * mb_Q[2] + (1 - W_sub_matrix[sub, task]) * mf_Q[2];

                    // Add choice stickiness
                    if (trial != 1)
                        hybrid_Q[trial_start_state, trial_choice] = hybrid_Q[trial_start_state, trial_choice] + pi_sub[sub];

                    // Get choice on this trial
                    trial_choice = Y[sub,task,trial];
                    trial_start_state = X[sub,task,trial];

                    // Likelihood of choice
                    (trial_choice - 1) ~ bernoulli_logit( beta_sub[sub] *  (hybrid_Q[trial_start_state, 2] - hybrid_Q[trial_start_state, 1]));

                    // Update first level MF values
                    mf_Q[trial_start_state, trial_choice] += alpha_sub_matrix[sub, task] * (mf_Q_second_stage[second_stage_state[sub,task,trial]] - mf_Q[trial_start_state, trial_choice] );

                    // Update second level MF values
                    pe_second_stage = (R[sub,task,trial] - mf_Q_second_stage[second_stage_state[sub,task,trial]] );
                    mf_Q_second_stage[second_stage_state[sub,task,trial]]  += alpha_sub_matrix[sub, task] * pe_second_stage;

                    // Eligibility trace
                    mf_Q[trial_start_state, trial_choice]    += lambda_sub[sub] * alpha_sub_matrix[sub, task] * pe_second_stage;

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
                    hybrid_Q[1] = W_sub_matrix[sub, task] * mb_Q[1] + (1 - W_sub_matrix[sub, task]) * mf_Q[1];
                    hybrid_Q[2] = W_sub_matrix[sub, task] * mb_Q[2] + (1 - W_sub_matrix[sub, task]) * mf_Q[2];

                    // Add choice stickiness
                    if (trial != 1)
                        hybrid_Q[trial_start_state, trial_choice] = hybrid_Q[trial_start_state, trial_choice] + pi_sub[sub];

                    // Get choice on this trial
                    trial_choice = Y[sub,task,trial];
                    trial_start_state = X[sub,task,trial];

                    // Likelihood of choice
                    log_lik[sub,task,trial] = bernoulli_logit_lpmf(trial_choice - 1 | beta_sub[sub] *  (hybrid_Q[trial_start_state, 2] - hybrid_Q[trial_start_state, 1]));


                    // Update first level MF values
                    mf_Q[trial_start_state, trial_choice] += alpha_sub_matrix[sub, task] * (mf_Q_second_stage[second_stage_state[sub,task,trial]] - mf_Q[trial_start_state, trial_choice] );

                    // Update second level MF values
                    pe_second_stage = (R[sub,task,trial] - mf_Q_second_stage[second_stage_state[sub,task,trial]] );
                    mf_Q_second_stage[second_stage_state[sub,task,trial]]  += alpha_sub_matrix[sub, task] * pe_second_stage;

                    // Eligibility trace
                    mf_Q[trial_start_state, trial_choice]    += lambda_sub[sub] * alpha_sub_matrix[sub, task] * pe_second_stage;
                }

                else {
                    log_lik[sub, task, trial]  = 0;
                }

            }
        }
    }
}