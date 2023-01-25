library(rstan)
library(bridgesampling)
library(loo)
library(arrow)

rstan_options(auto_write = TRUE)
options(mc.cores = 4)

SEED = 42
set.seed(SEED)

### Set-up simulation hyperparameters
N_sim = 100
N_trials = 100

### Set-up Stan hyperparameters
n_draws = 10000
n_chains = 4
n_cores = 4
warmup = 1000
adapt_delta = .95
max_treedepth = 15

### Prepare placeholders for elpd and bs
start_sim = 1
# load intermediate results in case of interruptions
bmc_results_rds = paste0('./assets/bmc_results_', N_trials, '.rds')
if (file.exists(bmc_results_rds)) {
  bmc_results = readRDS(bmc_results_rds)
  start_sim = max(which(complete.cases(bmc_results))) + 1
} else {
  bmc_results = matrix(rep(NA, N_sim * 3), nrow=N_sim, ncol=3)
  bmc_results = data.frame(bmc_results)
  colnames(bmc_results) = c('ELPD', 'ELPD_SE', 'LML')
}

### Extract pre-simulated data
data_all = read_feather("./assets/evaluation/sim_data/01.feather")

### Calculate LML and ELPD
for (n in start_sim:N_sim) {
  print(n)
  
  ### Select data set
  data = data_all[data_all$id == n,]
  data = data[1:N_trials,]
  
  rt = data$rt
  resp = data$resp
  ### Create dummy variable, set it to zero (do the same in BayesFlow)
  dummy = rep(0, N_trials)
  
  ## Prepare stan input list and initial values for ndt
  stan_data = list(rt = rt, resp = resp, dummy = dummy, N=N_trials)
  
  ## MCMC Sampling
  fit = stan(
    file = './assets/stan/avt0w-bridge_sampling.stan',
    data = stan_data, 
    chains = n_chains, 
    warmup = warmup, 
    iter = n_draws, 
    cores = n_cores,
    control=list(adapt_delta=adapt_delta, max_treedepth=max_treedepth),
    seed = SEED
  )
  
  ## ELPD
  elpd_results = elpd(extract_log_lik(fit, parameter_name = 'log_lik', merge_chains = TRUE))
  elpd_val = elpd_results$estimates[1, 1]
  elpd_se = elpd_results$estimates[1, 2] 
  
  ## Bridge Sampling
  bridge_results = bridge_sampler(fit, cores=4)
  lml = bridge_results$logml
  
  ## Collect everything into placeholder matrix
  bmc_results[n, "ELPD"] = elpd_val
  bmc_results[n, "ELPD_SE"] = elpd_se
  bmc_results[n, "LML"] = lml

  # Store bmc results
  saveRDS(bmc_results, paste0('./assets/bmc_results_', N_trials, '.rds'))
  write.table(bmc_results, paste0('./assets/bmc_results_', N_trials, '.csv'), row.names = F)
  rm("fit", "data")
}
