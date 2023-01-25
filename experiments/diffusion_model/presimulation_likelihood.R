library(truncnorm)  # needed for priors
library(rtdists)  # simulator
library(foreach)
library(parallel)
library(arrow)  # to store in feather format

#Set configuration
output_dir = "data/osfstorage/likelihood"
num_trials = 500
cores = 32

## Function to simulate one experiment
simulate_exp = function(k=2, N=100, a=1, w=0.5, v=c(+1,-1), t0=0.2, sw=0, sv=0, st0=0) {
  data = data.frame();
  for (c in 1:k) {
    N.c  = ifelse(is.na(N[c]),  N[1],  N[c])
    a.c  = ifelse(is.na(a[c]),  a[1],  a[c])
    w.c = ifelse(is.na(w[c]), w[1], w[c])
    v.c  = ifelse(is.na(v[c]),  v[1],  v[c])
    t0.c = ifelse(is.na(t0[c]), t0[1], t0[c])
    
    sample = rtdists::rdiffusion(N.c, a=a.c, v=v.c, t0=t0.c, z = w.c * a.c,
                                 d = 0, sz = sw * a.c, sv=sv, st0=st0, s=1,
                                 precision = 5, method="fastdm")
    data = rbind(
      data,
      data.frame(
        cnd = c,
        resp = ifelse(sample$response == "upper", 1, 0),
        rt = sample$rt
      )
    )
    
  }
  return(data)
}

# Function to create a single dataset
create_dataset <- function (subj, num_trials) {
  # Uniform distribution to cover all regions of interest equally well.
  # We learn a bit more than the region of interest to make sure the values
  # at the boundaries are learned well.
  params = list(
    a   = runif(1, 0.4, 3.1),
    t0  = runif(1, 0.1, 1.1),
    w   = runif(1, 0.28, 0.72),
    v   = runif(1, -5.5, 5.5)
  )
  
  # convert parameter list into data frame
  params_df = data.frame(a=params$a, t0 = params$t0, w = params$w,
                         v = params$v, id=subj)
  
  # simulate a single condition based on those parameters
  sim.data = simulate_exp(k = 1, N=num_trials, t0=params$t0, st0=0.0, a=params$a,
                          w = params$w, sw=0.0, v = params$v, sv=0.0)
  sim.data$id = rep(subj, nrow(sim.data))
  return(list(
    params = params_df,
    sim.data = sim.data
  ))
}

# Function to create a whole epoch of data sets
simulate_epoch <- function(num_trials, sim_per_epoch, name, epoch, output_dir) {
  param_names = c("v", "a", "t0", "w", "id")
  sim_names = c("cnd", "resp", "rt", "id")
  
  # set up file paths
  base_dir = file.path(output_dir, name)
  params_dir = file.path(base_dir, "parameters")
  sim_dir = file.path(base_dir, "sim_data")
  dir.create(params_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(sim_dir, recursive = TRUE, showWarnings = FALSE)
  base_name = paste0(formatC(epoch, width = 2, flag = "0"), ".feather")
  param_file_name = file.path(params_dir, base_name)
  simdata_file_name = file.path(sim_dir, base_name)

  # custom function to bind results from different threads together
  cfun <- function(a, b) {
    a[["sim.data"]] <- rbind(a[["sim.data"]], b[["sim.data"]])
    a[["params"]] <- rbind(a[["params"]], b[["params"]])
    return(a)
  }
  # calculate size of each split
  split_size = sim_per_epoch / cores
  # simulate the data sets in parallel using the foreach package
  res = foreach(sim_split=1:(sim_per_epoch / split_size), .combine = "cfun", .export = c("create_dataset")) %dopar% {
    params = data.frame(matrix(nrow = split_size, ncol = length(param_names)))
    colnames(params) = param_names
    sim.data = data.frame(matrix(nrow=split_size * num_trials, ncol = length(sim_names)))
    colnames(sim.data) = sim_names
    idx = 1
    for (sim in ((sim_split - 1) * split_size + 1):(sim_split * split_size)) {
      if (idx %% 100 == 0) {
        # can't write to stdout from threads -> write progress to log file
        sink(paste0("log/simout_log_", epoch, ".txt"))
        cat(paste0("epoch: ", epoch, " batch: ", idx, "/", split_size, "\n"))
        sink()
      }
      dataset = create_dataset(sim, num_trials)
      params[idx,] = dataset[["params"]][,param_names]
      sim.data[((idx - 1) * num_trials + 1):(idx * num_trials),] = dataset[["sim.data"]][,sim_names]
      idx = idx + 1
    }
    
    return(list(
      params = params,
      sim.data = sim.data
    ))
  }
  # write the results to file using the feather format
  write_feather(res[["params"]], param_file_name)
  write_feather(res[["sim.data"]], simdata_file_name)
}


## Simulating in parallel
cl <- parallel::makeForkCluster(cores)
doParallel::registerDoParallel(cl)

dir.create("log", showWarnings = FALSE)

# training data set
training_epochs = 10
sim_per_epoch = 128000  # should be multiple of cores
for (epoch in 1:training_epochs) {
  cat(paste0("Starting epoch ", epoch), "\n")
  simulate_epoch(num_trials = num_trials, sim_per_epoch = sim_per_epoch, name = "training",
                 epoch = epoch, output_dir = output_dir)
}
