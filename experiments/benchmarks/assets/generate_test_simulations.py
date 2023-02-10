import sys, os
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join('../../../BayesFlow')))
from bayesflow import benchmarks

# Creates a re-usable test set of data sets from the different benchmark models
# The data sets will be pre-configured for directly entering a BayesFlow amortizer

N_TEST = 1000
SEED = 42

for benchmark_name in benchmarks.available_benchmarks:
    benchmark = benchmarks.Benchmark(benchmark_name, seed=SEED)
    sims = benchmark.generative_model(N_TEST)
    if benchmark_name in ['slcp_distractors', 'bernoulli_glm_raw', 'sir', 'lotka_volterra']:
        sims_conf = benchmark.configurator(sims, as_summary_condition=True)
    else:
        sims_conf = benchmark.configurator(sims)
    with open(f'test_data/{benchmark_name}_test.pkl', 'wb') as f:
        pickle.dump(sims_conf, f)
