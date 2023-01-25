data {
  int<lower=0> N;                       // No trials
  array[N] real rt;                     // response times (seconds)
  array[N] real dummy;                  // dummy variable
  array[N] int<lower=0, upper=1> resp;  // responses (0,1)
}

parameters {
  real<lower=0.5, upper=3> a; // boundary separation
  real<lower=-5,upper=5> v;  // drift rate
  real<lower=0.2, upper=min([min(rt)-1e-12, 1.0])> t0; // non-decision-time
  real<lower=0.3, upper=0.7> w;  // relative starting point
}

transformed parameters {
  // calculate likelihood here to store it for loo
  array[N] real log_lik;

  for (i in 1:N) {
    if (resp[i] == 1) {
    // upper threshold
      log_lik[i]= wiener_lpdf(rt[i] | a, t0, w, v);
    } else {
      // lower threshold (mirror drift and starting point)
      log_lik[i] = wiener_lpdf(rt[i] | a, t0, 1-w, -v);
    }
    // add likelihood of dummy variable to allow direct comparison with BayesFlow
    log_lik[i] += normal_lpdf(dummy[i]|0, 1);
  }
}

model {
  // For use with bridgesampling, we have to use the `target +=` notation.
  // As the normal truncation via `T[lower,upper]` isn't available in this notation,
  // we have to reformulate it manually. See https://arxiv.org/pdf/1710.08162.pdf
  // for details

  // a ~normal(1, 1)T[0.5,3]
  target += normal_lpdf(a | 1, 1) -
    log_diff_exp(normal_lcdf(3 | 1, 1),
                 normal_lcdf(0.5 | 1, 1));
  // v ~ normal(0, 10)T[-5, 5];
  target += normal_lpdf(v | 0, 10) -
    log_diff_exp(normal_lcdf(5 | 0, 10),
                 normal_lcdf(-5 | 0, 10));
  // t0 ~ normal(0.4, 0.2)T[0.2,1.0];
  target += normal_lpdf(t0 | 0.4, 0.2) -
    log_diff_exp(normal_lcdf(1.0 | 0.4, 0.2),
                 normal_lcdf(0.2 | 0.4, 0.2));
  // w ~ normal(0.5, 0.1)T[0.3,0.7];
  target += normal_lpdf(w | 0.5, 0.1) -
    log_diff_exp(normal_lcdf(0.7 | 0.5, 0.1),
                 normal_lcdf(0.3 | 0.5, 0.1));
  // add pre-computed likelihood to target
  for (i in 1:N) {
    target += log_lik[i];
  }
}
