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

model {
  // prior
  a ~ normal(1, 1)T[0.5,3];
  v ~ normal(0, 10)T[-5, 5];
  t0 ~ normal(0.4, 0.2)T[0.2,1.0];
  w ~ normal(0.5, 0.1)T[0.3,0.7];
  // likelihood
  for (i in 1:N) {
    if (resp[i] == 1) {
    // upper threshold
      target += wiener_lpdf(rt[i] | a, t0, w, v);
    } else {
      // lower threshold (mirror drift and starting point)
      target += wiener_lpdf(rt[i] | a, t0, 1-w, -v);
    }
    target += normal_lpdf(dummy[i]|0, 1);
  }
}
