# The learned likelihood can be compared to the likelihood obtained with an
# approximation to the analytic function. This corresponds to the likelihood
# which was used to sample the data, so matching this likelihood as closely
# as possible is the goal. To remove interdependency between R and Python in the
# code, the following scripts allows to export the relevant values for the
# evaluation data set to a Feather file.

library(rtdists)  # provides pdf and cdf
library(arrow)  # to store in feather format

#Set configuration
params_file = "assets/evaluation/parameters/01.feather"
output_dir = "assets/evaluation/analytic_likelihood"
min_rt = -5  # negative numbers indiciate lower boundary
max_rt = 5
rt.length = 1000
id_range = 1:10  # range of parameters for which the pdfs and cdfs should be stored

# Read parameter values
params = read_feather("assets/evaluation/parameters/01.feather")

# Creates vector of corresponding log-pdf and log-cdf
get_likelihood_curve <- function (rts, params, id) {
  lpdf = log(rtdists::ddiffusion(abs(rts), response=ifelse(rts > 0, "upper", "lower"), a=params$a, v=params$v, t0=params$t0, z = params$w * params$a,
                      d = 0, s=1, precision = 5))
  # RTs have to be increasing for pdiffusion, so split up lower and upper bounds
  lcdf = log(c(
    rev(rtdists::pdiffusion(-rev(rts[rts<0]), response="lower",
                            a=params$a, v=params$v, t0=params$t0, z = params$w * params$a,
                            d = 0, s=1, precision = 5)),
    rtdists::pdiffusion(rts[rts>=0], response="upper",
                        a=params$a, v=params$v, t0=params$t0, z = params$w * params$a,
                        d = 0, s=1, precision = 5)
  ))
  # There is a NaNs produced warning, but I think the corresponding values can be
  # safely set to negative infinity
  lcdf = ifelse(!is.nan(lcdf), lcdf, -Inf)
  return(data.frame(rt=rts, lpdf=lpdf, lcdf=lcdf, id=rep(id,length(lpdf))))
}

# Range of reaction times
rts = seq(min_rt, max_rt, length.out=rt.length)




# set up file paths
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
base_name = "01.feather"
file_name = file.path(output_dir, base_name)

ids = sort(unique(params$id))  # extract ids present in parameter data frame
for (id in ids[id_range]) {
  p = params[params$id == id,]
  if (id == ids[1]) {
    # initialize df with first data frame
    df = get_likelihood_curve(rts, p, id)
  } else {
    # append values to initialized data frame
    df = rbind(df, get_likelihood_curve(rts, p, id))
  }
}

# write the results to file using the feather format
write_feather(df, file_name)
