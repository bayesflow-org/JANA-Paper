Due to its large size, this folder is hosted on the Open Science Framework (OSF) at [https://osf.io/2nbk6/](https://osf.io/2nbk6/).

To download it, you can use the "Download as zip" option and then unpack the obtained file into this folder.

Alternatively, you can use the [osfclient](https://pypi.org/project/osfclient/) to clone the project. First, install `osfclient` via pip.

```
$ pip install osfclient
```

Next, you can use the command

```
$ osf clone .
```

to fetch all files from the project. As the project has a size of about 6GB, this might take some time.

The final structure should look like this:
`experiments/diffusion_model/data/osfstorage/`

- `fits_pymc/`: Files created by `pymc_sample.py`
- `fits_stan/`: Files created by `stan_sample.py`
- `likelihood/`: Training data for the likelihood network
- `posterior/`: Training and evaluation data for the posterior network

