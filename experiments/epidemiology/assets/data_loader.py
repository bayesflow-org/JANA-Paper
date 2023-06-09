import pandas as pd
import numpy as np
import datetime


def load_data():
    confirmed_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    recovered_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    dead_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

    confirmed_cases = pd.read_csv(confirmed_cases_url, sep=",")
    recovered_cases = pd.read_csv(recovered_cases_url, sep=",")
    dead_cases = pd.read_csv(dead_cases_url, sep=",")

    date_data_begin = datetime.date(2020, 3, 1)
    date_data_end = datetime.date(2020, 5, 21)

    format_date = lambda date_py: "{}/{}/{}".format(
        date_py.month, date_py.day, str(date_py.year)[2:4]
    )
    date_formatted_begin = format_date(date_data_begin)
    date_formatted_end = format_date(date_data_end)

    cases_obs = np.array(
        confirmed_cases.loc[
            confirmed_cases["Country/Region"] == "Germany",
            date_formatted_begin:date_formatted_end,
        ]
    )[0]
    recovered_obs = np.array(
        recovered_cases.loc[
            recovered_cases["Country/Region"] == "Germany",
            date_formatted_begin:date_formatted_end,
        ]
    )[0]

    dead_obs = np.array(
        dead_cases.loc[
            dead_cases["Country/Region"] == "Germany",
            date_formatted_begin:date_formatted_end,
        ]
    )[0]

    data_germany = np.stack([cases_obs, recovered_obs, dead_obs]).T
    data_germany = np.diff(data_germany, axis=0)
    T_germany = data_germany.shape[0]
    N_germany = 83e6
    mean_g = np.mean(data_germany, axis=0)
    std_g = np.std(data_germany, axis=0)
    out = dict(x=data_germany, T=T_germany, N=N_germany, Mean=mean_g, Std=std_g)
    return out
