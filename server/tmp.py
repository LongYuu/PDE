import pandas as pd
import numpy as np

df_africa = pd.read_csv('./data/covid19_confirmed_africa.csv')
country = df_africa['Country'].values

data = []
for each in country:
    index = df_africa[df_africa.Country == each].index.tolist()[0]
    series = df_africa.iloc[index].values.tolist()
    series.pop(0)
    x = series.pop(0)
    y = series.pop(0)
    country_data = {'name' : each, 'value' : series[0]}
    data.append(country_data)
    print(data)