import pandas as pd

train_data = pd.read_csv('data/train.csv', parse_dates = ['Time UTC'],
                         date_parser = dateparse)
print(train_data.head())

codes_data = pd.read_csv('data/codes.csv', sep = ';')
print(codes_data.head())
