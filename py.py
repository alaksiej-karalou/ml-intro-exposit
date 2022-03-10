import pandas as pd
from pandas_profiling import ProfileReport

train_data = pd.read_csv('data/train.csv')
train_data.reset_index(drop = True, inplace = True)
print(train_data)

profile = ProfileReport(train_data, title = "Pandas Profiling Report")
profile.to_file("data/train_profile.html")
