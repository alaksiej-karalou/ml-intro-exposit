import datetime
import math
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def unite_columns(df1, df2: pd.DataFrame):
    for col in df1.columns:
        if col not in df2.columns:
            df1.pop(col)
    for col in df2.columns:
        if col not in df1.columns:
            df2.pop(col)


def convert_time_to_seconds(date):
    t = time.strptime(date, '%H:%M:%S')
    return int(datetime.timedelta(hours = t.tm_hour, minutes = t.tm_min, seconds = t.tm_sec).total_seconds())


def prepare(df: pd.DataFrame):
    df[['unknown_0', 'time']] = df['datetime'].str.split(' ', expand = True)
    df.drop('datetime', axis = 1, inplace = True)
    df['time'] = df['time'].apply(convert_time_to_seconds)

    interval_count_sec = 3600
    print(pd.get_dummies(pd.cut(df['time'], np.arange(0, 3600 * 24 / interval_count_sec + 1) * interval_count_sec)))

    df = df.join(pd.get_dummies(df['code'], prefix = 'code'))
    df.drop(['code'], axis = 1, inplace = True)

    df = df.join(pd.get_dummies(df['type'], prefix = 'type'))
    df.drop(['type'], axis = 1, inplace = True)

    y = df.pop('target')
    return df, y


X_train, Y_train = prepare(pd.read_csv('data/train.csv'))
X_train.to_csv('data/train_prepared.csv', index = False)
X_val, Y_val = prepare(pd.read_csv('data/val.csv'))

unite_columns(X_train, X_val)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_res = pd.DataFrame(
    {
        'predicted': model.predict(X_val),
        'result': Y_val,
    }
)
y_res['correct'] = y_res['predicted'] == y_res['result']
print(y_res.groupby('correct').size())
