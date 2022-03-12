import datetime
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score
import numpy as np


def verify_model(y_test, y_pred):
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test, y_pred)}')


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


def prepare(df):
    df[['unknown_0', 'time']] = df['datetime'].str.split(' ', expand = True)
    df.pop('datetime')
    df['unknown_0'] = df['unknown_0'].astype('int64')
    df['amount_sign'] = df['amount'] >= 0
    df['amount'] = df['amount'].apply(lambda x: abs(x))
    interval_count_sec = 900
    df = df.join(pd.get_dummies(
        pd.cut(df['time'].apply(convert_time_to_seconds),
               np.arange(0, 3600 * 24 / interval_count_sec + 1) * interval_count_sec),
        prefix = 'time'))
    df.pop('time')

    df['amount'] = df['amount'].apply(lambda x: abs(x))

    cat_vars = ['type', 'code', 'unknown_0']
    for var in cat_vars:
        cat_list = pd.get_dummies(df[var], prefix = var)
        df = df.join(cat_list)

    y = df.pop('target')

    return df, y


def sec_to_time(sec):
    return f'{int(sec / 3600 % 24):02d}:{int(sec % 3600 / 60):02d}:{sec % 60:02d}'


codes_data = pd.read_csv('data/codes.csv', sep = ';', index_col = 'code')
types_data = pd.read_csv('data/types.csv', sep = ';', index_col = 'type')

X_train, Y_train = prepare(pd.read_csv('data/train.csv'))
X_val, Y_val = prepare(pd.read_csv('data/val.csv'))

print(Y_train)

unite_columns(X_train, X_val)

weights = {
    0: Y_train.sum(),
    1: Y_train.size - Y_train.sum()
}

model = LogisticRegression(solver = 'sag', class_weight = weights)
model.fit(X_train, Y_train)

verify_model(Y_train, model.predict(X_train))
print("===")
verify_model(Y_val, model.predict(X_val))

result = pd.DataFrame(
    {
        'column': X_train.columns,
        'weight': model.coef_.flatten()
    }
)
print(result.sort_values('weight', ascending = False, key = abs))

description = []
for col in result['column']:
    if col.startswith('code_'):
        code = int(col.replace('code_', ''))
        t = codes_data.loc[code]['code_description']
        description.append(t)
    elif col.startswith('type_'):
        type_ind = int(col.replace('type_', ''))
        t = types_data.loc[type_ind]['type_description']
        description.append(t)
    elif col.startswith('time_'):
        time_int = col.replace('time_(', '').replace(']', '').split(', ')
        time = sec_to_time(int(float((time_int[0])))) + '-' + sec_to_time(int(float((time_int[1]))))
        description.append(time)
    else:
        description.append(col)

result['description'] = description
