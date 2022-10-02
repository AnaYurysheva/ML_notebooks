import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

print('Preparing the data')
data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.columns = data.columns.str.lower().str.replace(' ', '_')
for column in data.dtypes[data.dtypes == 'object'].index:
    data[column] = data[column].str.lower().replace(' ', '_')

df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=1
)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.churn.values
y_test = df_test.churn.values
y_val = df_val.churn.values

del df_train['churn']
del df_test['churn']
del df_val['churn']

numercal = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies', 'contract',
               'paperlessbilling', 'paymentmethod']


def train(df, y_train, C=1.0):
    dicts = df[categorical+numercal].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical+numercal].to_dict(orient='records')

    X = dv.transform(dicts)
    y = model.predict_proba(X)[:, 1]
    return y


print('Training the model')

scores = []
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=10)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_predict = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_predict)
    scores.append(auc)

print('%s %.3f +- %.3f' % (C, np.mean(scores),  np.std(scores)))


dv, model = train(df_full_train, df_full_train.churn.values, C=1)
y_predict = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_predict)

print('Actual score', auc)
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
