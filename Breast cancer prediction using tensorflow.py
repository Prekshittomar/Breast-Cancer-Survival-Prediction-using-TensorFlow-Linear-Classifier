from __future__ import absolute_import,division,print_function,unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib 

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

print(tf.version)

dftrain=pd.read_csv('Downloads\\Bcan\\breast_cancer_survival.csv')
dftrain.dtypes

dftrain=dftrain.dropna()
dftrain.isnull().sum(axis=0)

def convert_status(status):
    if status == 'Alive':
        return 1
    elif status == 'Dead':
        return 0

dftrain['Patient_Status']=dftrain['Patient_Status'].apply(convert_status)

dfeval=dftrain.sample(n=133,replace=False,random_state=31208)

dftrain=dftrain.drop(columns=['ER status','PR status','HER2 status'])
dfeval=dfeval.drop(columns=['ER status','PR status','HER2 status'])

y_train=dftrain.pop('Patient_Status')
y_eval=dfeval.pop('Patient_Status')

dftrain.Age.hist(bins=20)
dftrain.Gender.value_counts().plot(kind='barh')
dftrain["Tumour_Stage"].value_counts().plot(kind="pie", autopct="%.2f%%")
dftrain['Surgery_type'].value_counts().plot(kind='barh')

dftrain.dtypes
dftrain.columns
categorical=['Gender','Tumour_Stage', 'Histology',
       'Surgery_type', 'Date_of_Surgery', 'Date_of_Last_Visit']
numerical=['Age','Protein1','Protein2','Protein3','Protein4']

dftrain['Age']=dftrain['Age'].astype(float)
dfeval['Age']=dfeval['Age'].astype(float)
dftrain.dtypes

feature_columns=[]

for feature_name in categorical:
    vocabulary=dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in numerical:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

def make_input_fn(data_df,label_df,num_epochs=10,shuffle=True,batch_size=32):
    def input_function():
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn=make_input_fn(dftrain, y_train)
eval_input_fn=make_input_fn(dfeval, y_eval)

linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result=linear_est.evaluate(eval_input_fn)
results=list(linear_est.predict(eval_input_fn))

dfeval.reset_index(drop=True, inplace=True)
y_eval.reset_index(drop=True, inplace=True)


print(dfeval.loc[26])
print(y_eval.loc[26])
print(results[26]['probabilities'][1])


