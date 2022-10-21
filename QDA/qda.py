import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv('data/project_train.csv', header = 0)
df_test = pd.read_csv('data/project_test.csv', header = 0)

df_train_X = df_train.iloc[:,0:11]
df_train_y = df_train.iloc[:,11]

qda = QuadraticDiscriminantAnalysis()
qda.fit(df_train_X, df_train_y)

# Cross_validation
scores = cross_val_score(qda, df_train_X, df_train_y, cv=5)
print('Accuracy: %.02f, Stdev: %.02f' %(scores.mean(), scores.std()))

y_pred = qda.predict(df_test)
print(y_pred)
np.savetxt('prediction_QDA.csv', y_pred, delimiter = ',', fmt = '%d')
