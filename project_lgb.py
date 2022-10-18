import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

data= pd.read_csv('C:\\Users\\Admin\\Desktop\\lecture notes\\Modern Methods of Statistical Learning\\project_train.csv')
y = data[['Label']].to_numpy().ravel() # Last column is our response
X = data.iloc[:, :-1].to_numpy() # The rest are the independent variables

def objective(trial, X, y):      # define an objective for Optuna

    param_grid= {                 # parameter grid
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=0)        # 5-fold cv for every trial
    cv_scores = np.empty(5)

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(**param_grid)                 # fit the model with certain parameters
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[ lgb.early_stopping(10)], #early stopping to not ovefit
        )
        trial.set_user_attr(key="best_booster", value=model)   # store the number of the trial for later

        preds = model.predict(X_test)
        pred_labels = np.rint(preds)
        accuracy = accuracy_score(y_test, pred_labels)
        cv_scores[idx]= accuracy
    return np.mean(cv_scores)

def callback(study, trial):                                   # a function to retrieve the best trained model for inference
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


study = optuna.create_study(direction="maximize")               # run the study for optimizing the parameters
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=20, callbacks=[callback])


print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_model=study.user_attrs["best_booster"]
test_data = pd.read_csv("C:\\Users\\Admin\\Desktop\\lecture notes\\Modern Methods of Statistical Learning\\project_test.csv").to_numpy()

pred_label = best_model.predict(test_data)

pd.DataFrame(pred_label).to_csv(path_or_buf="C:\\Users\\Admin\\Desktop\\lecture notes\\Modern Methods of Statistical Learning\\tree_based_pred2.csv", header = False, index = False)
