from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier

from tqdm import tqdm
from threading import Thread

import json
import sys
sys.path.append("../../CreditDefaultPrediction")

from FeaturesEngineering.preprocessing import *
from utils.sampling import *

tuning_grid = {
                "LogisticRegression":{'C': [0.001, 0.01, 0.1, 1, 10, 100, 500,1000]},
                "XGBClassifier": {
                            "max_depth"        : [ 3, 4, 5, 6, 8, 12, 15],
                            "gamma"            : [ 0.1, 0.2 , 0.3, 0.4 ],
                            },
                
                "KNeighborsClassifier" : {"n_neighbors":range(1,30,2), 
                                          "leaf_size": range(10,60,10),
                                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                
                "SVC" : {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
                
                "GaussianProcessClassifier" : {'kernel':[1.0 * RBF(1), 1.0 * RBF(0.5)]},
                
                "DecisionTreeClassifier" : {"criterion": ["gini", "entropy"],
                                            'max_depth': range(2,16,2),
                                            'min_samples_split': range(2,16,2)},
                
                "RandomForestClassifier" : {
                                    "n_estimators" : [100, 500, 1200],
                                    "max_depth" : [5, 8, 15, 25],
                                    "min_samples_split" : [10, 15, 50],
                                },
                
                "MLPClassifier" : {
                        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                        'alpha': [0.0001, 0.05],
                        },
                
                "AdaBoostClassifier" : {
                                "n_estimators" : [100, 300, 500, 800, 1200],
                                'learning_rate': [0.01, 0.10, 0.30, 0.8, 1],
                             },
                "LGBMClassifier": {
                                'learning_rate': [0.02,0.1,0.5],
                                'n_estimators': [100, 500, 1200,100],
                                'num_leaves': [6,8,12,16,34],
                                }
                
                }

def tune_model(classifiers, X, y, scoring='f1'):
    tuned_params = {}
    tuned_estimator = {}
    for name, clf in classifiers.items():
        print("Grid search for %s"%name)
        param_grid = {name.lower()+"__"+k:v for k,v in tuning_grid[name].items()}
        search = GridSearchCV(clf, 
                              param_grid, 
                              cv=5, n_jobs=-1, 
                              scoring=scoring, verbose=1)
        search.fit(X,y)
        tuned_params[name] = search.best_params_
        tuned_estimator[name] = search.best_estimator_
    
    return tuned_params, tuned_estimator
        

if __name__ == "__main__":
    df = prepocess_data("../data/raw/CreditTraining.csv")
    df = encode_data(df)
    df = random_under_sampling(df)
    y = df.Y
    X = df.drop("Y", axis=1)

    classifiers = {
                    "LogisiticRegression": LogisticRegression(penalty='l2',max_iter=4000, n_jobs=-1),
                    # "XGBoost": XGBClassifier(n_jobs=-1),
                    # "KNeighbors" : KNeighborsClassifier(3, n_jobs=-1),
                    # "SVC" : SVC(gamma=2, C=1),
                    # "GaussianProcess" : GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1),
                    # "DecisionTree" : DecisionTreeClassifier(max_depth=5),
                    # "RandomForest" : RandomForestClassifier(max_depth=5, n_estimators=500, max_leaf_nodes=16, n_jobs=-1),
                    # "MLP" : MLPClassifier(alpha=1, max_iter=1000),
                    # "AdaBoost" : AdaBoostClassifier(SGDClassifier(loss='log')),
                    }
    classifiers = {
                    # "LogisiticRegression": LogisticRegression(penalty='l2',max_iter=4000, n_jobs=-1),
                    # "XGBoost": XGBClassifier(n_jobs=-1),
                    # "KNeighbors" : KNeighborsClassifier(3, n_jobs=-1),
                    # # "SVC_linear" : SVC(kernel="linear", C=0.025),
                    # # "SVC" : SVC(gamma=2, C=1),
                    # # "GaussianProcess" : GaussianProcessClassifier(1.0 * RBF(1.0)),
                    # "DecisionTree" : DecisionTreeClassifier(max_depth=5),
                    # "RandomForest" : RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1),
                    # "MLP" : MLPClassifier(max_iter=1000),
                    # "AdaBoost" : AdaBoostClassifier(),
                    "LGBM": LGBMClassifier(
                                            n_estimators=10000,
                                            num_leaves=34,
                                            colsample_bytree=0.9497036,
                                            subsample=0.8715623,
                                            max_depth=8,
                                            objective=['binary'],
                                            reg_alpha=0.041545473,
                                            reg_lambda=0.0735294,
                                            min_split_gain=0.0222415,
                                            min_child_weight=39.3259775,
                                            silent=-1,
                                            verbose=-1, )
                 }
    param = tune_model(classifiers, X, y)
    import ipdb; ipdb.set_trace()