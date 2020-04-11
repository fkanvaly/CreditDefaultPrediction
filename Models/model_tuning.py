from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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

df = prepocess_data("../data/raw/CreditTraining.csv")
df = encode_data(df)
df = random_under_sampling(df)
y = df.Y
X = df.drop("Y", axis=1)

classifiers = {
                # "XGBoost": XGBClassifier(n_jobs=-1),
                "KNeighbors" : KNeighborsClassifier(3, n_jobs=-1),
                # "SVC" : SVC(gamma=2, C=1),
                # "GaussianProcess" : GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1),
                # "DecisionTree" : DecisionTreeClassifier(max_depth=5),
                # "RandomForest" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                # "MLP" : MLPClassifier(alpha=1, max_iter=1000),
                # "AdaBoost" : AdaBoostClassifier(),
                # "GaussianNB" : GaussianNB(),
                # "QGA" : QuadraticDiscriminantAnalysis()
                }

tuning_params = {
                # "XGBoost": {
                #             # "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
                #             "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
                #             "min_child_weight" : [ 1, 3, 5, 7 ],
                #             "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                #             # "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
                #             },
                
                "KNeighbors" : {"n_neighbors":range(1,30,2), "leaf_size": range(10,60,10)},
                # "SVC" : [
                #         {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                #         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                #         ],
                # "GaussianProcess" : {'kernel':[1.0 * RBF(1), 1.0 * RBF(0.5)]},
                # "DecisionTree" : DecisionTreeClassifier(max_depth=5),
                # "RandomForest" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                # "MLP" : MLPClassifier(alpha=1, max_iter=1000),
                # "AdaBoost" : AdaBoostClassifier(),
                # "GaussianNB" : GaussianNB(),
                # "QGA" : QuadraticDiscriminantAnalysis()
                }

tuned_params = {}

def search(foo, name, args):
    search = GridSearchCV(**args)
    # tuned_params[name] = 

tr = Thread(target=GridSearchCV, kwargs={"estimator": classifiers["KNeighbors"],
                                         "param_grid": tuning_params["KNeighbors"],
                                         "cv":5, "n_jobs":-1})
tr.start()
tr.join()
# search = GridSearchCV(classifiers["GaussianProcess"], tuning_params["GaussianProcess"], cv=5, n_jobs=-1)
# search.fit(X,y)

import ipdb; ipdb.set_trace()