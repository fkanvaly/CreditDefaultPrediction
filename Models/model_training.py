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
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.metrics import classification_report_imbalanced

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report


from xgboost import XGBClassifier

from tqdm import tqdm

import json
import sys
sys.path.append("../../CreditDefaultPrediction")

from FeaturesEngineering.preprocessing import *


def get_models(clf_list, X,y):
    """Train classifiers on the input data and return the results
    
    Arguments:
        X {NxM dataframe} -- the features data
        y {Nx1 datafram} -- the groundthruth
    """
    classifiers = {
                    "LogisiticRegression": LogisticRegression(max_iter=500, n_jobs=-1),
                    "XGBClassifier": XGBClassifier(n_jobs=-1, n_estimators=1000, max_depth=10, **{'gpu_id': 0, 'tree_method': 'gpu_hist'}),
                    "KNeighborsClassifier" : KNeighborsClassifier(3, n_jobs=-1),
                    "SVC" : SVC(gamma=2, C=1),
                    "GaussianProcessClassifier" : GaussianProcessClassifier(1.0 * RBF(1.0)),
                    "DecisionTreeClassifier" : DecisionTreeClassifier(max_depth=5),
                    "RandomForestClassifier" : RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1),
                    "MLPClassifier" : MLPClassifier(max_iter=1000),
                    "AdaBoostClassifier" : AdaBoostClassifier(),
                    "LGBMClassifier": LGBMClassifier(
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
    clf = {}
    for i, clf_name in enumerate(clf_list):
        # print("%i/%i : Training %s"%(i, len(clf_list), clf_name), end="")
        # classifiers[clf_name].fit(X,y)
        clf[clf_name] = classifiers[clf_name]
        # print("[OK]")
    
    
    return clf

def train_with_cv(classifiers, X, y, n_split=5):
    sss = StratifiedKFold(n_splits=n_split, random_state=None, shuffle=False)
    
    result = pd.DataFrame(columns=["clf_name", "pr", "rc", "f1", "auc", "acc"])
    
    mean = lambda L: np.mean(L)
    X_ = X.values
    y_ = y.values
    for i, (name, clf) in enumerate(classifiers.items()):
        acc, pr, rc, f1, auc = [], [], [], [], []
        for train, val in sss.split(X_, y_):
            undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), clf) # SMOTE happens during Cross Validation not before..
            undersample_model = undersample_pipeline.fit(X_[train], y_[train])
            undersample_prediction = undersample_model.predict(X_[val])
        
            acc.append(undersample_pipeline.score(X_[val], y_[val]))
            pr.append(precision_score(y_[val], undersample_prediction))
            rc.append(recall_score(y_[val], undersample_prediction))
            f1.append(f1_score(y_[val], undersample_prediction))
            auc.append(roc_auc_score(y_[val], undersample_prediction))
        
        result.loc[i] = [name, mean(pr), mean(rc), mean(f1), mean(auc), mean(acc)]
        
    return classifiers, result
        
            
    

def cross_validation(classifiers, X, y, cv=5):
    result = pd.DataFrame(columns=["clf_name", "f1", "f1-std"])
    N = len(classifiers)
    for i, (name, clf) in enumerate(classifiers.items()):
        print("Cross-validation %s....."%name,end="")
        scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
        result.loc[i] = [name, scores.mean(), scores.std() * 2]
        print("[OK] f1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return result

if __name__ == "__main__":
    df = prepocess_data("../data/raw/CreditTraining.csv")
    df = encode_data(df)
    y = df.Y
    X = df.drop("Y", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
    
    _, res = train_models(X_train, y_train)
    print(res)
