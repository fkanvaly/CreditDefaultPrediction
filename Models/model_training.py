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
from imblearn.under_sampling import NearMiss, RandomUnderSampler
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
from FeaturesEngineering.encoding import *

def get_models(clf_list, X,y,
               sampling_method = RandomUnderSampler(sampling_strategy='majority'), 
               ):
    classifiers = {
                    "LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1),
                    "XGBClassifier": XGBClassifier(n_jobs=-1, n_estimators=1000, max_depth=10),#, **{'gpu_id': 0, 'tree_method': 'gpu_hist'}),
                    "KNeighborsClassifier" : KNeighborsClassifier(3, n_jobs=-1),
                    "SVC" : SVC(gamma=2, C=1),
                    "GaussianProcessClassifier" : GaussianProcessClassifier(1.0 * RBF(1.0)),
                    "DecisionTreeClassifier" : DecisionTreeClassifier(max_depth=5),
                    "RandomForestClassifier" : RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1),
                    "MLPClassifier" : MLPClassifier(max_iter=1000),
                    "AdaBoostClassifier" : AdaBoostClassifier(),
                    "LGBMClassifier": LGBMClassifier(
                                                    boosting_type= 'gbdt',
                                                    max_depth = 10,
                                                    objective= 'binary',
                                                    nthread= 5,
                                                    num_leaves= 32,
                                                    learning_rate= 0.05,
                                                    max_bin= 512,
                                                    subsample_for_bin= 200,
                                                    subsample= 0.7,
                                                    subsample_freq= 1,
                                                    colsample_bytree= 0.8,
                                                    reg_alpha= 20,
                                                    reg_lambda= 20,
                                                    min_split_gain= 0.5,
                                                    min_child_weight= 1,
                                                    min_child_samples= 10,
                                                    scale_pos_weight= 1,
                                                    num_class = 1,
                                                    metric = 'auc')
                 }
    clf = {}
    for i, clf_name in enumerate(clf_list):
        # print("%i/%i : Training %s"%(i, len(clf_list), clf_name), end="")
        # classifiers[clf_name].fit(X,y)
        clf[clf_name] = imbalanced_make_pipeline(sampling_method, classifiers[clf_name])
        # print("[OK]")
    
    
    return clf

def train_models(classifiers, X, y,
                 ):

    result = pd.DataFrame(columns=["clf_name", "pr", "rc", "f1", "auc", "acc"])
    
    for i, clf_name in enumerate(classifiers):
        print("%i/%i : Training %s .... "%(i, len(classifiers), clf_name), end="")
        classifiers[clf_name].fit(X,y)
        
        y_pred = classifiers[clf_name].predict(X)
        acc = accuracy_score(y, y_pred)
        pr = precision_score(y, y_pred)
        rc = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        
        result.loc[i] = [clf_name, pr, rc, f1, auc, acc]
        print("[OK]")
        
    return classifiers, result

def train_with_cv(classifiers, X, y, 
                  n_split=5):
    """train model using cross validation
    
    Arguments:
        classifiers  -- dict of classifier
        X  -- train data
        y  -- groundtruth
    
    Keyword Arguments:
        n_split  -- cross validation split (default: {5})
    
    Returns:
        trained classifier and result
    """
    sss = StratifiedKFold(n_splits=n_split, random_state=None, shuffle=False)
    result = pd.DataFrame(columns=["clf_name", "pr", "rc", "f1", "auc", "acc"])
    
    mean = lambda L: np.mean(L)
    X_ = X.values
    y_ = y.values
    for i, (name, clf) in enumerate(classifiers.items()):
        acc, pr, rc, f1, auc = [], [], [], [], []
        for train, val in sss.split(X_, y_):
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

    #encode data
    cat = [col for col in df if df[col].dtype.name == 'category'] + ['is_closed_date']
    dict_encod = {"WOEEncoder": cat}
    encoder = encoding(dict_encod)
    encoder.fit(df,df.Y)
    df = encoder.transform(df)

    X = df.drop("Y", axis=1)
    y = df.Y
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

    
    clf_list = ["LogisiticRegression",
            "XGBClassifier",
            "RandomForestClassifier", 
            "MLPClassifier",
            "AdaBoostClassifier",
            "LGBMClassifier"
           ]

    classifiers = get_models(clf_list, X_train, y_train)
    import ipdb; ipdb.set_trace()