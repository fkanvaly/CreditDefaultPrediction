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
from sklearn.model_selection import cross_val_score


from xgboost import XGBClassifier

from tqdm import tqdm

import json
import sys
sys.path.append("../../CreditDefaultPrediction")

from FeaturesEngineering.preprocessing import *


def train_models(X,y):
    """Train classifiers on the input data and return the results
    
    Arguments:
        X {NxM dataframe} -- the features data
        y {Nx1 datafram} -- the groundthruth
    """
    classifiers = {
                    "XGBoost": XGBClassifier(),
                    "KNeighbors" : KNeighborsClassifier(3),
                    # # "SVC_linear" : SVC(kernel="linear", C=0.025),
                    # # "SVC" : SVC(gamma=2, C=1),
                    # # "GaussianProcess" : GaussianProcessClassifier(1.0 * RBF(1.0)),
                    # "DecisionTree" : DecisionTreeClassifier(max_depth=5),
                    "RandomForest" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                    "MLP" : MLPClassifier(alpha=1, max_iter=1000),
                    "AdaBoost" : AdaBoostClassifier(),
                    # "GaussianNB" : GaussianNB(),
                    # "QGA" : QuadraticDiscriminantAnalysis()
                 }
    
    
    result = pd.DataFrame(columns=["clf", "f1", "f1-std"])
    N = len(classifiers)
    for i, (name, clf) in enumerate(classifiers.items()):
        print("%i/%i : Training %s....."%(i,N,name),end="")
        clf.fit(X,y)
        scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
        result.loc[i] = [name, scores.mean(), scores.std() * 2]
        print("[OK] f1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    return classifiers, result



if __name__ == "__main__":
    df = prepocess_data("../data/raw/CreditTraining.csv")
    df = encode_data(df)
    y = df.Y
    X = df.drop("Y", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
    
    _, res = train_models(X_train, y_train)
    print(res)
