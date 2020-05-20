import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

## Default frequency
def plot_dist(data, col, fig=None):
    n_customer=len(data)
    n_default=data[col].sum()

    ##Plotting
    if fig==None: plt.figure(figsize=(7,4))
    sns.set_context('notebook', font_scale=1.2)
    sns.countplot(col,data=data, palette="Blues")
    plt.title('CREDIT Default', size=14)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

def plot_roc_curve_multiple(classifiers, X, y, title='ROC Curves', cv=None, fig=None):
    prediction = {}
    for name, clf in classifiers.items():
        if hasattr(clf, "predict_proba"):
            prediction[name] = clf.predict_proba(X)[:,1]
        elif hasattr(clf, "decision_function"):
            prediction[name] = clf.decision_function(X)
        else:
            prediction[name] = clf.predict(X)
    
    if fig==None: plt.figure(figsize=(16,8))
    plt.title(title, fontsize=18)
    
    for name, y_pred in prediction.items():
        fpr, tpr, thresold = roc_curve(y, prediction[name])
        plt.plot(fpr, tpr, label='%s Score: %0.2f'%(name, roc_auc_score(y, prediction[name])))
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()

# Let's Plot LogisticRegression Learning Curve
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

def plot_learning_curve(classifiers, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='roc_auc'):
    
    n_clf = len(classifiers)
    r = int(np.ceil(n_clf/2))
    f, axes = plt.subplots(r, 2, figsize=(20,7*r), sharey=True)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    # First Estimator
    for i, (name, clf) in enumerate(classifiers.items()):
        
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax = axes[i//2,i%2] if n_clf>2 else axes[i]
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="#ff9124")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                 label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                 label="Cross-validation score")
        ax.set_title("%s Learning Curve"%name, fontsize=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")
    
    return plt