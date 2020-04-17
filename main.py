from Models.model_training import * 
from FeaturesEngineering.preprocessing import *
from utils.sampling import *
from FeaturesEngineering.encoding import *

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

df = prepocess_data("data/raw/CreditTraining.csv")


y = df.Y
X = df.drop("Y", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

df_train = X_train.assign(Y=y_train.values)
y_train = df_train.Y
X_train = df_train.drop("Y", axis=1)


#* encoding
cat = [col for col in df if df[col].dtype.name == 'category'] + ['is_closed_date']
dict_encod = {"WOEEncoder": cat}
encoder = encoding(dict_encod)
encoder.fit(X_train,y_train)

#transform
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

#* under sampling
df_train = X_train.assign(Y=y_train.values)
df_train = random_under_sampling(df_train)
y_train = df_train.Y
X_train = df_train.drop("Y", axis=1)

#* train model
classifiers, result = train_models(X_train,y_train)

print(result)
# #* VIZ
# sns.set(style="whitegrid")
# # Draw a nested barplot to show survival for class and sex
# g = sns.barplot(x="clf", y="f1", color="purple", data=result)
# g.set_xticklabels(labels=g.get_xticklabels(),rotation=90)
# plt.savefig('output.png')
# plt.show()

#
# ──────────────────────────────────────────────── I ──────────
#*   :::::: T E S T : :  :   :    :     :        :          :
# ──────────────────────────────────────────────────────────
#
result_test = pd.DataFrame(columns=["clf", "f1", "acc"])
N = len(classifiers)
for i, (name, clf) in enumerate(classifiers.items()):
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test,y_pred)
    result_test.loc[i] = [name, f1, acc]
    
sns.set(style="whitegrid")
# Draw a nested barplot to show survival for class and sex
g = sns.barplot(x="clf", y="f1", color="purple", data=result_test)
g.set_xticklabels(labels=g.get_xticklabels(),rotation=90)
plt.savefig('output.png')
plt.show()
