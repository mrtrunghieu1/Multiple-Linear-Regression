from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def Posterior(training,label,sample,flag):
    mdl = PosteriorModel(training,label,flag)
    P = PosteriorTest(mdl,sample)
    return P

def PosteriorTest(mdl,sample):
    P_predict = mdl.predict_proba(sample)
    return P_predict

def PosteriorModel(training,label,flag):
    if flag == 0:
        mdl = XGBClassifier(n_estimators=10)
        mdl.fit(training, label)
    elif flag == 1:
        mdl = GaussianNB()
        mdl.fit(training,label)
    elif flag == 2:
        neigh = KNeighborsClassifier(n_neighbors=5)
        mdl = neigh.fit(training,label)
    elif flag == 3:
        lr = LogisticRegression(random_state=0)
        mdl = lr.fit(training,label)
    elif flag == 4:
        rf = RandomForestClassifier(n_estimators=10)
        mdl = rf.fit(training,label)
    elif flag == 5:
        neigh = KNeighborsClassifier(n_neighbors=20)
        mdl = neigh.fit(training,label)
    elif flag == 6:
        neigh = KNeighborsClassifier(n_neighbors=25)
        mdl = neigh.fit(training,label)
    elif flag == 7:
        neigh = KNeighborsClassifier(n_neighbors=50)
        mdl = neigh.fit(training,label)  
    return mdl

