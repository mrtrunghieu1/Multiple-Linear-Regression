from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

def Posterior(training,group,sample,flag):
    mdl = PosteriorModel(training,group,flag)
    P = PosteriorTest(mdl,training,sample,flag)
    return P

def PosteriorTest(mdl,training,sample,flag):
    P_predict = mdl.predict_proba(sample)
    return P_predict

def PosteriorModel(training,group,flag):
    if flag == 0:
        mdl = LinearDiscriminantAnalysis()
        mdl.fit(training, group)
    elif flag == 1:
        mdl = GaussianNB()
        mdl.fit(training,group)
    elif flag == 2:
        neigh = KNeighborsClassifier(n_neighbors=5)
        mdl = neigh.fit(training,group)
    elif flag == 3:
        lr = LogisticRegression(solver = 'newton-cg')
        mdl = lr.fit(training,group)
    elif flag == 4:
        rf = RandomForestClassifier(n_estimators=10)
        mdl = rf.fit(training,group)
    elif flag == 5:
        neigh = KNeighborsClassifier(n_neighbors=10)
        mdl = neigh.fit(training,group)
    elif flag == 6:
        neigh = KNeighborsClassifier(n_neighbors=25)
        mdl = neigh.fit(training,group)
    elif flag == 7:
        neigh = KNeighborsClassifier(n_neighbors=50)
        mdl = neigh.fit(training,group)  
    return mdl

