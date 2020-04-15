import numpy as np
import os
from sklearn.model_selection import KFold
from util import write_file
from scipy.optimize import lsq_linear
import Posterior
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

class MLR(object):
    def __init__(self, args):
        self.X_train_full = args['X_train_full']
        self.X_test = args['X_test']
        self.nfolds = args['nfolds']
        self.classifiers = args['classifiers']
        self.TLoop = args['TLoop']
    
    def predict_probability(self):
        self.n0 = self.X_train_full.shape[0]  
        
        self.features_X_train_full = self.X_train_full[:, :-1] # Features train
        self.labels_y_train_full = self.X_train_full[:,-1].astype(np.int32)   # Label train
        # print(self.labels_y_train_full)
        self.classes = np.unique(self.labels_y_train_full)
        
        self.n_classes = len(self.classes)
        self.flag = [i for i, clsf in enumerate(self.classifiers) if clsf == 1]
        self.n_classifiers = len(self.flag)
        
        self.P = np.zeros((self.n0, self.n_classes * self.n_classifiers))
        
        kf = KFold(n_splits= self.nfolds, shuffle=True)
        self.kf_split = list(kf.split(self.X_train_full))
        
        for train_ids, test_ids in self.kf_split:
            sample = self.features_X_train_full[test_ids, :]
            # print(sample.shape)
            training = self.features_X_train_full[train_ids, :]
            # print(training.shape)
            # print(self.labels_y_train_full[train_ids].shape)
            group = self.labels_y_train_full[train_ids]
            
            PTempt = Posterior.Posterior(training,group,sample,self.flag[0])
            for i in range(1, self.n_classifiers):
                Pr = Posterior.Posterior(training,group,sample,self.flag[i])
                PTempt = np.concatenate((PTempt, Pr), axis = 1)
            # print(PTempt.shape)
            self.P[test_ids,:] = PTempt

    def weight_caculation(self):
        Y = np.zeros((self.n0, self.n_classes))
        for i0 in range(self.n0):
            for m in range(self.n_classes):
                if self.labels_y_train_full[i0] == (m+1):
                    Y[i0][m] = 1
        W = np.zeros((self.n_classifiers, self.n_classes))
        for m in range(self.n_classes):
            X = self.P[:, m :(m + (self.n_classifiers - 1)*self.n_classes) + 1: self.n_classes]
            y = Y[:,m]
            W[:,m] = lsq_linear(X,y).x
            # print(lsq_linear(X,y))
        train_model = []
        for i in range(self.n_classifiers):
            model = Posterior.PosteriorModel(self.features_X_train_full, self.labels_y_train_full,self.flag[i])
            train_model.append(model)

        n_test = self.X_test.shape[0]
        meas_test = self.X_test[:, :-1]
        species_test = self.X_test[:,-1]
        P_test = np.zeros((n_test, self.n_classes * self.n_classifiers))

        P_temp = Posterior.PosteriorTest(train_model[0], self.features_X_train_full)
        for i in range(1,self.n_classifiers):
            Pr_test = Posterior.PosteriorTest(train_model[i], self.features_X_train_full)
            P_temp = np.concatenate((P_temp, Pr_test), axis = 1)
        P_test = P_temp
        # print('P_test[0]:',P_test[0])
        # print('W:', W)
        LR = np.zeros((n_test, self.n_classes))
        for itest in range(n_test):
            for m in range(self.n_classes):
                tempt = 0;
                for k in range(self.n_classifiers):
                    tempt = tempt + W[k][m]*P_test[itest][m + k*self.n_classes]
                
                LR[itest][m] = tempt
        # print('LR[0]:',LR[0])
        ytestMLR = np.zeros((n_test))
        for itest in range(n_test):
            id = np.argmax(LR[itest])
            ytestMLR[itest] = self.classes[id]

        # print(classification_report(species_test,ytestMLR,self.classes))
        p_macro, r_macro, f1_macro, _  = precision_recall_fscore_support(species_test, ytestMLR, average='macro')
        p_weighted, r_weighted, f1_weighted, _  = precision_recall_fscore_support(species_test, ytestMLR, average='weighted')
        accuracy = accuracy_score(species_test, ytestMLR)
        return p_macro, r_macro, f1_macro, p_weighted, r_weighted, f1_weighted, accuracy

        






            
            
        
        
        