import sys
import datetime
import scipy.io as sio
import numpy as np
import os

from util import write_file_cv
from data_helper import file_list, data_folder, cv_folder
from MLR_Perf import MLR_Perf

try:
    from_id = int(sys.argv[1])
    to_id = int(sys.argv[2])
except:
    from_id = 0
    to_id = len(file_list)

for i_file in range(from_id, to_id):
    file_name = file_list[i_file]
    print(datetime.datetime.now(), ' File {}: '.format(i_file), file_name)
    #------------------------------DATA PREPROCESS-------------------------------------
    D = np.loadtxt("{}\{}.dat".format(data_folder, file_name), delimiter=',')
    cv = sio.loadmat("{}\cv_{}.mat".format(cv_folder, file_name))['cv']
    #----------------------------- Initial parameters ---------------------------------
    n = D.shape[0]
    n_classifiers = 5
    knn = 5
    n_folds = 10
    n_iters = 3
    binary_classifiers = [1, 1, 1, 1, 1, 0]
    #--------------------------- MLR Perf -------------------------------------------

    P_macro, R_macro, F1_macro, P_weighted, R_weighted, F1_weighted, Accuracy = MLR_Perf(D,n_folds,
    binary_classifiers,n_iters,cv)
    
    result_folder = "result/{}".format(file_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # =====================Write file ====================================
    
    write_file_cv(Accuracy, result_folder,'accuracy')

    write_file_cv(P_macro, result_folder,'precision_macro')
    write_file_cv(R_macro, result_folder,'recall_macro')
    write_file_cv(F1_macro, result_folder,'f1_macro')

    write_file_cv(P_weighted, result_folder,'precision_weighted')
    write_file_cv(R_weighted, result_folder,'recall_weighted')
    write_file_cv(F1_weighted, result_folder,'f1_weighted')



