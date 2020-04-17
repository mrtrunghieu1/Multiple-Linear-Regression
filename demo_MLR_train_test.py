# Library
import datetime
import numpy as np
import sys
import os
# File code
from data_helper import raw_data, file_list
from MLR import MLR
from util import write_file

try:
    from_id = int(sys.args[0])
    to_id = int(sys.args[1])
except:
    from_id = 0
    to_id = len(file_list)

for i_file in range(from_id, to_id):
    print(i_file)
    file_name = file_list[i_file]
    print(datetime.datetime.now(),' File {}: '.format(i_file), file_name)

    '''-------------------------Data Loader -----------------------'''
    D_train = np.loadtxt(raw_data + '/train1/' + file_name + '_train1.dat', delimiter=',')
    D_val = np.loadtxt(raw_data + '/val/' + file_name + '_val.dat', delimiter=',')
    D_test = np.loadtxt(raw_data + '/test/' + file_name + '_test.dat', delimiter=',')

    X_train_full = np.concatenate((D_train, D_val), axis=0)
    X_test = D_test

    '''----------------------------- Initial parameters --------------------------------'''
    binary_classifiers = [1, 1, 1, 1, 1, 0] # 1:Activate | 0:Deactivate  Classifiers
    n_folds = 10

    mlr_parameters = {
        'X_train_full': X_train_full,
        'X_test': X_test,
        'nfolds':n_folds,
        'classifiers': binary_classifiers,
        'TLoop': None
    }

    mlr_object = MLR(mlr_parameters)
    mlr_object.predict_probability()
    p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro, accuracy = mlr_object.weight_caculation()

    '''=====================Write file ================================'''
    result_folder = "result_train_test/{}".format(file_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    write_file(accuracy, result_folder,'accuracy')

    write_file(p_macro, result_folder,'precision_macro')
    write_file(r_macro, result_folder,'recall_macro')
    write_file(f1_macro, result_folder,'f1_macro')

    write_file(p_micro, result_folder,'precision_micro')
    write_file(r_micro, result_folder,'recall_micro')
    write_file(f1_micro, result_folder,'f1_micro')
