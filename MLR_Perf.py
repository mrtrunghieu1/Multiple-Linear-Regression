
import numpy as np
from util import write_file
from MLR import MLR

def MLR_Perf(D, n_folds, classifiers, niters, cv):
    species = D[:,-1]
    n0 = len(species)
    all_indexs = np.array(range(D.shape[0]))

    # Initial matrix P, R, F macro or weighted
    P_macro = np.zeros(niters * n_folds,)
    R_macro = np.zeros(niters * n_folds,)
    F1_macro = np.zeros(niters * n_folds,)

    P_weighted = np.zeros(niters * n_folds,)
    R_weighted = np.zeros(niters * n_folds,)
    F1_weighted = np.zeros(niters * n_folds,)

    Accuracy = np.zeros(niters * n_folds,)

    for i_iter in range(niters):
        base_loop = i_iter * n_folds
        for i_fold in range(n_folds):
            current_loop = base_loop + i_fold
            print(current_loop)
            test_indexs = cv[0, current_loop][:, 0] - 1   # Matlab from index 1, Python from index 0
            train_indexs = np.setdiff1d(all_indexs, test_indexs)

            X_train_original = D[train_indexs]
            X_test_original = D[test_indexs]

            # Class MLR ...............
            mlr_args = {
                'X_train_full': X_train_original,
                'X_test': X_test_original,
                'nfolds':n_folds,
                'classifiers':classifiers,
                'TLoop': ((i_iter-1)*n_folds + i_fold)
            }

            mlr_object = MLR(mlr_args)
            mlr_object.predict_probability()
            p_macro, r_macro, f1_macro, p_weighted, r_weighted, f1_weighted, accuracy = mlr_object.weight_caculation()

            P_macro[current_loop] = p_macro
            R_macro[current_loop] = r_macro
            F1_macro[current_loop] = f1_macro

            P_weighted[current_loop] = p_weighted
            R_weighted[current_loop] = r_weighted
            F1_weighted[current_loop] = f1_weighted

            Accuracy[current_loop] = accuracy

    return P_macro, R_macro, F1_macro, P_weighted, R_weighted, F1_weighted, Accuracy
    