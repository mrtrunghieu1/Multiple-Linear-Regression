import numpy as np
import json

def write_file_cv(array, folder, filename):
    array_mean = np.mean(array)
    array_var = np.var(array)
    np.savetxt(folder + "/" + filename, array, delimiter=',', fmt='%0.6e')
    f = open(folder + "/" + filename, "a")
    f.write("----------\n")
    f.write("Mean:\n")
    f.write("{0:6E}\n".format(array_mean))
    f.write("Variance:\n")
    f.write("{0:6E}".format(array_var))
    f.close()

def write_file(result,folder, filename):
    with open(folder + "/" + filename, 'w') as outfile:
        json.dump(result, outfile)