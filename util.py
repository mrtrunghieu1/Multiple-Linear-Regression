import numpy as np
def write_file(array, folder, filename):
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