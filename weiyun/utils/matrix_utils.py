
from weiyun.utils.log_utils import logger

def print_matrix(matrix):
    for a in matrix:
        ln = ""
        for i in a:
            ln += str(i) + " "
        # print(ln)
        logger.debug(ln)

def matrix2str(matrix):
    s = ""
    for a in matrix:
        ln = ""
        for i in a:
            ln += str(i) + " "
        s += ln + "\n"

def copy_matrix(matrix):
    cp_m = []
    for i in range(len(matrix)):
        cp_m.append([])
        for j in range(len(matrix[i])):
            cp_m[i].append(matrix[i][j])
    return cp_m

def copy_arr(array):
    cp_a = []
    for i in range(len(array)):
        cp_a.append(array[i])
    return cp_a