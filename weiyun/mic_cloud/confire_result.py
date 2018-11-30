import pickle
import numpy

def confirm_result():
    with open('resources_num_test_ours', 'rb') as fp:
        results_ours = pickle.load(fp)
    with open('resources_num_test_best', 'rb') as fp:
        results_best = pickle.load(fp)
    numpy.savetxt('resources_num_test_ours.csv', results_ours, delimiter=',')
    numpy.savetxt('resources_num_test_best.csv', results_best, delimiter=',')
    # results = []
    # for i in range(len(results_ours)):
    #     ros = results_ours[i]
    #     rbs = results_best[i]
    #     results.append([sum(ros)/len(ros), sum(rbs)/len(rbs)])
    # print(results)
confirm_result()

# my_matrix = numpy.loadtxt(open("resources_num_test_ours.csv","rb"), delimiter=",", skiprows=0)
# print(my_matrix)