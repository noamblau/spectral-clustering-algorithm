import sys
import numpy as np
import myspkmeans


##################################################################
# error_message:
################
# Inputs:       None
# Output:       None
# Description:  Print error message and terminate the program.
##################################################################
def error_message():
    print("An Error Has Occured")
    sys.exit()

##################################################################
# first_centroid:
################
# Inputs:       Vectors array, number of vectors, k
# Output:       Centroids_array
# Description:  Create the centroids list and pick randomly the
#               first centroid to keep in the centroids array.
##################################################################
def first_centroid(v_arr, num_of_vectors, k_val):
    centroids_array = [0] * k_val
    rnd = np.random.choice(num_of_vectors, 1)
    print(int(rnd[0]), end="")
    centroids_array[0] = v_arr[rnd[0]]
    return centroids_array


##################################################################
# init_d_array:
##############
# Inputs:       Vectors array, first centroid, number of vectors
# Output:       D_array
# Description:  Build array that keeps the initial D_i values for
#               every vector.
#               The initial D_i is the Euclidean distance of the
#               vector from the first centroid squared.
##################################################################
def init_d_array(v_arr, centroid, num_of_vectors):
    d_array = np.empty(num_of_vectors)
    counter = 0
    for v in v_arr:
        distance = np.linalg.norm(np.array(v) - centroid)
        d_array[counter] = distance * distance
        counter += 1
    return d_array


##################################################################
# centroids_calculation:
#######################
# Inputs:       Vectors array, centroids array, d_array, number of vectors, k
# Output:       None
# Description:  Build the centroids array.
#               For every centroid, build array that keeps the
#               probability to choose every key from the keys array.
#               The probability to choose vector with index i is
#               D_i/sum_d when sum_d is the sum of all the D_i's.
#               After choose randomly the centroid recalculate
#               the D_i's array.
##################################################################
def centroids_calculation(v_arr, centroids_array, d_array, num_of_vectors, k_val):
    if (k_val == 1): print("")
    for z in range(1, k_val):
        sum_d = np.sum(d_array)
        # calculate the probabilities array
        p_array = np.empty(num_of_vectors)
        for v in range(num_of_vectors):
            p_array[v] = d_array[v] / sum_d
        rnd = np.random.choice(num_of_vectors, 1, p=p_array)
        centroids_array[z] = v_arr[rnd[0]]
        if z < k_val - 1:
            # recalculate the D_i's array consider the new centroid
            print(",", end=""), print(int(rnd[0]), end="")
            counter = 0
            for v in v_arr:
                distance = np.linalg.norm(np.array(v) - np.array(centroids_array[z]))
                new_val = distance * distance
                if new_val < d_array[counter]:
                    d_array[counter] = new_val
                counter += 1
        else:
            print(",", end=""), print(int(rnd[0]))

##################################################################
# print_vector:
##############
# Inputs:       Vector, whether to print \n in the end
# Output:       None
# Description:  Print vector.
##################################################################
def print_vector(vector, is_new_line):
    last = len(vector) - 1
    precision = 4
    for v in range(last):
        entry = vector[v]
        if entry < 0 and entry > -0.00005: entry = 0.0;
        print(format(entry, '.4f'), end=",")
    entry = vector[last]
    if entry < 0 and entry > -0.00005: entry = 0.0;
    if is_new_line:
        print(format(entry, '.4f'))
    else:
        print(format(entry, '.4f'), end="")


if __name__ == '__main__':
    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_name = sys.argv[3]
    max_iter = 300
    try:
        f = open(file_name, "r")
    except FileNotFoundError:
        error_message()
    try:
        vectors = np.genfromtxt(f, delimiter=",")
    except ValueError:
        error_message()
    f.close()
    vectors_list = vectors.tolist()
    if type(vectors_list[0]) is not list:
        for i in range(len(vectors_list)):
            vectors_list[i] = [vectors_list[i]]
    num_of_v = len(vectors_list)
    dim = len(vectors_list[0])

    # send the data for the new points calculation
    vectors_list = myspkmeans.calculate_vectors(vectors_list, k, dim, num_of_v, goal, 0)

    # if goal is spk do kmeans++ and find initial centroids
    if (goal == "spk"):
        # the 1.1 algorithm
        np.random.seed(0)
        k = len(vectors_list[0])
        centroids = first_centroid(vectors_list, num_of_v, k)
        D = init_d_array(vectors_list, np.array(centroids[0]), num_of_v)
        centroids_calculation(vectors_list, centroids, D, num_of_v, k)

        # send the data for the final centroids calculation
        centroids_list = myspkmeans.calculate_centroids(centroids, vectors_list, k, k, num_of_v)

        # print the final centroids
        for i in range(k - 1):
            print_vector(np.array(centroids_list[i]), is_new_line=True)
        print_vector(centroids_list[k - 1], is_new_line=False)
