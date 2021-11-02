#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"

/*py_to_c_vector
 ***************
Inputs:       PyObject * py_vector, Py_ssize_t vector_size
Output:       c_vector
Description:  Receives a pointer to vector in python form and the size
              of the vector, creates the suitable vector in C and returns it.
*/
vector py_to_c_vector(PyObject * py_vector, Py_ssize_t vector_size){
    Py_ssize_t i;
    PyObject* item;
    vector c_vector = (vector)prog_malloc(sizeof c_vector * vector_size);
    if (c_vector==NULL) error_message();
    for (i = 0; i < vector_size; i++) {
        item = PyList_GetItem(py_vector, i);
        if (!PyFloat_Check(item)) continue;
        c_vector[i] = PyFloat_AsDouble(item);
        if (c_vector[i]  == -1 && PyErr_Occurred()){
            error_message();
            prog_free(c_vector);
            return NULL;
        }
    }
    return c_vector;
}

/*py_to_c_vectors
 ****************
Inputs:       PyObject* py_vectors
Output:       vectors (vectors in C)
Description:  Receives a pointer to vectors in python form,
              creates the suitable vectors in C (using py_to_c_vector function)
              and returns the vectors.
*/
vector* py_to_c_vectors(PyObject* py_vectors)
{
    PyObject* item;
    Py_ssize_t i, n;
    vector* vectors;
    if (!PyList_Check(py_vectors))
        return NULL;
    n = PyList_Size(py_vectors);
    vectors = (vector*)prog_malloc(sizeof(vector)*n);
    if (vectors==NULL) error_message();
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(py_vectors, i);
        if (!PyList_Check(item)){
            continue;
        }
        vectors[i] = py_to_c_vector(item, PyList_Size(item));
    }
    return vectors;
}

/*c_to_py_vector
 ***************
Inputs:       vector c_vector, PyObject* py_vector,  Py_ssize_t vector_size
Output:       None
Description:  Receives a pointer to vector in C form, a pointer to vector in
              Python form and the size of the vector. Updates the vector in Python
              form to have the same coordinates' values as the vector in C form.
*/
void c_to_py_vector(vector c_vector, PyObject* py_vector,  Py_ssize_t vector_size){
    Py_ssize_t i;
    PyObject *item;
    for (i = 0; i < vector_size; i++) {
        item = PyFloat_FromDouble(c_vector[i]);
        if (item == NULL){
            error_message();
            prog_free(c_vector);
        }
        PyList_SetItem(py_vector, i, item);
    }
}

/*c_to_py_vectors
 ****************
 Inputs:       vector* c_vectors, int rows_num, int columns_num
 Output:       PyObject* (PyList of PyLists)
 Description:  Receives a pointer to vector in C form, creates PyList of PyLists
               with the same values of c_vectors (the element i,j of the PyList
               is c_vectors[i][j]) and returns it.
*/
PyObject* c_to_py_vectors(vector* c_vectors, int rows_num, int columns_num)
{
    PyObject* py_vectors, *py_vector;
    Py_ssize_t i, j;
    py_vectors = PyList_New(rows_num);
    for (i = 0; i < rows_num; ++i) {
        py_vector = PyList_New(columns_num);
        for (j = 0; j < columns_num; ++j) {
            PyList_SetItem(py_vector, j, PyFloat_FromDouble(c_vectors[i][j]));
        }
        PyList_SetItem(py_vectors, i, py_vector);
    }
    return py_vectors;
}

/*kmeans_py
 **********
 Inputs:       PyObject* py_centroids, PyObject* py_vectors,
               int k, int dimension, int vectors_num (the name that are not shortened)
 Output:       PyObject* py_centroids
 Description:  Receives pointers to the centroids and vectors in Python form,
               the k (number of clusters/centroids), the vectors' dimension and the
               number of vectors.
               Runs a Kmeans algorithm (given the initial centroids, so it fits
               K-means++ of Python) and returns the final centroids after the K-means
               run (using kmeans_loop function), as Python objects.
*/
static PyObject* kmeans_py(PyObject* py_centroids, PyObject* py_vecs, int k, int dim, int v_num){
    cluster* clusters;
    vector* vectors, *centroids;
    vectors = py_to_c_vectors(py_vecs); /* vectors_pointer is in C form */
    centroids = py_to_c_vectors(py_centroids);
    clusters = init_clusters(dim, k);
    kmeans_loop(vectors, centroids, clusters, dim, k, v_num);
    py_centroids = c_to_py_vectors(centroids, k, dim);
    /* Updates py_centroids to have the current centroids (in Python form)*/
    free_all(centroids,clusters,k); /* free C allocated memory*/
    return py_centroids;
}

/*vd_vecs_arrange
 ****************
 Inputs:       vectors_dataset* vd, vector* vecs(the vectors after
               py_to_c_vectors function), int v_num (the number of vectors
               from the input), int dim (the dimension of the vectors)
 Output:       None
 Description:  Receives initialized vectors_dataset struct,
               and enters the vectors to the struct (so be held in continuous block).
*/
void vd_vecs_arrange(vectors_dataset* vd, vector* vecs, int v_num, int dim){
    int i,j;
    for(i=0; i<v_num;++i){
        for(j=0;j<dim;++j){
                vd->vectors[i][j] = vecs[i][j];
        }
        prog_free(vecs[i]);
    }
    prog_free(vecs);
}

/*spk_func
 *********
Inputs:       PyObject* py_vecs (vectors as python objects), int k (the k given by
              the user as an argument), int dim (the dimension of the vectors),
              int v_num (the number of vectors, char* goal (the goal given by
              the user as argument), int p_or_c (indicator if the function is
              ran in C or in Python).
Output:       py_centroids
Description:  Transforms the vectors Python form to C form, creates vectors_dataset,
              runs the spk function on these vectors, and returns T matrix
              (as called in instructions document) rows as vectors in Python form.
*/
static PyObject* spk_func(PyObject* py_vecs, int k, int dim, int v_num, char* goal, int p_or_c){
    vector* vectors_pointer;
    reg_mat* t;
    PyObject* py_datapoints;
    int new_k;
    vector all_vd_vals = (vector)prog_malloc(dim*v_num*sizeof(coor));
    vectors_dataset* vd = (vectors_dataset*)alloc_vectors_dataset(all_vd_vals, dim, v_num, 0);
    vectors_pointer = py_to_c_vectors(py_vecs); /* vectors_pointer is in C form */
    vd_vecs_arrange(vd, vectors_pointer, v_num, dim);
    vd->dim = dim;
    vd->v_num = v_num;
    t = spk(vd, k, goal, p_or_c);
    new_k = t->columns_num;
    py_datapoints = c_to_py_vectors(t->vals, v_num, new_k);
    free_reg_mat(t);
    return py_datapoints;
}

/*calculate_vectors
 ******************
Inputs:       PyObject *self, PyObject *args
Output:       PyObject* res
Description:  Receives arguments from Python (the vectors, k, vectors' dimension,
              the number of vectors, the goal and indicator that not called by C function)
              and runs the spk_func using these parameters.
              Returns a pointer to the output of PyObject of  the result of spk_func
              (vectors that are suitable for the K-means).
*/
static PyObject* calculate_vectors(PyObject *self, PyObject *args){
    PyObject* vectors;
    PyObject* res;
    int k_val, dimension, v_num, p_or_c;
    char* goal;
    if(!PyArg_ParseTuple(args, "Oiiisi", &vectors, &k_val, &dimension, &v_num, &goal, &p_or_c)){
        return NULL;
    }
    res = spk_func(vectors, k_val, dimension, v_num, goal, p_or_c);
    return Py_BuildValue("O", res);
}

/*calculate_centroids
 ********************
Inputs:       PyObject *self, PyObject *args
Output:       PyObject* res
Description:  Receives arguments from Python (the centroids, vectors, k,
              vectors' dimension and the number of vectors) and runs the kmeans_py
              (so it will be K-means++) using these parameters.
              Returns a pointer to the output of PyObject of the result of Kmeans.
*/
static PyObject* calculate_centroids(PyObject *self, PyObject *args){
    PyObject* centroids;
    PyObject* vectors;
    int k_val, dimension, v_num;
    if(!PyArg_ParseTuple(args, "OOiii", &centroids, &vectors, &k_val, &dimension, &v_num)){
        return NULL;
    }
    centroids = kmeans_py(centroids, vectors, k_val, dimension, v_num);
    return Py_BuildValue("O", centroids);
}

#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }

static PyMethodDef _methods[] = {
        FUNC(METH_VARARGS, calculate_vectors, "Runs spk Algorithm"),
        FUNC(METH_VARARGS, calculate_centroids, "Runs kmeans Algorithm"),
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _moduledef = {
        PyModuleDef_HEAD_INIT,
        "myspkmeans",
        NULL,
        -1,
        _methods
};

PyMODINIT_FUNC
PyInit_myspkmeans(void)
{
    return PyModule_Create(&_moduledef);
}