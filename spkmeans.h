#ifndef SPKMEANS_H
#define SPKMEANS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

typedef double coor;
typedef coor* vector;

typedef struct vd{
    int dim;
    int v_num;
    vector* vectors;
} vectors_dataset;

typedef struct symmetric_matrix{
    vector* vals;
    int rows_num;
} sym_mat;

typedef struct diagonal_matrix{
    vector diag_vals;
    int rows_num;
} diag_mat;

typedef struct regular_matrix{
    vector* vals;
    int rows_num;
    int columns_num;
} reg_mat;

typedef struct rotation_matrix{
    int i;
    int j;
    coor c;
    coor s;
}rot_mat;

typedef struct full_entry{
    int i;
    int j;
    coor val;
} entry_with_index;

typedef struct eigenvalue{
    int index;
    coor eigenvalue;
} eigenvalue;

typedef struct cluster{
    int amount;
    vector sum;
}cluster;

/************* errors handling *************/
void error_message();
/*******************************************/


/************ memory allocation ************/
void* prog_malloc(size_t size);
vectors_dataset* alloc_vectors_dataset(vector all_vals, int dim, int v_num, int p_or_c);
sym_mat* alloc_sym_mat(int rows_num);
diag_mat* alloc_diag_mat(int rows_num);
reg_mat* alloc_reg_mat(int rows_num, int columns_num);
eigenvalue* alloc_eigenvalues(int v_num);
/*******************************************/


/*************** free memory ***************/
void prog_free(void *ptr);

/*structs*/
void free_vectors_dataset(vectors_dataset* vd);
void free_sym_mat(sym_mat* m);
void free_diag_mat(diag_mat* m);
void free_reg_mat(reg_mat* m);

/*spk*/
void vd_w_free(vectors_dataset* vd, sym_mat* w);
void vd_w_d_free(vectors_dataset* vd, sym_mat* w, diag_mat* d);
void jacobi_free(vectors_dataset* vd, sym_mat* lnorm, reg_mat* v, eigenvalue* eigenvalues);
void spk_free(reg_mat* v, eigenvalue* eigenvalues);
void free_eigenvalues_array(eigenvalue* eigenvalues);

/*kmeans*/
void free_vectors(vector* vectors, int vectors_num);
void free_clusters(cluster* clusters, int k);
void free_all(vector* centroids, cluster* clusters, int k);
/*******************************************/


/****************** print ******************/
void print_coor_vector(coor entry, int i, int n, int is_new_line);
void print_coor_marix(coor entry, int i, int j, int n, int m);
void print_vector(vector v, int dim, int is_new_line);
void print_sym_mat(sym_mat* m);
void print_diag_mat(diag_mat* m);
void print_reg_mat(reg_mat* m); /*delete*/
void print_reg_mat_transpose(reg_mat* m);
void print_eigenvalues_array(eigenvalue* eigenvalues, int n);
/*******************************************/


/*************** read input ****************/
FILE* fopen_and_check(const char* filename, const char* mode);
int get_vector_dim_from_file(FILE* input_file, const int MAX_CHARS);
vectors_dataset* vectors_dataset_from_file(FILE* input_file);
sym_mat* sym_mat_from_input(vector* vectors, int n);
/*******************************************/


/*********** vectors operations ************/
double distance_sq(vector v1, vector v2, int dim);
/*******************************************/


/*********** matrices properties ***********/
coor get_entry_of_sym_mat(sym_mat* m, int i, int j);
coor get_entry_of_diag_mat(diag_mat* m, int i, int j);
void set_entry_of_sym_mat(sym_mat* m, int i, int j, coor val);
void set_entry_of_diag_mat(diag_mat* m, int i, coor val);
int get_sym_mat_dim(sym_mat* m);
int get_diag_mat_dim(diag_mat* m);
int get_reg_mat_rows_num(reg_mat* m);
int get_reg_mat_columns_num(reg_mat* m);
/*******************************************/


/*********** matrices operations ***********/
coor sum_row_of_sym_mat(sym_mat* m, int row);
diag_mat* sqrt_inv_diag_mat(diag_mat* m);
sym_mat* mult_diag_sym_same_diag(sym_mat* s, diag_mat* d);
sym_mat* subtract_sym_mat_from_I(sym_mat* s);
/*******************************************/


/************ jacobi operations ************/
entry_with_index* find_max_entry_not_on_diag(sym_mat* m);
rot_mat* create_p(sym_mat* a);
sym_mat* calc_a_tag(sym_mat* a, rot_mat* p);
coor calc_t(coor theta);
coor calc_theta(sym_mat* a, int i, int j);
coor calc_off_sq(sym_mat* m);
reg_mat* turn_rot_to_reg(rot_mat* rt, int rows_num);
reg_mat* mult_reg_rot(reg_mat* rg, rot_mat* p);
reg_mat* build_v_mat(sym_mat* l_norm, eigenvalue* eigenvalues);
/*******************************************/


/******* eigen heuristic operations ********/
int cmpfunc (const void* a, const void* b);
int find_k(eigenvalue* eigenvalues, int n);
/*******************************************/


/************** spk algorithm **************/
coor calc_w_entry(vector* vectors, int dim, int i, int j);
sym_mat* build_w_mat(vector* vectors, int dim, int v_num);
diag_mat* build_d_mat(sym_mat* w);
sym_mat* build_l_norm_mat(sym_mat* w, diag_mat* d);
reg_mat* build_u_mat(reg_mat* v, eigenvalue* eigenvalues, int k, int n);
reg_mat* build_t_mat(reg_mat* u, int k, int n);
reg_mat* spk(vectors_dataset* vd, int k_val, char* goal, int p_or_c);
/*******************************************/


/************ kmeans algorithm *************/
vector* init_centroids(vector* vectors, int dim, int k);
vector* average_vector(vector* vectors, int dim, int n);
cluster* init_clusters(int dim, int k);
void reset_clusters(cluster* clusters, int dim, int k);
void calculate_clusters(vector* vectors, vector* centroids, cluster* clusters, int dim, int k, int vectors_num);
int update_centroids(vector* centroids, cluster* clusters, int dim, int k);
void kmeans_loop(vector* vectors, vector* centroids, cluster* clusters, int dimension, int k_val, int v_num);
void kmeans_c(vector* vectors, int k_val, int v_num, int dimension);
/******************************************/

int main(int argc, char* argv[]);

#endif
