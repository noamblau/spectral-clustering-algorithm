#include "spkmeans.h"

/*******************************************/
/************* errors handling *************/
/*******************************************/

/*error_message
 **************
Inputs:       None
Output:       None
Description:  Print error message and terminate the program.
*/
void error_message(){
    printf("An Error Has Occured");
    exit(0);
}

/*******************************************/
/************ memory allocation ************/
/*******************************************/

/*prog_malloc
 ************
Inputs:       Size of the memory
Output:       Memory allocation
Description:  Allocate a memory with a given size.
*/
void* prog_malloc(size_t size){
    void* a = malloc(size);
    if (a==NULL) error_message();
    return a;
}

/*alloc_vectors_dataset
 **********************
inputs:       All vectors' coordinates (as one vector), vectors' dimension,
              number of vectors in the dataset
output:       Vectors dataset
description:  Allocate a memory for a vectors dataset and initalizes its fields (the
              vectors pointers, the vectors dimension and the number of vectors)
*/
vectors_dataset* alloc_vectors_dataset(vector all_vals, int dim, int v_num, int p_or_c){
    int i;
    vector vd_vals = all_vals;
    vectors_dataset* vd = (vectors_dataset*)prog_malloc(sizeof(vectors_dataset));
    vector* vectors = (vector*)prog_malloc(v_num*sizeof(vector));
    if(p_or_c==1)
        vd_vals = (vector)realloc(all_vals,dim*v_num*sizeof(coor));
    vd->v_num = v_num;
    vd->dim = dim;
    for(i=0; i<v_num; ++i){
        vectors[i] = vd_vals + i*dim;
    }
    vd->vectors = vectors;
    return vd;
}

/*alloc_sym_mat
 **************
Inputs:       Number of rows
Output:       Symmetric matrix
Description:  Allocate a memory for a symmetric matrix.
*/
sym_mat* alloc_sym_mat(int rows_num){
    int i;
    sym_mat* m = (sym_mat*)prog_malloc(sizeof(sym_mat));
    vector* vals = (vector*)prog_malloc(rows_num*sizeof(vector));
    vector all_sym_mat_vals = (vector)prog_malloc((rows_num*(rows_num+1)/2)* sizeof(coor));
    m->rows_num = rows_num;
    m->vals = vals;
    for(i = 0; i < rows_num; ++i){
        m->vals[i] = all_sym_mat_vals + i*(i+1)/2;
    }
    return m;
}

/*alloc_diag_mat
 ***************
Inputs:       Number of rows
Output:       Diagonal matrix
Description:  Allocate a memory for a diagonal matrix.
*/
diag_mat* alloc_diag_mat(int rows_num){
    diag_mat* m = (diag_mat*)prog_malloc(sizeof(diag_mat));
    vector all_diag_mat_vals = (vector)prog_malloc(rows_num*sizeof(coor));
    m->rows_num = rows_num;
    m->diag_vals = all_diag_mat_vals;
    return m;
}

/*alloc_reg_mat
 **************
Inputs:       Number of rows, Number of columns
Output:       Regular matrix
Description:  Allocate a memory for a regular matrix.
*/
reg_mat* alloc_reg_mat(int rows_num, int columns_num){
    int i;
    reg_mat* m = (reg_mat*)prog_malloc(sizeof(reg_mat));
    vector* vals = (vector*)prog_malloc(rows_num*sizeof(vector));
    vector all_reg_mat_vals = (vector)prog_malloc(rows_num*columns_num*sizeof(coor));
    m->rows_num = rows_num;
    m->columns_num = columns_num;
    m->vals = vals;
    for(i = 0; i < rows_num; ++i){
        m->vals[i] = all_reg_mat_vals + i*columns_num;
    }
    return m;
}

/*alloc_eigenvalues
 ******************
Inputs:       Number of eigenvalues
Output:       Eigenvalues array
Description:  Allocate a memory for a regular matrix.
*/
eigenvalue* alloc_eigenvalues(int v_num){
    eigenvalue* eigenvalues = (eigenvalue*)prog_malloc(sizeof(eigenvalue)*v_num);
    return eigenvalues;
}

/*******************************************/
/*************** free memory ***************/
/*******************************************/

/*prog_free
 **********
Inputs:       Pointer
Output:       None
Description:  Free the pointer memory.
*/
void prog_free(void *ptr){
    free(ptr);
}

/***************** structs *****************/

/*free_vectors_dataset
 *********************
Inputs:       Vectors dataset
Output:       None
Description:  Free the vectors dataset memory.
*/
void free_vectors_dataset(vectors_dataset* vd){
    prog_free((vd->vectors)[0]);
    prog_free(vd->vectors);
    prog_free(vd);
}

/*free_sym_mat
 *************
Inputs:       Symmetric matrix
Output:       None
Description:  Free the symmetric matrix memory.
*/
void free_sym_mat(sym_mat* m){
    prog_free((m->vals)[0]);
    prog_free(m->vals);
    prog_free(m);
}

/*free_diag_mat
 **************
Inputs:       Diagonal matrix
Output:       None
Description:  Free the diagonal matrix memory.
*/
void free_diag_mat(diag_mat* m){
    prog_free(m->diag_vals);
    prog_free(m);
}

/*free_reg_mat
 *************
Inputs:       Regular matrix
Output:       None
Description:  Free the regular matrix memory.
*/
void free_reg_mat(reg_mat* m){
    prog_free((m->vals)[0]);
    prog_free(m->vals);
    prog_free(m);
}

/******************* spk *******************/

/*vd_w_free
 **********
Inputs:       Vectors dataset, symmetric matrix
Output:       None
Description:  Free all the given inputs memory.
*/
void vd_w_free(vectors_dataset* vd, sym_mat* w){
    free_vectors_dataset(vd);
    free_sym_mat(w);
}

/*vd_w_d_free
 ************
Inputs:       Vectors dataset, symmetric matrix, diagonal matrix
Output:       None
Description:  Free all the given inputs memory.
*/
void vd_w_d_free(vectors_dataset* vd, sym_mat* w, diag_mat* d){
    vd_w_free(vd, w);
    free_diag_mat(d);
}

/*jacobi_free
 ************
Inputs:       Vectors dataset, symmetric matrix - lnorm, regular matrix,
              eigenvalues array
Output:       None
Description:  Free all the given inputs memory.
*/
void jacobi_free(vectors_dataset* vd, sym_mat* lnorm, reg_mat* v, eigenvalue* eigenvalues){
    free_vectors_dataset(vd);
    free_sym_mat(lnorm);
    free_reg_mat(v);
    free_eigenvalues_array(eigenvalues);
}

/*spk_free
 *********
Inputs:       Regular matrix, eigenvalues array
Output:       None
Description:  Free all the given inputs memory.
*/
void spk_free(reg_mat* v, eigenvalue* eigenvalues){
    free_reg_mat(v);
    free_eigenvalues_array(eigenvalues);
}

/*free_eigenvalues_array
 ***********************
Inputs:       Eigenvalues array
Output:       None
Description:  Free the eigenvalues array memory.
*/
void free_eigenvalues_array(eigenvalue* eigenvalues){
    prog_free(eigenvalues);
}

/***************** kmeans ******************/

/*free_vectors
 *************
Inputs:       Vectors array, number of vectors
Output:       None
Description:  Receives a pointer to vectors and the number of vectors
              and frees the allocated memory for those vectors.
*/
void free_vectors(vector* vectors, int vectors_num){
    int i;
    for(i=0; i<vectors_num; i++){
        prog_free(vectors[i]);
    }
    prog_free(vectors);
}

/*free_clusters
 **************
Inputs:       Clusters array, number of clusters - k
Output:       None
Description:  Receives a pointer to clusters and k (the number of clusters)
              and frees the allocated memory for those clusters.
*/
void free_clusters(cluster* clusters, int k){
    int i;
    for(i=0; i<k; i++) {
        prog_free(clusters[i].sum);
    }
    prog_free(clusters);
}

/*free_all
 *********
Inputs:       Centroids array, clusters array, number of clusters - k
Output:       None
Description:  Receives centroids, clusters and the number of clusters.
              Frees the allocated memory for those centroids and clusters.
*/
void free_all(vector* centroids, cluster* clusters, int k){
    free_vectors(centroids, k);
    if (k != 1){
        free_clusters(clusters, k);
    }
}

/*******************************************/
/****************** print ******************/
/*******************************************/

/*print_coor_vector
 ******************
Inputs:       Value, location - i, length of the vector - n
Output:       None
Description:  Print the given value.
*/
void print_coor_vector(coor entry, int i, int n, int is_new_line){
    if (entry < 0 && entry > -0.00005) entry = 0.0;
    if(i != n-1)
        printf("%.4f,",entry);
    else{
        if(is_new_line)
            printf("%.4f\n",entry);
        else
            printf("%.4f",entry);
    }
}

/*print_coor_matrix
 *****************
Inputs:       Value, location - (i,j), size of the matrix - nxm
Output:       None
Description:  Print the given value.
*/
void print_coor_matrix(coor entry, int i, int j, int n, int m){
    if (entry < 0 && entry > -0.00005) entry = 0.0;
    if(j != m-1)
        printf("%.4f,",entry);
    else{
        if(i != n-1)
            printf("%.4f\n",entry);
        else
            printf("%.4f",entry);
    }
}

/*print_vector
 *************
Inputs:       Vector, dimension of the vector, whether this is a new line
Output:       None
Description:  Print the given vector.
*/
void print_vector(vector v, int dim, int is_new_line){
    int i;
    coor entry;
    for(i=0; i<dim; ++i){
        entry = v[i];
        if (entry < 0 && entry > -0.00005) entry = 0.0;
        print_coor_vector(entry, i, dim, is_new_line);
    }
}

/*print_sym_mat
 **************
Inputs:       Symmetric matrix
Output:       None
Description:  Print the given matrix.
*/
void print_sym_mat(sym_mat* m){
    int i, j;
    int n = get_sym_mat_dim(m);
    for (i=0; i<n; ++i){
        for (j=0; j<n; ++j){
            print_coor_matrix(get_entry_of_sym_mat(m,i,j), i, j, n, n);
        }
    }
}

/*print_diag_mat
 ***************
Inputs:       Diagonal matrix
Output:       None
Description:  Print the given matrix.
*/
void print_diag_mat(diag_mat* m){
    int i, j;
    int n = m->rows_num;
    for (i=0; i<n; ++i){
        for (j=0; j<n; ++j){
            print_coor_matrix(get_entry_of_diag_mat(m,i,j), i, j, n, n);
        }
    }
}

/*print_reg_mat
 **************
Inputs:       Regular matrix
Output:       None
Description:  Print the given matrix.
*/
void print_reg_mat(reg_mat* m){
    int i, j;
    int rows = get_reg_mat_rows_num(m);
    int columns = get_reg_mat_columns_num(m);
    for (i=0; i<rows; ++i){
        for (j=0; j<columns; ++j){
            print_coor_matrix((m->vals)[i][j], i, j, rows, columns);
        }
    }
}

/*print_reg_mat_transpose
 ************************
Inputs:       Regular matrix
Output:       None
Description:  Print the given matrix as transpose matrix.
*/
void print_reg_mat_transpose(reg_mat* m){
    int i, j;
    int n = get_reg_mat_rows_num(m);
    for (i=0; i<n; ++i){
        for (j=0; j<n; ++j){
            print_coor_matrix((m->vals)[j][i], i, j, n, n);
        }
    }
}

/*print_eigenvalues_array
 ************************
Inputs:       Eigenvalues array, number of eigenvalues
Output:       None
Description:  Print the given eigenvalues array.
*/
void print_eigenvalues_array(eigenvalue* eigenvalues, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        print_coor_vector(eigenvalues[i].eigenvalue, i, n, 1);
    }
}

/*******************************************/
/*************** read input ****************/
/*******************************************/

/*fopen_and_check
 ****************
Inputs:       File Name, mode for fopen
Output:       File
Description:  Open the file.
*/
FILE* fopen_and_check(const char *filename, const char *mode){
    FILE* fp = fopen(filename, mode);
    if(fp==NULL) error_message();
    return fp;
}

/*get_vector_dim_from_file
 *************************
Inputs:       File pointer and the maximal number
              of characters in input file's line
Output:       the dimension of the vectors in the file
Description:  Returns the dimension of the vectors in the file
              based on scanning the first line of the file.
*/
int get_vector_dim_from_file(FILE* input_file, const int MAX_CHARS){
    int len=0, dim = 0, i;
    char* s=(char*)prog_malloc(MAX_CHARS * sizeof(char));
    if(fscanf(input_file,"%s", s)!=EOF){
        len = strlen(s);
        for(i=0; i<len; i++){
            if(s[i]==',')
                dim++;
        }
    }
    dim++;
    prog_free(s);
    rewind(input_file);
    return dim;
}

/*vectors_dataset_from_file
 **************************
Inputs:       File pointer
Output:       Pointer to struct vectors_dataset
Description:  Creates vectors_dataset struct from file.
              initializes the structs dim (for dimension),
              v_num (for number of vectors) and vectors.
*/
vectors_dataset* vectors_dataset_from_file(FILE* input_file){
    /*In the longest line the characters are as followed:
     for each coordinate there are 10 characters (8 digits according
     to the forum, a point character and possibly a minus)
     in each line there are up to 50 numbers (If the goal is jacobi),
     and there are 50 separator characters between the numbers
     and after the last number.
     For spare distance, we'll choose to allocate spare 100 characters
     (the string is freed after each iteration).
     The total MAX_CHARS_IN_INPUT_FILE_LINE therefore will be:
     50 * 10 + 50 + 100 = 650
     For spare distance, we chose the limit to 650.
     */
    const int MAX_CHARS_IN_INPUT_FILE_LINE = 650;
    const int MAX_VEC = 50;
    const int MAX_COOR = 50;
    /*because of jacobi tests inputs (Symmetric matrices)*/
    int dim, v_num, coor_count=0;
    vectors_dataset* vd;
    char* coor_s, *s;
    vector all_vectors_coors = (vector)prog_malloc(MAX_VEC*MAX_COOR*sizeof(coor));
    dim = get_vector_dim_from_file(input_file, MAX_CHARS_IN_INPUT_FILE_LINE);
    s=(char*)prog_malloc(MAX_CHARS_IN_INPUT_FILE_LINE * sizeof(char));
    while(fscanf(input_file,"%s", s)!=EOF){
        coor_s = strtok(s, ",");
        while(coor_s != NULL){
            all_vectors_coors[coor_count] = strtod(coor_s, NULL);
            coor_count++;
            coor_s = strtok(NULL, ",");
            /* The use of strtok function is based on an example in the URL:
            https://www.tutorialspoint.com/c_standard_library/c_function_strtok.htm*/
        }
    }
    prog_free(s);
    v_num = coor_count/dim;
    vd = alloc_vectors_dataset(all_vectors_coors, dim, v_num, 1);
    return vd;
}

/*sym_mat_from_input
 *******************
Inputs:       Vectors array, Number of rows
Output:       Symmetric matrix
Description:  Create a symmetric matrix from input.
*/
sym_mat* sym_mat_from_input(vector* vectors, int n){
    int i, j;
    sym_mat* mat = alloc_sym_mat(n);
    for (i=0; i<n; ++i) {
        for (j = 0; j <= i; ++j) {
            set_entry_of_sym_mat(mat, i, j, vectors[i][j]);
        }
    }
    return mat;
}

/*******************************************/
/*********** vectors operations ************/
/*******************************************/

/*distance_sq
 ************
Inputs:       Vector1, vector2, vectors' dimension
Output:       Distance
Description:  Receives 2 vectors and calculate the distance between the vectors.
*/
double distance_sq(vector v1, vector v2, int dim){
    double distance = 0;
    int i;
    for (i = 0; i < dim; i++) {
        double diff = (v1[i]-v2[i])*(v1[i]-v2[i]);
        distance += diff;
    }
    return distance;
}

/*******************************************/
/*********** matrices properties ***********/
/*******************************************/

/*get_entry_of_sym_mat
 *********************
Inputs:       Symmetric matrix, location in matrix - (i,j)
Output:       Value of entry
Description:  Returns the value of the (i,j) entry in the matrix.
*/
coor get_entry_of_sym_mat(sym_mat* m, int i, int j){
    if(j<=i)
        return (m->vals)[i][j];
    else
        return (m->vals)[j][i];
}

/*get_entry_of_diag_mat
 **********************
Inputs:       Diagonal matrix, location in matrix - (i,j)
Output:       Value of entry
Description:  Returns the value of the (i,j) entry in the matrix.
*/
coor get_entry_of_diag_mat(diag_mat* m, int i, int j){
    if(i==j)
        return (m->diag_vals)[i];
    else
        return 0;
}

/*set_entry_of_sym_mat
 *********************
Inputs:       Symmetric matrix, location in matrix - (i,j), Value of entry
Output:       None
Description:  Changes the value of the (i,j) entry in the matrix.
*/
void set_entry_of_sym_mat(sym_mat* m, int i, int j, coor val){
    if(j<=i){
        (m->vals)[i][j] = val;
    } else{
        (m->vals)[j][i] = val;
    }
}

/*set_entry_of_diag_mat
 **********************
Inputs:       Diagonal matrix, location in matrix - (i,j), Value of entry
Output:       None
Description:  Changes the value of the (i,j) entry in the matrix.
*/
void set_entry_of_diag_mat(diag_mat* m, int i, coor val){
    (m->diag_vals)[i] = val;
}

/*get_sym_mat_dim
 ****************
Inputs:       Symmetric matrix
Output:       Dimension of matrix
Description:  Returns the dimension of the matrix.
*/
int get_sym_mat_dim(sym_mat* m){
    return m->rows_num;
}

/*get_diag_mat_dim
 *****************
Inputs:       Diagonal matrix
Output:       Dimension of matrix
Description:  Returns the dimension of the matrix.
*/
int get_diag_mat_dim(diag_mat* m){
    return m->rows_num;
}

/*get_reg_mat_rows_num
 *********************
Inputs:       Regular matrix
Output:       Number of matrix rows
Description:  Returns the number of the matrix rows.
*/
int get_reg_mat_rows_num(reg_mat* m){
    return m->rows_num;
}

/*get_reg_mat_columns_num
 ************************
Inputs:       Regular matrix
Output:       Number of matrix columns
Description:  Returns the number of the matrix columns.
*/
int get_reg_mat_columns_num(reg_mat* m){
    return m->columns_num;
}

/*******************************************/
/*********** matrices operations ***********/
/*******************************************/

/*sum_row_of_sym_mat
 *******************
Inputs:       Symmetric matrix, row number
Output:       Sum of the given row
Description:  Returns the sum of all the entries in the row.
*/
coor sum_row_of_sym_mat(sym_mat* m, int row){
    int j;
    coor sum=0;
    for(j=0; j<get_sym_mat_dim(m); ++j){
        sum += get_entry_of_sym_mat(m, row, j);
    }
    return sum;
}

/*sqrt_inv_diag_mat
 ******************
Inputs:       Diagonal matrix
Output:       Diagonal matrix
Description:  Turn all values on the diagonal(x) to 1/x^2.
*/
diag_mat* sqrt_inv_diag_mat(diag_mat* d){
    int i;
    coor prev_entry;
    for(i=0; i<get_diag_mat_dim(d); ++i){
        prev_entry = get_entry_of_diag_mat(d, i, i);
        set_entry_of_diag_mat(d, i, 1/(sqrt(prev_entry)));
    }
    return d;
}

/*mult_diag_sym_same_diag
 ************************
Inputs:       Symmetric matrix and diagonal matrix
Output:       Symmetric matrix
Description:  Do matrices multiplication D*S*D
              where D is diagonal and S is symmetric
              (under the assumption it can be done).
              This function is used for calculating
              D^(-0.5)*W*D^(-0.5) in the project.
*/
sym_mat* mult_diag_sym_same_diag(sym_mat* s, diag_mat* d){
    int i, j;
    coor d_ii, s_ij, d_jj;
    int mat_dim = get_diag_mat_dim(d);
    for(i=0;i<mat_dim;++i){
        for(j=0;j<=i;++j){
            d_ii = get_entry_of_diag_mat(d, i, i);
            s_ij = get_entry_of_sym_mat(s, i, j);
            d_jj = get_entry_of_diag_mat(d, j, j);
            s_ij = d_ii * s_ij * d_jj;
            set_entry_of_sym_mat(s, i, j, s_ij);
        }
    }
    return s;
}

/*subtract_sym_mat_from_I
 ************************
Inputs:       Symmetric matrix
Output:       Symmetric matrix
Description:  For a given symmetric matrix S,
              returns the symmetric matrix I-S
              (I is the unit matrix in the same dimensions).
*/
sym_mat* subtract_sym_mat_from_I(sym_mat* m) {
    int i, j;
    coor prev_entry;
    int mat_dim = get_sym_mat_dim(m);
    for (i = 0; i < mat_dim; ++i) {
        for (j = 0; j < i; ++j) {
            prev_entry = get_entry_of_sym_mat(m, i, j);
            set_entry_of_sym_mat(m, i, j, -prev_entry);
        }
    }
    for (i = 0; i < mat_dim; ++i) {
        prev_entry = get_entry_of_sym_mat(m, i, i);
        set_entry_of_sym_mat(m, i, i, 1 - prev_entry);
    }
    return m;
}

/*******************************************/
/************ jacobi operations ************/
/*******************************************/

/*find_max_entry_not_on_diag
 ***************************
Inputs:       Symmetric matrix
Output:       Entry with index
              (struct for entry of matrix with the indices)
Description:  Finds the maximal entry (in absolute value)
              of symmetric matrix which is not on on diagonal
              and returns it with its indices (row an column).
*/
entry_with_index* find_max_entry_not_on_diag(sym_mat* m){
    entry_with_index* e = (entry_with_index*)prog_malloc(sizeof(entry_with_index));
    int i, j, max_i=0, max_j=1;
    coor val, max = 0.0;
    int mat_dim = get_sym_mat_dim(m);
    for(i=0;i<mat_dim-1;++i){
        for(j=i+1;j<mat_dim;++j){
            val = fabs(get_entry_of_sym_mat(m, i, j));
            if(val > max){
                max = val;
                max_i = i;
                max_j = j;
            }
        }
    }
    e->i = max_i;
    e->j = max_j;
    e->val = max;
    return e;
}
/*create_p
 *********
Inputs:       Symmetric matrix A
Output:       Rotation matrix P (struct that represent rotation matrix
              from the instructions)
Description:  Creates the rotation matrix P from a give A matrix
              (as explained in Jacobi algorithm in the instructions).
*/
rot_mat* create_p(sym_mat* a){
    int i, j;
    coor theta, t, s, c;
    rot_mat* p = (rot_mat*)prog_malloc(sizeof(rot_mat));
    entry_with_index* entry = find_max_entry_not_on_diag(a);
    i = entry->i;
    j = entry->j;
    if(entry->val==0){
        c = 1;
        s = 0;
    }
    else{
        theta = calc_theta(a, i, j);
        t = calc_t(theta);
        c = 1/(sqrt(pow(t,2)+1));
        s = t * c;
    }
    prog_free(entry);
    p->i = i;
    p->j = j;
    p->c = c;
    p->s = s;
    return p;
}

/*calc_a_tag
 ***********
Inputs:       Symmetric matrix A and rotation matrix P
Output:       Symmetric matrix A'
Description:  Calculates A' as P^t*A*P (based on the relation between A and A'
              explained in the instructions).
*/
sym_mat* calc_a_tag(sym_mat* a, rot_mat* p){
    int r, i, j;
    coor s, c, a_ri, a_rj, a_ii, a_jj, a_ij;
    coor a_tag_ii, a_tag_jj;
    i = p->i;
    j = p->j;
    s = p->s;
    c = p->c;
    a_ii = get_entry_of_sym_mat(a, i, i);
    a_jj = get_entry_of_sym_mat(a, j, j);
    a_ij = get_entry_of_sym_mat(a, i, j);
    for(r=0; r<get_sym_mat_dim(a); ++r){
        if(r != i && r != j){
            a_ri = get_entry_of_sym_mat(a, r, i);
            a_rj = get_entry_of_sym_mat(a, r, j);
            set_entry_of_sym_mat(a, r, i, c*a_ri-s*a_rj);
            set_entry_of_sym_mat(a, r, j, c*a_rj+s*a_ri);
        }
    }
    a_tag_ii = pow(c,2)*a_ii+pow(s,2)*a_jj-2*s*c*a_ij;
    a_tag_jj = pow(s,2)*a_ii+pow(c,2)*a_jj+2*s*c*a_ij;
    set_entry_of_sym_mat(a, i, i, a_tag_ii);
    set_entry_of_sym_mat(a, j, j, a_tag_jj);
    set_entry_of_sym_mat(a, i, j, 0.0);
    return a;
}

/*calc_t
 *******
Inputs:       Coordinate (double) theta
Output:       Coordinate (double) t
Description:  Calculates t given theta
              (by given formula from the document).
*/
coor calc_t(coor theta){
    coor t;
    if(theta>=0)
        t = 1/(theta + sqrt(pow(theta,2)+1));
    else
        t = (-1)/(-theta + sqrt(pow(theta,2)+1));
    return t;
}

/*calc_theta
 ***********
Inputs:       Symmetric matrix A and 2 indices i and j
Output:       Coordinate (double) theta
Description:  Calculates theta (by given formula from the document).
*/
coor calc_theta(sym_mat* a, int i, int j){
    coor a_ii = get_entry_of_sym_mat(a, i, i);
    coor a_jj = get_entry_of_sym_mat(a, j, j);
    coor a_ij = get_entry_of_sym_mat(a, i, j);
    coor theta = (a_jj - a_ii)/(2*a_ij);
    return theta;
}

/*calc_off_sq
 ************
Inputs:       Symmetric matrix M
Output:       Coordinate (double) off^2(M)
Description:  Calculates the off square parameter of the input matrix
              (by given formula from the document).
*/
coor calc_off_sq(sym_mat* m){
    int i, j;
    coor off_sq=0;
    for(i=1; i<get_sym_mat_dim(m); ++i){
        for(j=0; j<i;++j){
            off_sq+=pow(get_entry_of_sym_mat(m, i, j), 2);
        }
    }
    off_sq = off_sq * 2;
    return off_sq;
}

/*turn_rot_to_reg
*****************
Inputs:       Rotation matrix and its number of rows
Output:       Regular matrix
Description:  Build the given rotation matrix as regular matrix struct.
*/
reg_mat* turn_rot_to_reg(rot_mat* rt, int rows_num){
    int i, j;
    int rt_i = rt->i;
    int rt_j = rt->j;
    reg_mat* rg = (reg_mat*) alloc_reg_mat(rows_num,rows_num);
    for (i = 0; i < rows_num; ++i) {
        for (j = 0; j < rows_num; ++j) {
            if (i==j) (rg->vals)[i][j] = 1.0;
            else (rg->vals)[i][j] = 0.0;
        }
    }
    (rg->vals)[rt_i][rt_i] = rt->c;
    (rg->vals)[rt_j][rt_j] = rt->c;
    (rg->vals)[rt_i][rt_j] = rt->s;
    (rg->vals)[rt_j][rt_i] = (-1)*(rt->s);
    return rg;
}

/*mult_reg_rot
**************
Inputs:       Regular matrix, Rotation matrix
Output:       Regular matrix
Description:  Multiply the regular matrix by the rotation matrix from the left.
*/
reg_mat* mult_reg_rot(reg_mat* rg, rot_mat* p){
    int r, i, j;
    coor s, c, a_ri, a_rj;
    i = p->i;
    j = p->j;
    s = p->s;
    c = p->c;
    for(r=0; r<get_reg_mat_rows_num(rg); ++r){
        a_ri = (rg->vals)[r][i];
        a_rj =  (rg->vals)[r][j];
        (rg->vals)[r][i] = (coor)(c*a_ri-s*a_rj);
        (rg->vals)[r][j] = (coor)(c*a_rj+s*a_ri);
    }
    return rg;
}

/*build_v_mat
 ************
Inputs:       Symmetric matrix L_norm (represents L_norm from the instructions),
              empty eigenvalues array
Output:       Regular matrix V (represents v from the instructions) and full eigenvalues array
              after calculation
Description:  Calculates V by creating p1, p2, ... matrices and multiply them from the left.
              Number of iterations limited by 1000 or until convergence.
*/
reg_mat* build_v_mat(sym_mat* l_norm, eigenvalue* eigenvalues){
    const double EPSILON = pow(10, -15);
    coor off_sq_prev, off_sq_curr;
    sym_mat* a;
    rot_mat* p;
    reg_mat* v;
    int i, n = l_norm->rows_num;
    a = l_norm;
    off_sq_prev = calc_off_sq(a);
    p = create_p(a);
    v = turn_rot_to_reg(p, get_sym_mat_dim(a));
    a = calc_a_tag(a,p);
    prog_free(p);
    off_sq_curr = calc_off_sq(a);
    for (i = 1; i < 100 && off_sq_prev-off_sq_curr > EPSILON; ++i) {
        p = create_p(a);
        v = mult_reg_rot(v, p);
        a = calc_a_tag(a, p);
        prog_free(p);
        off_sq_prev = off_sq_curr;
        off_sq_curr = calc_off_sq(a);
    }
    for (i = 0; i < n; ++i){
        eigenvalues[i].index = i;
        eigenvalues[i].eigenvalue = get_entry_of_sym_mat(a, i, i);
    }
    return v;
}

/*******************************************/
/******* eigen heuristic operations ********/
/*******************************************/

/*cmpfunc
*********
Inputs:       const void* a, const void* b
Output:       The result of the comparison
Description:  Compare between eigenvalue a and eigenvalue b for sorting.
*/
int cmpfunc (const void* a, const void* b) {
    eigenvalue *x,*y;
    x = (eigenvalue*)a;
    y = (eigenvalue*)b;
    if (x->eigenvalue == y->eigenvalue){
        if (x->index>y->index) return 1;
        return -1;
    }
    if (x->eigenvalue>y->eigenvalue) return 1;
    return -1;
}

/*find_k
********
Inputs:       Eigenvalues array, number of eigenvalues - n
Output:       K
Description:  Calculate k according to the given eigenvalues (based on the method
              explained in the instructions).
*/
int find_k(eigenvalue* eigenvalues, int n){
    int i, max_i=0;
    coor d, max_d = 0;
    int iterations = floor(n/2);
    for (i = 1; i <= iterations; ++i) {
        d = fabs(eigenvalues[i].eigenvalue-eigenvalues[i-1].eigenvalue);
        if(d > max_d) {
            max_d = d;
            max_i = i;
        }
    }
    return max_i;
}

/*******************************************/
/************** spk algorithm **************/
/*******************************************/

/*calc_w_entry
 *************
Inputs:       Vectors(double**), the dimension of the vectors,
              and 2 indices
Output:       Coordinate i,j in matrix W
Description:  Calculates W_(i,j) as stated in the
              formula from the instructions.
*/
coor calc_w_entry(vector* vectors, int dim, int i, int j){
    vector x_i = vectors[i];
    vector x_j = vectors[j];
    coor w_ij = exp((-0.5)* sqrt(distance_sq(x_i,x_j, dim)));
    return w_ij;
}

/*build_w_mat
 ************
Inputs:       Vectors(double**), the dimension of the vectors
              and the number of vectors
Output:       Symmetric matrix W (from the instructions)
Description:  Allocates memory for W matrix, calculate all its entries
              and returns W.
*/
sym_mat* build_w_mat(vector* vectors,int dim, int v_num){
    int i, j;
    sym_mat* w = alloc_sym_mat(v_num);
    for (i=0; i<v_num; ++i){
        for (j=0; j<i; ++j){
            (w->vals)[i][j] = calc_w_entry(vectors, dim, i, j);
        }
        (w->vals)[i][i] = 0;
    }
    return w;
}

/*build_d_mat
 ************
Inputs:       Symmetric matrix W (represents W from the instructions)
Output:       Diagonal matrix D (represents D from the instructions)
Description:  Allocates memory for D matrix, calculate all its entries
              (by summing rows of W) and returns D.
*/
diag_mat* build_d_mat(sym_mat* w){
    int i;
    int rows_num = w->rows_num;
    diag_mat* d = alloc_diag_mat(rows_num);
    for(i=0; i<rows_num; ++i){
        set_entry_of_diag_mat(d, i, sum_row_of_sym_mat(w, i));
    }
    return d;
}

/*build_lnorm_mat
 ****************
Inputs:       Symmetric matrix W (represents W from the instructions)
              and Diagonal matrix D (represents D from the instructions)
Output:       Symmetric matrix L_norm (represents L_norm from the instructions)
Description:  Calculates L_norm by the formula L_norm = I-D^(-0.5)*W*D^(-0.5)
              and returns it.
*/
sym_mat* build_l_norm_mat(sym_mat* w, diag_mat* d){
    diag_mat* d_sqrt_inv = sqrt_inv_diag_mat(d);
    sym_mat* l_norm = mult_diag_sym_same_diag(w, d_sqrt_inv);
    l_norm = subtract_sym_mat_from_I(l_norm);
    return l_norm;
}

/*build_u_mat
 ************
Inputs:       Regular matrix V (represents V from the instructions), eigenvalues array,
              k that we found and number of eigenvalues - n.
Output:       Regular matrix U (represents U from the instructions)
Description:  Create U from V (first k columns).
*/
reg_mat* build_u_mat(reg_mat* v, eigenvalue* eigenvalues, int k, int n){
    int i, j;
    reg_mat* u = alloc_reg_mat(n,k);
    for (i = 0; i < k; ++i) {
        for (j = 0; j < n; ++j) {
            (u->vals)[j][i] = (v->vals)[j][eigenvalues[i].index];
        }
    }
    return u;
}

/*build_t_mat
 ************
Inputs:       Regular matrix U (represents U from the instructions), eigenvalues array,
              k that we found and number of eigenvalues - n
Output:       Regular matrix T (represents T from the instructions)
Description:  Create T from U (t_ij = u_ij/(∑_j(u_ij^2))^1/2).
*/
reg_mat* build_t_mat(reg_mat* u, int k, int n){
    int i, j;
    coor sum = 0.0;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            sum += pow((u->vals)[i][j], 2);
        }
        for (j = 0; j < k; ++j) {
            (u->vals)[i][j] = (u->vals)[i][j]/sqrt(sum);
        }
        sum = 0.0;
    }
    return u;
}

/*spk
 ****
Inputs:       Vectors Dataset, k, goal, p_or_c - we call the function from python or c
Output:       Regular matrix T (represents T from the instructions)
Description:  Execute the spk algorithm according to instructions:
              1. Form the weighted adjacency matrix W.
              2. Compute L_norm
              3. Determine k
              4. Form U after jacobi algorithm
              5. Form T from U
              6. Send T rows as points to K-means algorithm
              Returned value is according to goal.
*/
reg_mat* spk(vectors_dataset* vd, int k_val, char* goal, int p_or_c){
    sym_mat* w, *l_norm;
    reg_mat* v, *u, *t;
    diag_mat* d;
    eigenvalue* eigenvalues;
    vector* vectors = vd->vectors;
    int dimension = vd->dim;
    int v_num = vd->v_num;
    if (strcmp(goal,"jacobi")!=0){
        w = build_w_mat(vectors, dimension, v_num);
        if (strcmp(goal,"wam")==0){
            print_sym_mat(w);
            vd_w_free(vd, w);
            exit(0);
        }
        d = build_d_mat(w);
        if (strcmp(goal,"ddg")==0){
            print_diag_mat(d);
            vd_w_d_free(vd, w, d);
            exit(0);
        }
        l_norm = build_l_norm_mat(w, d);
        if (strcmp(goal,"lnorm")==0){
            print_sym_mat(l_norm);
            vd_w_d_free(vd, w, d);
            exit(0);
        }
        eigenvalues = alloc_eigenvalues(v_num);
        v = build_v_mat(l_norm, eigenvalues);
        vd_w_d_free(vd, w, d);
    } else{
        l_norm = sym_mat_from_input(vectors, v_num);
        eigenvalues = alloc_eigenvalues(v_num);
        v = build_v_mat(l_norm, eigenvalues);
        print_eigenvalues_array(eigenvalues,v_num);
        print_reg_mat_transpose(v);
        jacobi_free(vd, l_norm, v, eigenvalues);
        exit(0);
    }
    qsort(eigenvalues, v_num, sizeof(eigenvalue), cmpfunc);
    if (k_val==0) k_val = find_k(eigenvalues, v_num);
    u = build_u_mat(v, eigenvalues, k_val, v_num);
    t = build_t_mat(u, k_val, v_num);
    if (p_or_c==1) kmeans_c(t->vals,k_val,v_num,k_val);
    spk_free(v, eigenvalues);
    return t;
}

/*******************************************/
/************ kmeans algorithm *************/
/*******************************************/

/*init_centroids
 ***************
Inputs:       Vectors(double**), their dimension and the K value for K-means
              algorithm
Output:       Vectors which are the initial centroids for the K-means algorithm
Description:  Allocates memory for the centroids and copies the first K vectors
              as the initial centroids(used for K-means and not for Kmeans++).
*/
vector* init_centroids(vector* vectors, int dim, int k){
    int i, j;
    vector* centroids_pointer = (vector*)prog_malloc(k * sizeof(vector));
    for (i = 0; i < k; i++) {
        centroids_pointer[i] = (vector)prog_malloc(dim * sizeof(double));
        for (j = 0; j < dim; j++) {
            centroids_pointer[i][j] = vectors[i][j];
        }
    }
    return centroids_pointer;
}

/*average_vector
 ***************
Inputs:       Vectors(double**), their dimension and the number of vectors
Output:       pointer to single-vector which is the average of all
              the given vectors
Description:  Calculates and returns a pointer to vector which is the
              average of all the given vectors. this is the final centroid
              in the case that K=1 in the K-means algorithm.
*/
vector* average_vector(vector* vectors, int dim, int n){
    int i, j, k;
    vector* centroids_pointer = init_centroids(vectors, dim, 1);
    for (i = 1; i < n; ++i) {
        for (j = 0; j < dim; ++j) {
            centroids_pointer[0][j] += vectors[i][j];
        }
    }
    for (k = 0; k < dim; ++k) {
        centroids_pointer[0][k] = centroids_pointer[0][k]/n;
    }
    return centroids_pointer;
}

/*init_clusters
 **************
Inputs:       The dimension of the input vectors to K-means and the K value
Output:       Initial clusters (pointer to cluster struct)
Description:  Allocates memory for clusters and enters a zero vector to all
              clusters sums (the sum field is used to be the sum of all vectors
              which belong to the cluster)
*/
cluster* init_clusters(int dim, int k){
    cluster* clusters_pointer;
    int i, j;
    clusters_pointer = (cluster*)prog_malloc(k * sizeof(cluster));
    for (i = 0; i < k; i++) {
        cluster c;
        c.amount = 0;
        c.sum = (vector)prog_malloc(dim * sizeof(double));
        for (j = 0; j < dim; j++) {
            c.sum[j] = 0;
        }
        clusters_pointer[i] = c;
    }
    return clusters_pointer;
}

/*reset_clusters
 ***************
Inputs:       Clusters pointer, the dimension of the input vectors to K-meams
              and the K value
Output:       None
Description:  Zeroing the amount and sum of all clusters in the given pointer.
*/
void reset_clusters(cluster* clusters, int dim, int k){
    int i, j;
    for (i = 0; i < k; i++) {
        clusters[i].amount = 0;
        for (j = 0; j < dim; j++) {
            clusters[i].sum[j] = 0;
        }
    }
}

/*calculate_clusters
 *******************
Inputs:       Vectors pointer, centroids pointer, clusters pointer,
              dimension of vectors, k, number of vectors
Output:       None
Description:  Receives pointers to the vectors and centroids,
              the dimension of the vectors, k (the number of centroids) and the
              number of vectors. Updates the clusters based on the given centroids
              as required in the Kmeans algorithm loop.
*/
void calculate_clusters(vector* vectors, vector* centroids, cluster* clusters, int dim, int k, int vectors_num){
    int l, i, j;
    for (l = 0; l < vectors_num; l++) {
        double min_distance = distance_sq(vectors[l],centroids[0], dim);
        int closest_cluster = 0;
        for(i=1; i<k; i++) {
            double distance = distance_sq(vectors[l], centroids[i], dim);
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = i;
            }
        }
        clusters[closest_cluster].amount += 1;
        for (j=0; j<dim; j++){
            clusters[closest_cluster].sum[j] += vectors[l][j];
        }
    }
}

/*update_centroids
 *****************
Inputs:       Centroids pointer, clusters pointer, dimension of vectors, k
Output:       Changed
Description:  Receives pointers to the centroids and clusters,
              the dimension of the vectors, and k (the number of centroids).
              Updates the centroids based on the given clusters as required in the Kmeans algorithm loop.
              Returns indicator if that not all the centroids stayed the same (for the stop condition
              of the algorithm).
*/
int update_centroids(vector* centroids, cluster* clusters, int dim, int k){
    int i, j, changed = 0;
    double new_cor;
    struct cluster curr_cluster;
    for (i = 0; i < k; i++) {
        for (j = 0; j < dim; j++) {
            curr_cluster = clusters[i];
            new_cor = curr_cluster.sum[j] / curr_cluster.amount;
            if (centroids[i][j] != new_cor) {
                changed = 1;
                centroids[i][j] = new_cor;
            }
        }
    }
    return changed;
}

/*kmeans_loop
 ************
Inputs:       Vectors pointer, centroids pointer, clusters pointer,
              dimension of vectors, k, number of vectors
Output:       None
Description:  Runs K-means loop of calculating the clusters based on
              the current centroids and calculating the centroids based on
              the current clusters, until reaching the stopping criteria of
              K-means algorithm. This loop is the same in K-means and K-means++
              and therefore used in both.
*/
void kmeans_loop(vector* vectors, vector* centroids, cluster* clusters, int dim, int k, int v_num){
    int iter_num = 0, max_iter = 300, centroids_changed = 1;
    while((iter_num < max_iter) & centroids_changed){
        calculate_clusters(vectors, centroids, clusters, dim, k, v_num);
        centroids_changed = update_centroids(centroids, clusters, dim, k);
        reset_clusters(clusters, dim, k);
        iter_num += 1;
    }
}

/*kmeans_c
 *********
Inputs:       Vectors (the vectors we run K-means on), the K value, the number of these
              vectors and their dimension
Output:       None
Description:  Run K-means algorithm on the vectors with the given k
              (In this function the centroid are initialized to be
              the first k vectors from the given vectors input, so it fits the
              K-means and not K-means++).
*/
void kmeans_c(vector* vectors, int k_val, int v_num, int dimension){
    int i;
    cluster* clusters;
    vector* centroids;
    if (k_val != 1){
        clusters = init_clusters(dimension, k_val);
        centroids = init_centroids(vectors,dimension,k_val);
        kmeans_loop(vectors, centroids, clusters, dimension, k_val, v_num);
        free_clusters(clusters, k_val);
    }
    else{
        centroids = average_vector(vectors, dimension, v_num);
    }
    for(i=0; i<k_val-1; i++){
        print_vector(centroids[i], dimension, 1);
    }
    print_vector(centroids[k_val-1], dimension, 0);
    free_vectors(centroids, k_val);
}


int main(int argc, char* argv[]) {
    int k_val;
    FILE* input_file;
    vectors_dataset* vd;
    reg_mat* t;
    if(argc != 4) error_message();
    k_val = strtol(argv[1], NULL, 10);
    input_file = fopen_and_check(argv[3], "r");
    vd = vectors_dataset_from_file(input_file);
    fclose(input_file);
    t = spk(vd, k_val, argv[2], 1);
    free_reg_mat(t);
    return 0;
}
