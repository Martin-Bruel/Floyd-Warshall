#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GATHER 1
#define SCATTER 2
#define BROADCAST 3

typedef struct Matrix
{
    long *array;
    int width;
    int height;
    bool row_opti;
} Matrix;

int broadcast(int data, int transmitter, int rank, int numprocs);
Matrix *scatter(Matrix *data, int size, bool row_opti, int transmitter, int rank, int numprocs);
void gather(int transmitter, int rank, int numprocs, Matrix *matrix);

Matrix *matrix_product(Matrix *m1, Matrix *m2);

void set(Matrix *matrix, int row, int column, long value);
long get(Matrix *matrix, int row, int column);

Matrix *generate_matrix(long *data, int h, int w, bool row_opti);
Matrix *create_matrix(int h, int w, bool row_opti);
void display_array(long *array, int size);
void display_matrix(Matrix *m);


int main(int argc, char *argv[])
{
    int rank, numprocs, N;
    Matrix *A, *B, *a, *b, *c;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    N = 8;

    N = broadcast(N, 0, rank, numprocs);

    A = create_matrix(N,N,true);
    B = create_matrix(N,N,false);
    if(rank == 0) display_matrix(A);
        
    a = scatter(A, N, true, 0, rank, numprocs);
    b = scatter(B, N, false, 0, rank, numprocs);

    c = generate_matrix((long *) malloc(a->height*a->width*sizeof(long)), a->height, a->width, a->row_opti);


    MPI_Barrier(MPI_COMM_WORLD);
    printf("RANK : %d\n",rank);
    display_matrix(a);
    display_matrix(b);
    fflush(stdout);


    
    //gather(0,rank,numprocs,A);
    

    MPI_Finalize();
    return 0;
}

int broadcast(int data, int transmitter, int rank, int numprocs)
{
    MPI_Status status;

    if(rank==transmitter)
    {
        MPI_Send(&data, 1, MPI_INT, (rank+1) % numprocs, BROADCAST, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT,  (rank-1) % numprocs, BROADCAST, MPI_COMM_WORLD, &status);
    }
    else
    {
        data = 0;
        MPI_Recv(&data, 1, MPI_INT, (rank-1) % numprocs, BROADCAST, MPI_COMM_WORLD, &status);
        MPI_Send(&data, 1, MPI_INT, (rank+1) % numprocs, BROADCAST, MPI_COMM_WORLD);
    }
    return data;
}


Matrix *scatter(Matrix *data, int size, bool row_opti, int transmitter, int rank, int numprocs)
{
    int block_size;
    MPI_Status status;
    long *block;

    //Dans le cas ou on est l'emmetteur
    //On découpe la matice en numprocs part
    //On envoie chaque part une par une à la machine suivante
    if(transmitter == rank)
    {        
        block_size = (size*size / numprocs);

        for(int p = 1; p < numprocs; p++)
        {
            //calcule du bloc à envoyé (le dernier bloc est envoyé en premier)
            block = data->array + (numprocs-p) * block_size;                                             
            MPI_Send(block, block_size, MPI_LONG, rank+1, SCATTER, MPI_COMM_WORLD);
            //printf("Rank %d : Send %d data to %d proc\n", rank, block_size, rank+1);
        }
        block = (long *)realloc(data->array, sizeof(long) * block_size);
        free(data);
        //display_array(block, block_size);
    }

    //Dans le cas ou on est une autre machine
    //On transmet les part qui ne nous sont pas dédié à la machine suivante
    //On conserve la derniere part
    else
    {
        MPI_Probe( ((rank-1)+numprocs)%numprocs, SCATTER, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_LONG, &block_size);
        block = (long *) malloc(sizeof(long) * block_size);

        for(int i = rank; i < numprocs - 1; i++)
        {
            MPI_Recv(block, block_size, MPI_LONG, (rank-1) + numprocs % numprocs, SCATTER, MPI_COMM_WORLD, &status);
            //printf("Rank %d : Receive %d data from %d proc\n", rank, block_size, (rank-1) + numprocs % numprocs);
            //printf("Rank %d : Send %d data to %d proc\n", rank, block_size, rank+1);
            MPI_Send(block, block_size, MPI_LONG, rank+1, SCATTER, MPI_COMM_WORLD);
        }
        //printf("Rank %d : Receive %d data from %d proc\n", rank, block_size, (rank-1) + numprocs % numprocs);
        MPI_Recv(block, block_size, MPI_LONG, (rank-1) + numprocs % numprocs, SCATTER, MPI_COMM_WORLD, &status);
        //display_array(block, block_size);
    }


    //On retourne une matrice construite avec le block recu
    if(row_opti) return generate_matrix(block, block_size / size, size, row_opti);
    else return generate_matrix(block, size, block_size / size, row_opti);
}

void gather(int transmitter, int rank, int numprocs, Matrix *matrix)
{
    MPI_Status status;
    int block_size = matrix->width * matrix->height;
    long *block = matrix->array;

    //Dans le cas ou on est l'emmetteur
    //On genere une matrice vide
    //rempli le debut de la matrice avec notre block
    //rempli le reste de la matrice avec les blocs qui arrivent
    if(rank==transmitter)
    {
        int size = block_size*numprocs;
        Matrix *result = generate_matrix(malloc(size*sizeof(long)),matrix->width, matrix->width, true);
        memmove(result->array,block, block_size * sizeof(long));
        for(int p = 1; p < numprocs; p++)
        {
            MPI_Recv(block, block_size, MPI_LONG, (rank-1) + numprocs % numprocs, GATHER, MPI_COMM_WORLD, &status);
            memmove(result->array + (numprocs-p) * block_size, block, block_size  * sizeof(long));
        }
        display_matrix(result);
    }
    //Dans le cas ou on est une autre machine
    //On envoie notre block au suivant
    //On receptionne les block des machines precedentes
    //On envoie leurs blocks 
    else
    {
        //printf("Rank %d : Send %d data to %d proc\n", rank, block_size, (rank+1) % numprocs);
        MPI_Send(block, block_size, MPI_LONG, (rank+1) % numprocs, GATHER, MPI_COMM_WORLD);
        for(int i = numprocs - rank; i < numprocs - 1; i++)
        {
            MPI_Recv(block, block_size, MPI_LONG, (rank-1) + numprocs % numprocs, GATHER, MPI_COMM_WORLD, &status);
            //printf("Rank %d : Receive %d data from %d proc\n", rank, block_size, (rank-1) + numprocs % numprocs);
            //printf("Rank %d : Send %d data to %d proc\n", rank, block_size, (rank+1) % numprocs);
            MPI_Send(block, block_size, MPI_LONG, (rank+1) % numprocs, GATHER, MPI_COMM_WORLD);
        }
    }
}





void set(Matrix *matrix, int row, int column, long value)
{
    if(matrix->row_opti) matrix->array[row*matrix->width+column]=value;
    else matrix->array[column*matrix->height+row]=value;
}

long get(Matrix *matrix, int row, int column)
{
    if(matrix->row_opti) return matrix->array[row*matrix->width+column];
    else return matrix->array[column*matrix->height+row];
}



Matrix *generate_matrix(long *data, int h, int w, bool row_opti)
{
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    m->height=h;
    m->width=w;
    m->row_opti=row_opti;
    m->array = data;
    return m;
}

Matrix *create_matrix(int h, int w, bool row_opti)
{
    Matrix *m = generate_matrix((long *) malloc(h*w*sizeof(long)), h,w,row_opti);
    int i = 1;

    for(int r = 0; r < m->height; r++)
    {
        for(int c = 0; c < m->width; c++)
        {
            set(m,r,c,i++);
        }
    }
    return m;
}

void display_matrix(Matrix *m)
{
    for(int r = 0; r < m->height; r++)
    {
        for(int c = 0; c < m->width; c++)
        {
            printf("%5ld ", get(m,r,c));
        }
        printf("\n");
    }
}

void display_array(long *array, int size)
{
    for(int i = 0; i < size; i++)
    {
        printf("%5ld ", array[i]);        
    }
    printf("\n");
}


Matrix *matrix_product(Matrix *m1, Matrix *m2)
{
    if(m1->width!=m2->height) return NULL;
    Matrix *res = generate_matrix((long *) malloc(m1->height*m2->width*sizeof(long)), m1->height, m2->width, false);
    int n=m1->height, p=m1->width, m=m2->width;
    for(int r = 0; r < n; r++)
    {
        for (int c = 0; c < m; c++)
        {
            int tmp=0;
            for (int i = 0; i < p; i++)
            {
                tmp += get(m1,r,i) *get(m2,i,c);   
            }
            set(res,r,c,tmp);
        }
    }
    return res;
}