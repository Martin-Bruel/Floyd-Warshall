#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#define GATHER 1
#define SCATTER 2
#define BROADCAST 3
#define PROCESS 4

#define TRANSMITTER 0

#define NEXT(r,p) ((r + 1) + p) % p
#define PREVIOUS(r,p) ((r - 1) + p) % p
#define CURRENT(r,p) (r + p) % p

typedef struct Matrix
{
    long *array;
    int width;
    int height;
    bool row_opti;
} Matrix;


//-----------------------------------------------------------------
//-------------------------DECLARATION-----------------------------
//-----------------------------------------------------------------

//Fonctions de haut niveau
int broadcast(int data, int transmitter, int rank, int numprocs);                                   //emmet data sur toutes les machines de l'anneau
Matrix *scatter(Matrix *data, int size, bool row_opti, int transmitter, int rank, int numprocs);    //transmet une part de data à chaque machine de l'anneau
Matrix *gather(int transmitter, int rank, int numprocs, Matrix *matrix);                            //transmet chaque part de data à l'emmeteur
Matrix *process(Matrix *a, Matrix *b, int rank, int numprocs);                                      //retourne la matrice traité

//Matrice manipulation
Matrix *matrix_product(Matrix *m1, Matrix *m2);                                                     //retourne le produit matriciel entre a et b
Matrix *matrix_process(Matrix *m1, Matrix *m2);                                                     //retourne le produit matriciel en remplacant l'opération de multiplication par une addition et l'opération de somme par le minimum 
void replace(Matrix *a, Matrix *b, int row, int column);                                            //remplace a par la matrice b à l'index donné

//Utils
void set(Matrix *matrix, int row, int column, long value);                                          //assigne la valeur dans la bonne case de la matrice
long get(Matrix *matrix, int row, int column);                                                      //retourne la valeur à la case correspondance
int size(Matrix *matrix);                                                                           //retourne le nombre d'element d'une matrice
void display_array(long *array, int size);                                                          //affiche le tableau
void display_matrix(Matrix *m);                                                                     //affiche la matrice

//Creation matrice
Matrix *generate_matrix(long *data, int h, int w, bool row_opti);                                   //genere une matrice
Matrix *create_matrix(int seed, int h, int w, bool row_opti);                                       //creer une matrice 
Matrix *load_matrix(char *path, int numprocs);                                                      //charge une matrice depuis un fichier en la transformant avec i et l'ajustant pour etre divisible par le nombre de procos
Matrix *copy_matrix(Matrix *m, bool row_opti);                                                      //copy une matrix avec l'optimisation demandée

//Tests
int test(int rank, int numprocs);                                                                   //tests





//-----------------------------------------------------------------
//------------------------------MAIN-------------------------------
//-----------------------------------------------------------------
int main(int argc, char *argv[])
{
    int rank, numprocs, N;
    Matrix *A, *B, *a, *b;

    //initialisation de MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //vérifie si il y a le bon nombre d'argument
    if(argc != 2) 
    {
        if(rank == 0) printf("Program is call with wrong numbers of argument... exit\n");
        MPI_Finalize();
        return 0;
    }

    //lance les tests
    if(strcmp(argv[1],"test")==0)
    {
        test(rank, numprocs);
        MPI_Finalize();
        return 0;
    }

    //lit la matrice si il s'agit de l'emmetteur et en déduit A, B et N
    if(rank == TRANSMITTER)
    {
        A = load_matrix(argv[1], numprocs);
        N = A->height;
        B=copy_matrix(A, false);
    }

    //transmet N à toutes les machines du réseau
    N = broadcast(N, 0, rank, numprocs);

    //transmet un bloc de B a chaque machine du réseau
    b = scatter(B, N, false, 0, rank, numprocs);
    a = scatter(A, N, true, 0, rank, numprocs);

    //multiplie a avec b
    for(int i = 0; i < N; i++) a = process(a,b,rank,numprocs);

    //assemble a pour reconstituer la matrice finale
    A = gather(0,rank,numprocs,a);

    //affiche le résultat
    if(rank == 0) display_matrix(A);
    
    MPI_Finalize();
    return 0;
}



//-----------------------------------------------------------------
//-------------------FONCTIONS DE HAUT NIVEAU----------------------
//-----------------------------------------------------------------
int broadcast(int data, int transmitter, int rank, int numprocs)
{
    MPI_Status status;

    if(rank==transmitter)
    {
        MPI_Send(&data, 1, MPI_INT, NEXT(rank, numprocs), BROADCAST, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT,  PREVIOUS(rank, numprocs), BROADCAST, MPI_COMM_WORLD, &status);
    }
    else
    {
        data = 0;
        MPI_Recv(&data, 1, MPI_INT, PREVIOUS(rank, numprocs), BROADCAST, MPI_COMM_WORLD, &status);
        MPI_Send(&data, 1, MPI_INT, NEXT(rank, numprocs), BROADCAST, MPI_COMM_WORLD);
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
            MPI_Send(block, block_size, MPI_LONG, NEXT(rank, numprocs), SCATTER, MPI_COMM_WORLD);
        }
        block = (long *)realloc(data->array, sizeof(long) * block_size);
        free(data);
    }

    //Dans le cas ou on est une autre machine
    //On transmet les part qui ne nous sont pas dédié à la machine suivante
    //On conserve la derniere part
    else
    {
        MPI_Probe(PREVIOUS(rank, numprocs), SCATTER, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_LONG, &block_size);
        block = (long *) malloc(sizeof(long) * block_size);

        for(int i = rank; i < numprocs - 1; i++)
        {
            MPI_Recv(block, block_size, MPI_LONG, PREVIOUS(rank, numprocs), SCATTER, MPI_COMM_WORLD, &status);
            MPI_Send(block, block_size, MPI_LONG, NEXT(rank, numprocs), SCATTER, MPI_COMM_WORLD);
        }
        MPI_Recv(block, block_size, MPI_LONG, PREVIOUS(rank, numprocs), SCATTER, MPI_COMM_WORLD, &status);
    }


    //On retourne une matrice construite avec le block recu
    if(row_opti) return generate_matrix(block, block_size / size, size, row_opti);
    else return generate_matrix(block, size, block_size / size, row_opti);
}

Matrix *gather(int transmitter, int rank, int numprocs, Matrix *matrix)
{
    MPI_Status status;
    Matrix *result;
    int block_size = size(matrix);
    long *block = matrix->array;

    //Dans le cas ou on est l'emmetteur
    //On genere une matrice vide
    //rempli le debut de la matrice avec notre block
    //rempli le reste de la matrice avec les blocs qui arrivent
    if(rank==transmitter)
    {
        int size = block_size*numprocs;
        result = generate_matrix(malloc(size*sizeof(long)),matrix->width, matrix->width, true);
        memmove(result->array,block, block_size * sizeof(long));
        for(int p = 1; p < numprocs; p++)
        {
            MPI_Recv(block, block_size, MPI_LONG, PREVIOUS(rank, numprocs), GATHER, MPI_COMM_WORLD, &status);
            memmove(result->array + (numprocs-p) * block_size, block, block_size  * sizeof(long));
        }
    }
    //Dans le cas ou on est une autre machine
    //On envoie notre block au suivant
    //On receptionne les block des machines precedentes
    //On envoie leurs blocks 
    else
    {
        MPI_Send(block, block_size, MPI_LONG, NEXT(rank, numprocs), GATHER, MPI_COMM_WORLD);
        for(int i = numprocs - rank; i < numprocs - 1; i++)
        {
            MPI_Recv(block, block_size, MPI_LONG, PREVIOUS(rank, numprocs), GATHER, MPI_COMM_WORLD, &status);
            MPI_Send(block, block_size, MPI_LONG, NEXT(rank, numprocs), GATHER, MPI_COMM_WORLD);
        }
    }
    return result;
}


Matrix *process(Matrix *a, Matrix *b, int rank, int numprocs)
{
    Matrix *p;
    long *tmp;
    MPI_Status status; 
    Matrix *c = generate_matrix((long *) malloc(size(a)*sizeof(long)), a->height, a->width, a->row_opti);

    //Pour chaque procos on traite la matrice
    //  On ajoute le résultat dans la matrice final
    //  Si on est pair, on envoie et on receprionne la matrice b
    //  Si on est impaire, on receprionne et on envoie la matrice b
    for(int i = 0; i < numprocs; i++)
    {
        //On fait le produit des 2 matrices
        p = matrix_process(a,b);

        //On remplace dans la matrice résultante la multiplication trouvée
        //à l'emplacement déterminé celon le rank, l'iteration et la taille d'un bloc
        replace(c, p, 0, CURRENT(rank-i,numprocs)*a->width/numprocs);
        free(p);

        if(rank % 2 == 0)
        {
            MPI_Send(b->array, size(b), MPI_LONG, NEXT(rank, numprocs), PROCESS, MPI_COMM_WORLD);
            MPI_Recv(b->array, size(b), MPI_LONG, PREVIOUS(rank, numprocs), PROCESS, MPI_COMM_WORLD, &status);
        }
        else
        {
            tmp = (long *) malloc(size(b)*sizeof(long));
            MPI_Recv(tmp, size(b), MPI_LONG, PREVIOUS(rank, numprocs), PROCESS, MPI_COMM_WORLD, &status);
            MPI_Send(b->array, size(b), MPI_LONG, NEXT(rank, numprocs), PROCESS, MPI_COMM_WORLD);
            b->array=tmp;
        }
    }
    return c;
}




//-----------------------------------------------------------------
//--------------------MANIPULATION DE MATRICE----------------------
//-----------------------------------------------------------------
Matrix *matrix_product(Matrix *m1, Matrix *m2)
{
    //genere une matrice avec pour hauteur m1 et pour largeur m2
    Matrix *res = generate_matrix((long *) malloc(m1->height*m2->width*sizeof(long)), m1->height, m2->width, true);
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


Matrix *matrix_process(Matrix *m1, Matrix *m2)
{
    //genere une matrice avec pour hauteur m1 et pour largeur m2
    Matrix *res = generate_matrix((long *) malloc(m1->height*m2->width*sizeof(long)), m1->height, m2->width, true);
    int n=m1->height, p=m1->width, m=m2->width;

    //remplace dans la multiplication classique de matrice l'opération de multiplication par une addition et l'opération de somme par le minimum 
    for(int r = 0; r < n; r++)
    {
        for (int c = 0; c < m; c++)
        {
            long min = LONG_MAX, curr;
            for (int i = 0; i < p; i++)
            {
                if(get(m1,r,i) == LONG_MAX || get(m2,i,c) == LONG_MAX) curr = LONG_MAX;
                else curr = get(m1,r,i)+get(m2,i,c);
                min = min<curr ? min:curr;
            }
            set(res,r,c,min);
        }
    }
    return res;
}


void replace(Matrix *a, Matrix *b, int row, int column)
{
    //remplace aux coordonnées row colums et aux suivantes les valeurs de a par celles de b 
    int rb = 0, cb = 0;
    for(int ra = row; ra < b->height+row; ra++)
    {   
        for(int ca = column; ca < b->width+column; ca++)
        {
            set(a, ra, ca, get(b,rb,cb++));
        }
        cb = 0;
        rb++;
    }
}




//-----------------------------------------------------------------
//----------------------------UTILS--------------------------------
//-----------------------------------------------------------------
void set(Matrix *matrix, int row, int column, long value)
{
    //change la valeur aux coordonnées row column
    //la position dans la memoire est determinée en foction de l'optimisation
    if(matrix->row_opti) matrix->array[row*matrix->width+column]=value;
    else matrix->array[column*matrix->height+row]=value;
}

long get(Matrix *matrix, int row, int column)
{
    //retourne la valeur aux coordonnées row column
    //la position dans la memoire est determinée en foction de l'optimisation
    if(matrix->row_opti) return matrix->array[row*matrix->width+column];
    else return matrix->array[column*matrix->height+row];
}

int size(Matrix *matrix)
{
    //retourne le nombre de données dans la matrice
    return matrix->width*matrix->height;
}

void display_matrix(Matrix *m)
{
    int Nlimit;
    //créer un délimiteur pour ne pas afficher les colonnes et lignes qui ont été ajoutées pour s'adapter au nombre de processeurs
    for(Nlimit = 0; Nlimit < m->height && get(m, Nlimit, Nlimit) != LONG_MAX; Nlimit++);
    
    //affiche la matrice ligne par ligne avec chaque valeur séparées par une espace
    for(int r = 0; r < Nlimit; r++)
    {
        for(int c = 0; c < Nlimit; c++)
        {
            if(get(m,r,c) == LONG_MAX) printf("%c ", 'i');
            else printf("%ld ", get(m,r,c));
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

//-----------------------------------------------------------------
//--------------------CREATION DE MATRICES-------------------------
//-----------------------------------------------------------------
Matrix *generate_matrix(long *data, int h, int w, bool row_opti)
{
    //génere une matrice en mémoire avec les données data
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    m->height=h;
    m->width=w;
    m->row_opti=row_opti;
    m->array = data;
    return m;
}

Matrix *create_matrix(int seed, int h, int w, bool row_opti)
{
    //créer une matrice avec des valeurs tests
    Matrix *m = generate_matrix((long *) malloc(h*w*sizeof(long)), h,w,row_opti);
    int i = seed+1;

    for(int r = 0; r < m->height; r++)
    {
        for(int c = 0; c < m->width; c++)
        {
            set(m,r,c,i++);
        }
    }
    return m;
}

Matrix *load_matrix(char *path, int numprocs)
{
    FILE * file;
    long val;
    int size_alloc = 256;
    int size=0, N, Nlimit;

    file = fopen(path, "r");
    if(file==NULL) return NULL;

    long *data = (long *) malloc(size_alloc*sizeof(long));

    //lit toutes les données et les range dans le tableau data
    //alloue de la mémoire suplémentaire si necessaire
    while(fscanf(file, " %ld", &val) == 1)
    {
        if(size == size_alloc)
        {
            size_alloc=size_alloc*size_alloc;
            data = realloc(data, size_alloc*sizeof(long));
        }
        data[size++] = val; 
    }
    fclose(file);

    N = sqrt(size);
    
    if(N % numprocs != 0) Nlimit = N + numprocs - (N % numprocs);
    else Nlimit = N;

    //Génere une matrice qui est divisible par le nombre de processeur
    Matrix *m = generate_matrix( (long *) malloc(Nlimit*Nlimit * sizeof(long)), Nlimit, Nlimit, true);

    
    //Ajoute des valeurs infinie lorsque il yu a un 0 et que celui-ci ne se trouve pas en diagonal
    //Ajoute des valeurs infinie pour TOUTES les valeur ajouter pour s'adapter au nombre de processeurs
    for(int r = 0; r < N; r++)
    {
        for(int c = 0; c <N; c++)
        {
            val = data[r*N+c];
            if(val == 0 && r!=c) set(m,r,c,LONG_MAX);
            else set(m,r,c,val);

            if(c==N-1)
            {
                for(int i = c+1; i < Nlimit; i++)
                {
                    set(m,r,i,LONG_MAX);
                }
            }
        }

        if(r==N-1)
        {
            for(int i = r+1; i < Nlimit; i++)
            {
                for(int c = 0; c <Nlimit; c++)
                {
                    set(m,i,c, LONG_MAX);
                }
            }
        }
    }

    return m;
}


Matrix *copy_matrix(Matrix *m, bool row_opti)
{
    Matrix *copy = generate_matrix((long *) malloc(size(m)*sizeof(long)), m->height, m->width, row_opti);
    
    //copy chaque valeurs de m dans la nouvelle matrice en respectant l'optimisation de colonne
    for(int r = 0; r < m->height; r++)
    {
        for(int c = 0; c < m->width; c++)
        {
            set(copy,r,c,get(m,r,c));
        }
    }
    return copy;
}





//-----------------------------------------------------------------
//----------------------------TESTS--------------------------------
//-----------------------------------------------------------------
int replace_test()
{
    Matrix *a = create_matrix(0,4,4,true);
    Matrix *b = create_matrix(100,4,2,true);
    replace(a,b,0,0);
    long tab1[16] = {101,102,3,4,103,104,7,8,105,106,11,12,107,108,15,16};
    if(memcmp(tab1, a->array, 16*sizeof(long))) return 1;

    a = create_matrix(0,2,8,true);
    b = create_matrix(100,2,2,true);
    replace(a,b,0,2);
    long tab2[16] = {1,2,101,102,5,6,7,8,9,10,103,104,13,14,15,16};
    if(memcmp(tab2, a->array, 16*sizeof(long))) return 1;
    b = create_matrix(100,2,2,true);
    replace(a,b,0,4);
    long tab3[16] = {1,2,101,102,101,102,7,8,9,10,103,104,103,104,15,16};
    if(memcmp(tab3, a->array, 16*sizeof(long))) return 1;
    return 0;
}

int next_previous_test()
{
    int rank = 0;
    int numprocs = 4;

    if(NEXT(rank, numprocs) != 1) return 1;
    if(PREVIOUS(rank, numprocs) != 3) return 1;

    rank = 3;
    if(NEXT(rank, numprocs) != 0) return 1;
    if(PREVIOUS(rank, numprocs) != 2) return 1;
    return 0;
}

int load_matrix_test()
{
    Matrix *m = load_matrix("data/mat_2", 1);
    if(m == NULL) return 1;
    long tab1[16] = {0, 1, 2, LONG_MAX, LONG_MAX, 0, LONG_MAX, 1, LONG_MAX, 3, 0, 6, LONG_MAX, LONG_MAX, LONG_MAX, 0};
    if(memcmp(tab1, m->array, 16*sizeof(long))) return 1;
    free(m);
    m = load_matrix("data/mat_3", 1);
    long tab2[64] = {   
                        0, 4, 2, LONG_MAX, 6, 9, LONG_MAX, 8, 
                        6, 0, 2, LONG_MAX, 9, 2, LONG_MAX, 8, 
                        2, 1, 0, 8, 4, LONG_MAX, 4, 2, 
                        4, 6, 3, 0, 4, LONG_MAX, 3, 8, 
                        7, 8, 1, LONG_MAX, 0, 9, 3, LONG_MAX, 
                        3, 8, 8, 1, 8, 0, LONG_MAX, 7, 
                        LONG_MAX, 2, 9, 3, 3, 4, 0, 6, 
                        6, 4, 5, 2, 3, 1, 7, 0 
                    };
    if(memcmp(tab2, m->array, 64*sizeof(long))) return 1;
    free(m);
    m = load_matrix("data/mat_1", 4);
    if(m == NULL) return 1;
    long tab3[16] = {0, 1, 2, LONG_MAX, 1, 0, LONG_MAX, LONG_MAX, LONG_MAX, 3, 0, LONG_MAX, LONG_MAX, LONG_MAX, LONG_MAX, LONG_MAX};
    if(memcmp(tab3, m->array, 16*sizeof(long))) return 1;
    free(m);
    return 0;
}

int matrix_process_test()
{
    Matrix *m1 = load_matrix("data/mat_2", 1);
    Matrix *m2 = copy_matrix(m1, false);
    Matrix *m3 = matrix_process(m1,m2);
    long tab[16] = {
                        0, 1, 2, 2,
                        LONG_MAX,0,LONG_MAX,1,
                        LONG_MAX,3,0,4,
                        LONG_MAX,LONG_MAX,LONG_MAX,0
                    };
    if(memcmp(tab, m3->array, 16*sizeof(long))) return 1;
    free(m3);
    free(m1);
    free(m2);
    return 0;
}

int multi_matrix_process_test()
{
    Matrix *m1 = load_matrix("data/mat_3", 1);
    for(int i = 0; i < 8; i++)
    {
        Matrix *m2 = copy_matrix(m1, false);
        m1 = matrix_process(m1,m2);
    }
    Matrix *res = load_matrix("data/result_3", 1);
    if(memcmp(res->array, m1->array, size(res)*sizeof(long))) return 1;
    return 0;
}

int run_test(char *test_name, int (*test_fnct)(), int id)
{
    if(test_fnct()) 
    {
        printf("\033[0;31mTest %d failed : '%s'\n\033[0m", id, test_name);
        return 1;
    }
    else 
    {
        printf("\033[0;32mTest %d pass : '%s'\n\033[0m", id, test_name);
        return 0;
    }
}

int test(int rank, int numprocs)  
{
    int nb_failed = 0, id = 0;
    if(rank==0)
    {
        nb_failed+=run_test("replace", replace_test, ++id);
        nb_failed+=run_test("next_previous", next_previous_test, ++id);
        nb_failed+=run_test("load_matrix", load_matrix_test, ++id);
        nb_failed+=run_test("matrix_process", matrix_process_test, ++id);
        nb_failed+=run_test("multi_matrix_process", multi_matrix_process_test, ++id);
        printf("%d test failed.\n", nb_failed);
    }
    return 0;
}