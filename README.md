## Build

```mpicc -o bin/bruel src/bruel.c -Wall -lm -fopenmp```

## Test

```mpirun -np 4 ./bin/bruel test```

## Run

```mpirun -np 4 ./bin/bruel data/mat_2```