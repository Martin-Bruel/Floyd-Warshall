## Build

```mpicc -o bin/bruel src/bruel.c -Wall```

## Test

```mpirun -np 4 ./bin/bruel test```

## Run

```mpirun -np 4 ./bin/bruel```