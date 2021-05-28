## Install

```sudo apt-get install libopenmpi-dev openmpi-bin libhdf5-openmpi-dev```

## Generate data

```python3 generate_data.py <matrix_size>```

## Build

```mpicc -o bin/bruel src/bruel.c -Wall -lm -fopenmp```

## Test

```mpirun -np 4 ./bin/bruel test```

## Run

```mpirun -np 4 ./bin/bruel <data_file>```

## Evaluate

```python3 evaluate.py```