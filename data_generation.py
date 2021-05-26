from random import randint
import sys
# Floyd Warshall Algorithm in python


# The number of vertices
nV = int(sys.argv[1])

INF = 999

# Generation implementation
def generate_matrix():

    G = [ [ 0 for i in range(nV) ] for j in range(nV) ]

    # Adding vertices individually
    for i in range(nV):
        for j in range(nV):
            if j == i:
                G[i][j] = 0
            else :
                if randint(0,30) != 0:
                    G[i][j] = 0
                else :
                    G[i][j] = randint(0,100)
                    
    return G


def transform_matrix(G):

    # Adding vertices individually
    for i in range(nV):
        for j in range(nV):
            if G[i][j] == 0 and i!=j:
                G[i][j] = INF
    return G

# Algorithm implementation
def floyd_warshall(G):
    distance = list(map(lambda i: list(map(lambda j: j, i)), G))

    # Adding vertices individually
    for k in range(nV):
        for i in range(nV):
            for j in range(nV):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance


# Printing the matrix
def print_matrix(G):
    for i in range(nV):
        for j in range(nV):
            if(G[i][j] == INF):
                print("i", end=" ")
            else:
                print(G[i][j], end=" ")
        print(" ")


G = generate_matrix()

original_stdout = sys.stdout # Save a reference to the original standard output

with open('data/mat_'+str(nV), 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print_matrix(G)

G = transform_matrix(G)
distance = floyd_warshall(G)

with open('data/result_'+str(nV), 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print_matrix(distance)
    sys.stdout = original_stdout # Reset the standard output to its original value

print("done")