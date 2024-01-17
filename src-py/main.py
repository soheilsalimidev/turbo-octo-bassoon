import numpy as np
import os
import pandas as pd
import sys

file_dir = os.path.dirname(os.path.realpath('__file__'))
PATH_RATING = os.path.join(file_dir, 'data/ratings.csv') 
PATH_MOVIE = os.path.join(file_dir, 'data/movies.csv') 

def norm(matrix):
    norm_sum = 0
    for e in matrix:
        norm_sum += e ** 2
    norm_sum = np.sqrt(norm_sum)
    return norm_sum


def power_iteration(matrix, simulations=100):
    # Create a random initial vector
    rnd_vec = np.random.rand(matrix.shape[1])
    for _ in range(simulations):
        # Compute the matrix-by-vector product Ab
        m_b = np.dot(matrix, rnd_vec)
        # Compute the norm
        b_m_norm = norm(m_b)
        # Re-normalize the vector
        rnd_vec = m_b / b_m_norm

    # Compute and return the largest eigenvalue and its corresponding eigenvector
    return np.dot(np.dot(matrix, rnd_vec), rnd_vec) / np.dot(rnd_vec, rnd_vec), rnd_vec

# calculate all eigenvalues and eigenvectors
def compute_eigen_values_vectors(matrix, simulations=100):
    # Get the size of the matrix
    n = matrix.shape[0]
    # Initialize the eigenvectors and eigenvalues with zero
    vectors = np.zeros((n, n))
    values = np.zeros(n)

    for ind in range(n):
        # Compute the largest eigenvalue and its corresponding eigenvector using power iteration
        val, vec = power_iteration(matrix, simulations)
        values[ind] = val
        vectors[:, ind] = vec
        matrix = matrix - val * np.outer(vec, vec)

    return values, vectors

def SVD(matrix):
    # calculate eigen values and eigen vectors
    eigen_values, eigen_vectors = compute_eigen_values_vectors(np.dot(matrix.T, matrix))
    # sort eigen values and eigen vectors
    # since we only need eigen_vectors for V_T we only sort them
    idx = eigen_values.argsort()[::-1]
    # eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # calculate SVD
    # in here we only calculate V_t since its the only thing we need for this project but other 
    # prameters can be calculated too, as shown below

    # S = np.sqrt(eigen_values)
    # U = matrix.dot(V) / S
    V = eigen_vectors
    return  V.T

def recommend(liked_movie, VT):
    rec = []
    i = 0
    for column in zip(*VT): 
        rec.append([i,np.dot(column,VT[liked_movie])])
        i = i+1
    final_rec = [i[0] for i in sorted(rec, key=lambda x: x[1],reverse=True)]
    return final_rec



# read csv
rating = pd.read_csv(PATH_RATING)
movie = pd.read_csv(PATH_MOVIE)
# get the requested user as args
user_id = int(sys.argv[1])
# read the csv file and change it to matrix
merged = pd.merge(rating, movie, on="movie_id")
print("for this algoritem to run in reasonable amount of time we take only first 500 moives but its can change to its original size. this's just faster for testing \n")
# for this algoritem to run in reasonable amount of time we take 20 user and 100 moives but its can change to its original size. this is just faster to be tested
matrix = np.nan_to_num(merged.pivot_table(index="user_id", columns="title", values="rating" ).iloc[:610 , :500] , nan= 0.1)
# Compute the SVD(only the V_T part is need for this project)
V_t = SVD(matrix)
# find the recommend moive(sorted from best to worse to be liked)
recommendList = recommend(user_id , V_t)
# create data frame with moive info that has been recommend
df = pd.DataFrame(data=recommendList , index=np.arange(len(recommendList)) ,columns=['movie_id']).merge(movie, on="movie_id")
#save the output to csv file
df.to_csv("output.csv")
print(df)

print("all data is in output.csv")

