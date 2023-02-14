import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def pickle_load(path):
    with open(path, mode='rb') as f:
        obj = pickle.load(f)
    return obj


user2movie = pickle_load('../collaborative_filtering/user2movie.pkl')
movie2user = pickle_load('../collaborative_filtering/movie2user.pkl')
usermovie2rating = pickle_load('../collaborative_filtering/usermovie2rating.pkl')
usermovie2rating_test = pickle_load('../collaborative_filtering/usermovie2rating_test.pkl')

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N: ", N, "M: ", M)

# initialize variables
K = 10  # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

# prediction[i,j] = W[i].dot(U[j]) + b[i] + c[j] + mu


def get_loss(d):
    # d: (user_id, movie_id) -> rating
    # returns MSE
    N = float(len(d))
    sse = 0
    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (p - r) ** 2
    return sse / N


# train the parameters
epochs = 25
reg = 20.  # regularization penalty
train_losses = []
test_losses = []
for epoch in tqdm(range(epochs), position=0, leave=True, desc="epochs"):
    # update W and b
    for i in tqdm(range(N), position=0, leave=True):
        # for W
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # for b
        bi = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i, j)]
            matrix += np.outer(U[j], U[j])
            vector += (r - b[i] - c[j] - mu) * U[j]
            bi += (r - W[i].dot(U[j]) - c[j] - mu)

        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + reg)

    print('updated W and b')

    # U and c
    for j in tqdm(range(M), position=0, leave=True):
        # for U
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # for c
        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i, j)]
                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu) * W[i]
                cj += (r - W[i].dot(U[j]) - c[j] - mu)

            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + reg)
        except KeyError:
            # possible not to have any ratings for a movie
            pass

    print('updated U and c')

    train_losses.append(get_loss(usermovie2rating))
    test_losses.append(get_loss(usermovie2rating_test))
    print(f'train losses {epoch}: ', train_losses[-1])
    print(f'Test losses {epoch}: ', test_losses[-1])

print("Train losses: ", train_losses)
print("Test losses: ", test_losses)

# plot losses
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

