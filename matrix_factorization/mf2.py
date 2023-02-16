import pickle
import numpy as np
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

# convert user2movie and movie2user to include ratings
print('converting...')
user2movie_rating = {}
for i, movies in user2movie.items():
    r = np.array([usermovie2rating[(i, j)] for j in movies])
    user2movie_rating[i] = (movies, r)

movie2user_rating = {}
for j, users in movie2user.items():
    r = np.array([usermovie2rating[(i, j)] for i in users])
    movie2user_rating[j] = (users, r)

movie2user_rating_test = {}
for (i, j), r in usermovie2rating_test.items():
    if j not in movie2user_rating_test:
        movie2user_rating_test[j] = [[i], [r]]
    else:
        movie2user_rating_test[j][0].append(i)
        movie2user_rating_test[j][1].append(r)

for j, (users, r) in movie2user_rating_test.items():
    movie2user_rating_test[j][1] = np.array(r)
print('conversion done')

# initialize variables
K = 10  # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))


def get_loss(m2u):
    # d: movie_id -> (user_ids, ratings)
    N = 0.
    sse = 0
    for j, (u_ids, r) in m2u.items():
        p = W[u_ids].dot(U[j]) + b[u_ids] + c[j] + mu
        delta = p - r
        sse += delta.dot(delta)
        N += len(r)
    return sse / N


# train the parameters
epochs = 25
reg = 0.1  # regularization penalty
train_losses = []
test_losses = []

for epoch in tqdm(range(epochs), position=0, leave=True):
    # update W and b
    for i in tqdm(range(N), position=0, leave=True):
        m_ids, r  = user2movie_rating[i]
        matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg
        vector = (r - b[i] - c[m_ids] - mu).dot(U[m_ids])
        bi = (r - U[m_ids].dot(W[i]) - c[m_ids] - mu).sum()

        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + reg)

    # update U and c
    for j in tqdm(range(M), position=0, leave=True):
        try:
            u_ids, r = movie2user_rating[j]
            matrix = W[u_ids].T.dot(W[u_ids]) + np.eye(K) * reg
            vector = (r - b[u_ids] - c[j] - mu).dot(W[u_ids])
            cj = (r - W[u_ids].dot(U[j]) - b[u_ids] - mu).sum()

            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + reg)
        except KeyError:
            pass

    train_losses.append(get_loss(movie2user_rating))
    test_losses.append(get_loss(movie2user_rating_test))
    print(f'train losses {epoch}: ', train_losses[-1])
    print(f'Test losses {epoch}: ', test_losses[-1])

# plot losses
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()
