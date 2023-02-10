import pickle
import numpy as np
from tqdm import trange
from sortedcontainers import SortedList


# load the data
def pickle_load(path):
    with open(path, mode='rb') as f:
        obj = pickle.load(f)
    return obj


user2movie = pickle_load('user2movie.pkl')
movie2user = pickle_load('movie2user.pkl')
usermovie2rating = pickle_load('usermovie2rating.pkl')
usermovie2rating_test = pickle_load('usermovie2rating_test.pkl')

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N: ", N, "M: ", M)

K = 20  # number of neighbors we'd like to select
limit = 5  # number of common movies users must have in common in order to consider
neighbors = []
averages = []
deviations = []

for i in trange(M):
    users_i = movie2user[i]
    users_i_set = set(users_i)

    rating_i = {user: usermovie2rating[(user, i)] for user in users_i}
    avg_i = np.mean(list(rating_i.values()))
    dev_i = {user: rating - avg_i for user, rating in rating_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    averages.append(avg_i)
    deviations.append(dev_i)

    sl = SortedList()
    for j in range(M):
        if j != i:
            users_j = movie2user[j]
            users_j_set = set(users_j)
            common_users = (users_i_set & users_j_set)
            if len(common_users) > limit:
                rating_j = {user: usermovie2rating[(user, j)] for user in users_j}
                avg_j = np.mean(list(rating_j.values()))
                dev_j = {user: rating_j - avg_j for user in rating_j.items()}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                numerator = sum(dev_i[m] * dev_j[m] for m in common_users)
                w_ij = numerator / (sigma_i * sigma_j)

                # insert into sorted list and truncate
                # negative weight, because list is sorted ascending
                sl.add((-w_ij, j))
                if len(sl) > K:
                    del sl[-1]

    neighbors.append(sl)


def predict(i, u):
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        try:
            numerator += -neg_w * deviations[j][u]
            denominator = abs(neg_w)
        except KeyError:
            # neighbor may not have rated the same movie
            pass

    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]

    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction


train_predictions = []
train_targets = []
for (u, m), target in usermovie2rating.items():
    prediction = predict(m, u)
    train_predictions.append(prediction)
    train_targets.append(target)

test_predictions = []
test_targets = []
for (u, m), target in usermovie2rating_test.items():
    prediction = predict(m, u)
    test_predictions.append(prediction)
    test_targets.append(target)


def mse(p, t):
    p = np.array(p)
    p = np.array(t)
    return np.mean((p - t)**2)


print('Train MSE: ', mse(train_predictions, train_targets))
print('Test MSE: ', mse(test_predictions, test_targets))
