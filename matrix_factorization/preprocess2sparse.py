import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz
from sklearn.model_selection import train_test_split

df = pd.read_csv('../movielens_data/edited_rating.csv')

N = df.userId.max() + 1  # number of users
M = df.movieId.max() + 1  # number of movies

# split into train and test
df_train, df_test = train_test_split(df, test_size=0.2)

A = lil_matrix((N, M))
print('Calling: update_train')
count = 0


def update_train(row):
    global count
    count += 1
    if count % 100_000 == 0:
        print('Processed: %.3f' % (float(count)/len(df_train)))

    i = int(row.userId)
    j = int(row.movie_idx)
    A[i, j] = row.rating


df_train.apply(update_train, axis=1)

# mask, to tell us which entries exist and which do not
A = A.tocsr()
mask = (A > 0)
save_npz('Atrain.npz', A)

A_test = lil_matrix((N, M))
print('Calling: update_test')
count = 0


def update_test(row):
    global count
    count += 1
    if count % 100_000 == 0:
        print('Processed: %.3f' % (float(count)/len(df_test)))

    i = int(row.userId)
    j = int(row.movie_idx)
    A_test[i, j] = row.rating


df_test.apply(update_test, axis=1)
A_test = A_test.tocsr()
mask_test = (A_test > 0)
save_npz('Atest.npz', A_test)
