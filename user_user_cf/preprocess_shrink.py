import pickle
import numpy as np
import pandas as pd
from collections import Counter

# load in the data
df = pd.read_csv('../movielens_data/edited_rating.csv')
print('Original dataframse size: ', len(df))

N = df.userId.max() + 1  # number of users
M = df.movie_idx.max() + 1  # number of movies

user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)

# number of users and movies to keep
n = 10000
m = 2000

user_ids = [u for u, c in user_ids_count.most_common(n)]
movie_ids = [u for u, c in movie_ids_count.most_common(m)]

# make a new df
df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()

# remake user ids and movie ids since they are no longer sequential
new_user_id_map = {}
for idx, user_id in enumerate(user_ids):
    new_user_id_map[user_id] = idx

new_movie_id_map = {}
for idx, movie_id in enumerate(movie_ids):
    new_movie_id_map[movie_id] = idx

df_small['userId'] = df_small.userId.map(new_user_id_map)
df_small['movie_idx'] = df_small.movie_idx.map(new_movie_id_map)

print('Max user id: ', df_small.userId.max())
print('Max movie id: ', df_small.movie_idx.max())
print('Small dataframe size: ', len(df_small))

df_small.to_csv('../movielens_data/very_small_rating.csv')
