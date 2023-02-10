import pandas as pd

# load in the data
df = pd.read_csv('../movielens_data/rating.csv')

# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
for idx, movie_id in enumerate(unique_movie_ids):
    movie2idx[movie_id] = idx

# add them to the data frame
df['movie_idx'] = df.movieId.map(movie2idx)
df = df.drop(columns=['timestamp'])
print()
df.to_csv('../movielens_data/edited_rating.csv', index=False)



