import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# load in the data
df = pd.read_csv('../movielens_data/very_small_rating.csv')

N = df.userId.max() + 1  # number of users
M = df.movie_idx.max() + 1  # number of movies

# split into train and test
df_train, df_test = train_test_split(df, test_size=0.2)

# a dictionary to tell us which users have rated which movies
user2movie = {}
# a dictionary to tell us which movies have been rated by which users
movie2user = {}
# a dictionary to look up ratings
usermovie2rating = {}
print('Calling: update_user2movie_and_movie2user')
count = 0


def update_user2movie_and_movie2user(row):
    global count
    count += 1
    if count % 100000 == 0:
        print('Processed: %.3f' % (float(count)/len(df_train)))

    i = int(row.userId)
    j = int(row.movie_idx)
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)

    usermovie2rating[(i, j)] = row.rating


df_train.apply(update_user2movie_and_movie2user, axis=1)

# test rating dictionary
usermovie2rating_test = {}
print("Calling: update_usermovie2rating_test")
count = 0


def update_usermovie2rating_test(row):
    global count
    count += 1
    if count % 100000 == 0:
        print('Processed: %.3f' % (float(count) / len(df_test)))

    i = int(row.userId)
    j = int(row.movie_idx)
    usermovie2rating_test[(i, j)] = row.rating


df_test.apply(update_usermovie2rating_test, axis=1)


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


pickle_dump(user2movie, 'user2movie.pkl')
pickle_dump(movie2user, 'movie2user.pkl')
pickle_dump(usermovie2rating, 'usermovie2rating.pkl')
pickle_dump(usermovie2rating_test, 'usermovie2rating_test.pkl')
