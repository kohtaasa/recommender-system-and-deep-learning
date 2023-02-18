import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, Activation, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD

df = pd.read_csv('../movielens_data/edited_rating.csv')

N = df.userId.max() + 1  # number of users
M = df.movieId.max() + 1  # number of movies

# split into train and test
df_train, df_test = train_test_split(df, test_size=0.2)

# initialize variables
K = 10  # latent dimensionality
mu = df_train.rating.mean()
epochs = 25

u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K)(u)  # (N, 1, K)
m_embedding = Embedding(M, K)(m)  # (N, 1, K)
u_embedding = Flatten()(u_embedding)  # (N, K)
m_embedding = Flatten()(m_embedding)  # (N, K)
x = Concatenate()([u_embedding, m_embedding])  # (N, 2K)

# neural network
x = Dense(400)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Dropout(0.5)(x)
x = Dense(1)(x)

model = Model(inputs=[u, m], outputs=x)
model.compile(loss='mse', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['mse'])
r = model.fit(x=[df_train.userId.values, df_train.movie_idx.values],
              y=df_train.rating.values - mu,
              epochs=epochs,
              batch_size=128,
              validation_data=([df_test.userId.values, df_test.movie_idx.values], df_test.rating.values - mu))

# plot losses
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='test loss')
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mse'], label='train mse')
plt.plot(r.history['val_mse'], label='test mse')
plt.legend()
plt.show()
