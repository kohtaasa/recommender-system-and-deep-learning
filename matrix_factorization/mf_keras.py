import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

df = pd.read_csv('../movielens_data/edited_rating.csv')

N = df.userId.max() + 1  # number of users
M = df.movieId.max() + 1  # number of movies

# split into train and test
df_train, df_test = train_test_split(df, test_size=0.2)

# initialize variables
K = 10  # latent dimensionality
mu = df_train.rating.mean()
epochs = 25
reg = 0.  # regularization penalty

# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u)  # (N, 1, K)
m_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(m)  # (N, 1, K)

u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u)  # (N, 1, 1)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m)  # (N, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding])  # (N, 1, 1)

x = Add()([x, u_bias, m_bias])
x = Flatten()(x)  # (N, 1)

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
plt.plot(r.hisotry['mean_squared_error'], label='train mse')
plt.plot(r.history['val_mean_squared_error'], label='test mse')
plt.legend()
plt.show()
