import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense



def Discriminator(n_inputs=2):

    model =Sequential()
    model.add(Dense(25,activation='relu',kernel_initializer='he_uniform',input_dim=n_inputs))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model



def Generator(latent_dim,n_outputs=2):

    model =Sequential()
    model.add(Dense(15,activation='relu',kernel_initializer='he_uniform',input_dim=latent_dim))
    model.add(Dense(n_outputs,activation='linear'))
    return model


def GAN(generator,discriminator):
    
    discriminator.trainable=False

    model=Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model


def real_samples(n):


    X1 = 2*np.pi*rand(n)
    X2 = np.sin(X1)
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    y = np.ones((n, 1))
    return X, y

def latent_points(latent_dim, n):
    
    x_input = np.pi+randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

def fake_samples(generator, latent_dim, n):

    x_input = latent_points(latent_dim, n)
    X = generator.predict(x_input)
    y = np.zeros((n, 1))
    return X, y


def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):

    x_real, y_real = real_samples(n)
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    
    x_fake, y_fake = fake_samples(generator, latent_dim, n)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    
    print(epoch, acc_real, acc_fake)
    plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    plt.legend(['Real Samples','Fake Samples'])
    plt.xlim([0,2*np.pi])
    plt.ylim([-2,2])
    plt.show()


def train(g_model, d_model, gan_model, latent_dim, n_epochs=50000, n_batch=128,
          n_eval=2000):
    half_batch = int(n_batch / 2)
    
    for i in range(n_epochs):
        
        x_real, y_real = real_samples(half_batch)
        x_fake, y_fake = fake_samples(g_model, latent_dim, half_batch)
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        x_gan = latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        
        gan_model.train_on_batch(x_gan, y_gan)
        if (i+1) % n_eval == 0:
                summarize_performance(i, g_model, d_model, latent_dim)


latent_dim = 5

discriminator = Discriminator()
generator = Generator(latent_dim)
gan_model = GAN(generator, discriminator)
train(generator, discriminator, gan_model, latent_dim)



















    
