#After Training for visualizing output

from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
 
def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
def save_plot(examples, n):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()
 
model = load_model('generator_model_100.h5')
latent_points = generate_latent_points(100, 25)
X = model.predict(latent_points)
save_plot(X, 5)
