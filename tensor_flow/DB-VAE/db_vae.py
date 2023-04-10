
"""
MIT 6.S191, lab2
https://github.com/aamini/introtodeeplearning/blob/master/lab2/Part2_Debiasing.ipynb
http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf
"""

import IPython
import tensorflow as tf
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mitdeeplearning as mdl


def make_standard_classifier(n_outputs=1, n_filters=12):
    """Function to define a standard CNN model
    :param n_outputs: the number of units in the last layer
    :param n_filters: base number of convolutional filters
    """
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu')

    model = tf.keras.Sequential([
        Conv2D(filters=1 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=2 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=4 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Conv2D(filters=6 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Flatten(),
        Dense(512),
        Dense(n_outputs, activation=None),
    ])
    return model


@tf.function
def standard_train_step(x, y):
    with tf.GradientTape() as tape:
        # feed the images into the model
        logits = standard_classifier(x)
        # Compute the loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

    # Backpropagation
    grads = tape.gradient(loss, standard_classifier.trainable_variables)
    optimizer.apply_gradients(zip(grads, standard_classifier.trainable_variables))
    return loss


def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005):
    """ Function to calculate VAE loss given:
          an input x,
          reconstructed output x_recon,
          encoded means mu,
          encoded log of standard deviation logsigma,
          weight parameter for the latent loss kl_weight
    """
    # Define the latent loss. Note this is given in the equation for L_{KL} in the text block, and measures how closely
    # the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence.
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1.0 - logsigma, axis=1)

    # Define the reconstruction loss as the mean absolute pixel-wise
    # difference between the input and reconstruction. Hint: you'll need to
    # use tf.reduce_mean, and supply an axis argument which specifies which
    # dimensions to reduce over. For example, reconstruction loss needs to average
    # over the height, width, and channel image dimensions.
    # https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
    reconstruction_loss = tf.reduce_mean(tf.abs(tf.math.subtract(x, x_recon)), axis=(1, 2, 3))

    # Define the VAE loss. Note this is given in the equation for L_{VAE}
    # in the text block directly above
    vae_loss = kl_weight * latent_loss + reconstruction_loss

    return vae_loss


def sampling(z_mean, z_logsigma):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
    # Returns
        z (tensor): sampled latent vector
    """
    # By default, random.normal is "standard" (ie. mean=0 and std=1.0)
    batch, latent_dim = z_mean.shape  # 32 x 100
    epsilon = tf.random.normal(shape=(batch, latent_dim))

    # Define the reparameterization computation!
    # Note the equation is given in the text block immediately above.
    z = z_mean + tf.exp(0.5 * z_logsigma) * epsilon  # changed a little from the center of each sample
    return z


def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    """Loss function for DB-VAE.
    # Arguments
        x: true input x
        x_pred: reconstructed x
        y: true label (face or not face)
        y_logit: predicted labels
        mu: mean of latent distribution (Q(z|X))
        logsigma: log of standard deviation of latent distribution (Q(z|X))
    # Returns
        total_loss: DB-VAE total loss
        classification_loss = DB-VAE classification loss
    """
    # call the relevant function to obtain VAE loss
    vae_loss = vae_loss_function(x=x, x_recon=x_pred, mu=mu, logsigma=logsigma)

    # define the classification loss using sigmoid_cross_entropy
    # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logit)

    # Use the training data labels to create variable face_indicator:
    #   indicator that reflects which training data are images of faces
    face_indicator = tf.cast(tf.equal(y, 1), tf.float32)

    # define the DB-VAE total loss! Use tf.reduce_mean to average over all
    # samples
    total_loss = tf.reduce_mean(classification_loss + face_indicator * vae_loss)

    return total_loss, classification_loss


class DB_VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim

        # Define the number of outputs for the encoder. Recall that we have
        # `latent_dim` latent variables, as well as a supervised output for the
        # classification (index 0).
        num_encoder_dims = 2 * self.latent_dim + 1  # 2 * 100 + 1

        self.encoder = make_standard_classifier(num_encoder_dims)
        self.decoder = make_face_decoder_network()

    # function to feed images into encoder, encode the latent space, and output
    #   classification probability
    def encode(self, x):
        # encoder output
        encoder_output = self.encoder(x)

        # classification prediction
        y_logit = tf.expand_dims(encoder_output[:, 0], -1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1:self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1:]

        return y_logit, z_mean, z_logsigma

    # VAE reparameterization: given a mean and logsigma, sample latent variables
    def reparameterize(self, z_mean, z_logsigma):
        # call the sampling function defined above
        z = sampling(z_mean=z_mean, z_logsigma=z_logsigma)
        return z

    # Decode the latent space and output reconstruction
    def decode(self, z):
        # use the decoder to output the reconstruction
        reconstruction = self.decoder(z)  # how to pass z???
        return reconstruction

    # The call function will be used to pass inputs x through the core VAE
    def call(self, x):
        # Encode input to a prediction and latent space
        y_logit, z_mean, z_logsigma = self.encode(x)

        # reparameterization
        z = self.reparameterize(z_mean=z_mean, z_logsigma=z_logsigma)

        # reconstruction
        recon = self.decode(z)
        return y_logit, z_mean, z_logsigma, recon

    # Predict face or not face logit for given input x
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)