from tensorflow import keras

"""
generative adversial network for data augmentation purposes
"""

class DCGAN(keras.Model):
    def __init__(self, latent_dim, output_dims):
        super(DCGAN, self).__init__()

        # Create the discriminator.
        self.discriminator = keras.Sequential(
            [
                keras.layers.InputLayer((28, 28, output_dims)),
                keras.layers.Conv1D(64, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv1D(32, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv1D(16, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Flatten(),
                keras.layers.Dense(1),
                keras.layers.Sigmoid()
            ],
            name="discriminator",
        )

        # Create the generator.
        self.generator = keras.Sequential(
            [
                keras.layers.InputLayer((latent_dim,)),
                keras.layers.Dense(3*6*128),
                keras.layers.ReLU(),
                keras.layers.Reshape((3, 6, 128)),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1DTranspose(128, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.ReLU(),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1DTranspose(64, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.ReLU(),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1DTranspose(32, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.ReLU(),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1D(16, (4, 4), padding="same", activation="sigmoid"),
                keras.layers.ReLU(),
                keras.layers.BatchNormalization(),
            ],
            name="generator",
        )


        self.latent_dim = latent_dim

        self.discriminator.trainable = False


        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")