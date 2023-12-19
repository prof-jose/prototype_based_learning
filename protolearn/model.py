"""
Model implementation for prototype-based learning.
"""


import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class PrototypeLayer(tf.keras.layers.Layer):

    def __init__(
            self, units=3, scale=1.0, eta=1e-4, means=None,
            trainable_scales=False
            ):
        super().__init__()
        self._units = units
        self._eta = eta
        self._scale = scale
        self._means_init = means
        self._trainable_scales = trainable_scales

    def build(self, input_shape):
        self._means = self.add_weight(
                shape=(self._units, input_shape[-1]),
                initializer="random_normal",
                trainable=True,
            )
        self._dim = self._means.shape[-1]

        if self._means_init is not None:
            self.weights[0].assign(self._means_init)

        self._scales = self.add_weight(
                shape=(self._units, input_shape[-1]),
                initializer=tf.keras.initializers.Constant(self._scale),
                trainable=self._trainable_scales,
            )

        self._prototype_dist = tfd.MultivariateNormalDiag(
            loc=self._means,
            scale_diag=self._scales
        )

    def call(self, X):

        distances = tf.norm(X[:, None, :] - self._means, axis=-1)
        avg_min_distance = tf.reduce_mean(tf.reduce_min(distances, axis=0))
        self.add_metric(avg_min_distance, name="mean_of_radii")
        self.add_loss(self._eta*avg_min_distance)
        return -self._prototype_dist.log_prob(X[:, None, :])


def get_rlvq_model(dim=2, n_prototypes=3, scale=0.1, reg_constant=0.001,
                   means=None, values=None, activation="linear", trainable_scales=False,
                   projection=None):

    input = tf.keras.layers.Input(shape=dim)

    # projection is an array specifying the number of units in each layer

#    if projection is not None:
#        h = tf.keras.layers.Dense(
#            projection[0], activation="relu"
#            )(input)
#        for i in range(1, len(projection)):
#            h = tf.keras.layers.Dense(
#                projection[i], activation="relu"
#                )(h)
#        loglike = PrototypeLayer(units=n_prototypes, scale=scale, eta=reg_constant,
#                             means=means, trainable_scales=trainable_scales)(h)

#    else:
#        loglike = PrototypeLayer(units=n_prototypes, scale=scale, eta=reg_constant,
#                             means=means, trainable_scales=trainable_scales)(input)

    loglike = PrototypeLayer(units=n_prototypes, scale=scale, eta=reg_constant,
                             means=means, trainable_scales=trainable_scales)(input)
    probs = tf.keras.layers.Softmax(name="softmax")(-loglike)
    if values is None:
        regression = tf.keras.layers.Dense(
            1, use_bias=False, name="values"
            )(probs)
    else:
        regression = tf.keras.layers.Dense(
            1, use_bias=False, name="values", weights=[values],
            activation=activation
            )(probs)

    model = tf.keras.models.Model(inputs=input, outputs=regression)

    return model


def init_means_and_values(X, y, n_prototypes, method="kmeans"):
    # Initialize the means and values

    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_prototypes, n_init="auto")
        kmeans.fit(X)
        means = kmeans.cluster_centers_
        values = np.zeros((n_prototypes, 1))
        for i in range(n_prototypes):
            values[i] = np.mean(y[kmeans.labels_ == i])
    elif method == "random_pick":
        means = np.zeros((n_prototypes, X.shape[-1]))
        values = np.zeros((n_prototypes, 1))
        idx = np.random.choice(X.shape[0], n_prototypes, replace=False)
        for i in range(n_prototypes):
            print(idx[i])
            print(X[idx[i], :])
            means[i, :] = X[idx[i], :]
            values[i] = y[idx[i]]
    else:
        raise NotImplementedError("Unknown method: ", method)
    return means, values
