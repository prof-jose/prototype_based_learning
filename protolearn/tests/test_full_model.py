import numpy as np
from protolearn.model_wrapper import PrototypeFullModel
import tensorflow as tf
import numpy as np
from numpy.random import seed
import random as ra


def test_full_model():

    # Set seeds for reproducibility
    seed(1)
    tf.random.set_seed(2)
    ra.seed(1)

    # Create a dataset with 2 classes and 3 dimensions
    X = np.array([[1., 0., 0.], [0., 1., 0], [0., 0., 1.]])
    X = np.repeat(X, 100, axis=0)
    X += np.random.normal(0, 0.05, X.shape)
    y = np.array([0., 1., 2.])
    y = np.repeat(y, 100, axis=0)

    # Create an embedding network
    embedding_network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2, input_shape=(3,), activation="relu"),
        tf.keras.layers.Dense(2)
    ])

    # Create a model with 3 prototypes
    model = PrototypeFullModel(
        n_prototypes=3, scale=.1, reg_constant=0.0,
        learning_rate=0.0001, epochs=10, batch_size=50,
        verbose=False, restart=True, init_method="kmeans",
        network=embedding_network
    )

    model.fit(X, y)

    # Test predictions: the first sample should have a prediction closer to 0 than to 1 or 2
    X1 = np.array([[1., 0., 0.]])
    preds = model.predict(X1)
    assert np.abs(preds[0][0] - 0) < np.abs(preds[0][0] - 1)
    assert np.abs(preds[0][0] - 0) < np.abs(preds[0][0] - 2)

    # The second sample should have a prediction closer to 1 than to 0 or 2
    X2 = np.array([[0., 1., 0.]])
    preds = model.predict(X2)
    assert np.abs(preds[0][0] - 1) < np.abs(preds[0][0] - 0)
    assert np.abs(preds[0][0] - 1) < np.abs(preds[0][0] - 2)

    # The third sample should have a prediction closer to 2 than to 0 or 1
    X3 = np.array([[0., 0., 1.]])
    preds = model.predict(X3)
    assert np.abs(preds[0][0] - 2) < np.abs(preds[0][0] - 0)
    assert np.abs(preds[0][0] - 2) < np.abs(preds[0][0] - 1)

    # The size of the prototypes should be 3x2
    assert model.get_prototypes().shape == (3, 2)

    # The size of the prototype values should be 3
    assert model.get_prototype_values().shape == (3,)

    assert model.get_scales().shape == (3, 2)
