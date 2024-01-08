
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from protolearn.model import (
    PrototypeLayer,
    get_rlvq_model,
    init_means_and_values
)
from protolearn.model_wrapper import PrototypeModel
tf.config.set_visible_devices([], 'GPU')


def test_layer():
    model = tf.keras.models.Sequential([
        PrototypeLayer(
            units=3,
            scale=3.0,
            eta=0.0
        ),
        tf.keras.layers.Softmax(),
        tf.keras.layers.Dense(1)
    ])

    datum = np.array([[1., 0.]])
    output = model(datum).numpy()

    assert output.shape[0] == 1
    assert output.shape[1] == 1


def test_get_model():

    model = get_rlvq_model(dim=3, n_prototypes=10, scale=3.0, reg_constant=0.0)
    assert model.layers[1].trainable_weights[0].shape == (10, 3)
    assert model.layers[-1].trainable_weights[0].shape[0] == 10


def test_get_model_means():
    # Test that the means are initialized correctly
    means = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    model = get_rlvq_model(
        dim=3, n_prototypes=3, scale=3.0, reg_constant=0.0,
        means=means
        )
    assert np.all(model.layers[1].trainable_weights[0].numpy() == means)


def test_get_model_values():
    # Test that the values are initialized correctly
    values = np.array([[1.], [2.], [3.]])
    model = get_rlvq_model(
        dim=3, n_prototypes=3, scale=3.0, reg_constant=0.0,
        values=values
        )
    assert np.all(model.layers[-1].trainable_weights[0].numpy() == values)


def test_init_means_and_values():
    # Test that the means and values are initialized correctly
    X = np.array([[1., 0.], [0., 1.]])
    X = np.repeat(X, 10, axis=0)
    y = np.array([2., 3.])
    y = np.repeat(y, 10, axis=0)

    means, values = init_means_and_values(X, y, 2)

    np.testing.assert_almost_equal(means[0][0]+means[0][1], 1.0, decimal=4)
    np.testing.assert_almost_equal(means[1][0]+means[1][1], 1.0, decimal=4)
    np.testing.assert_almost_equal(means[0][0]+means[1][0], 1.0, decimal=4)
    np.testing.assert_almost_equal(means[0][1]+means[1][1], 1.0, decimal=4)

    np.testing.assert_almost_equal(np.min(values), 2.0, decimal=4)
    np.testing.assert_almost_equal(np.max(values), 3.0, decimal=4)

    mv, vv = init_means_and_values(X, y, 2, method="kmeans")
    assert means.shape == mv.shape
    assert values.shape == vv.shape

    mv2, vv2 = init_means_and_values(X, y, 2, method="random_pick")
    assert means.shape == mv2.shape
    assert values.shape == vv2.shape

    # For random pick, if X has 2 points and we pick 2 prototypes,
    # the prototypes and values should be the same

    X2 = np.array([[1., 0.], [0., 1.], [0., 0.]])
    X2 = np.repeat(X2, 10, axis=0)
    y2 = np.array([2., 3., 4.])
    y2 = np.repeat(y2, 10, axis=0)

    mv3, vv3 = init_means_and_values(X2, y2, X2.shape[0], method="random_pick")
    assert np.sum(mv3) == np.sum(X2)
    assert np.sum(vv3) == np.sum(y2)


def test_prototype_layer():
    means = np.array([[1., 0.], [0., 1.], [2., 2.]])

    model = tf.keras.models.Sequential([
        PrototypeLayer(units=3, scale=0.1, eta=0.01,
                       means=means)
    ])

    X = np.array([[1., 0.], [0., 1.]])

    output = model(X).numpy()
    expected00 = -(-np.log(2*np.pi) - 2*np.log(0.1))
    expected01 = -(-np.log(2*np.pi) - 2*np.log(0.1) - 1/(0.1*0.1))
    expected02 = -(-np.log(2*np.pi) - 2*np.log(0.1) - 2.5/(0.1*0.1))

    np.testing.assert_almost_equal(output[0][0], expected00, 2)
    np.testing.assert_almost_equal(output[0][1], expected01, 2)
    np.testing.assert_almost_equal(output[0][2], expected02, 2)


def test_rvlq_model():
    means = np.array([[1., 0.], [0., 1.]])
    vals = np.array([[-1.], [1.]])

    model = get_rlvq_model(
        dim=2, n_prototypes=2, scale=0.1, reg_constant=0.0,
        means=means, values=vals
        )

    X = np.array([[1., 0.], [0., 0.], [0., 1.]])
    output = model(X).numpy()

    np.testing.assert_almost_equal(output[0][0], -1.0, 2)
    np.testing.assert_almost_equal(output[1][0], 0.0, 2)
    np.testing.assert_almost_equal(output[2][0], 1.0, 2)

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.001),
        metrics='mse'
        )
    loss, mse, mean_radii = model.evaluate(X, np.array([[-1.], [0.], [1.]]))

    np.testing.assert_almost_equal(loss, 0.0, 2)
    np.testing.assert_almost_equal(mse, 0.0, 2)

    # Each prototype is on top of a real point, therefore
    # mean_radii should be 0
    np.testing.assert_almost_equal(mean_radii, 0.0, 2)

    # loss = 2^2 + 0 + 2^2 over 3
    loss, mse, mean_radii = model.evaluate(X, np.array([[1], [0], [-1]]))
    np.testing.assert_almost_equal(mse, 8./3, 2)
    np.testing.assert_almost_equal(loss, 8./3, 2)

    # With regularization
    model2 = get_rlvq_model(
        dim=2, n_prototypes=2, scale=0.1, reg_constant=0.5,
        means=means, values=vals
    )
    model2.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.001),
        metrics='mse'
        )
    loss, mse, mean_radii = model2.evaluate(X, np.array([[-1], [0], [1]]))
    np.testing.assert_almost_equal(mse, 0, 2)
    np.testing.assert_almost_equal(mean_radii, 0.0, 2)
    
    # TODO: Add example where mean_radii is not 0


def test_model_wrapper():
    model = PrototypeModel(
        n_prototypes=2, scale=1, reg_constant=0.0,
        learning_rate=0.01, epochs=0, batch_size=256,
        verbose=False, restart=False
    )

    X = np.array([[1., 0.]]*500 + [[0., 1.]]*500)
    y = np.array([[-1.]]*500 + [[1.]]*500)

    X += np.random.normal(0, 0.1, X.shape)
    y += np.random.normal(0, 0.1, y.shape)

    # Try to predict with unfitted model and verify exception
    with pytest.raises(Exception) as e_info:
        preds = model.predict(X)
    assert str(e_info.value) == "Model not fitted yet"

    model.fit(X, y)

    # Zero epochs = kmeans
    prototypes = model.get_prototypes()
    assert prototypes.shape[0] == 2
    assert prototypes.shape[1] == 2

    np.testing.assert_almost_equal(prototypes.sum(), 2.0, 1)

    model = PrototypeModel(
        n_prototypes=2, scale=.1, reg_constant=0.0,
        learning_rate=0.01, epochs=3, batch_size=256,
        verbose=False, restart=True, init_method="random_pick"
    )
    model.fit(X, y)

    # Make sure loss decreases
    losses = model._training_log.history['loss']
    assert losses[0] > losses[-1]

    model = PrototypeModel(
        n_prototypes=2, scale=.1, reg_constant=0.0,
        learning_rate=0.01, epochs=5, batch_size=256,
        verbose=False, restart=True, init_method="kmeans"
    )
    model.fit(X, y)

    # Make sure prototypes are correct
    np.testing.assert_almost_equal(prototypes.sum(), 2.0, 2)
    np.testing.assert_almost_equal(prototypes.sum(axis=0).mean(), 1.0, 2)
    np.testing.assert_almost_equal(prototypes.sum(axis=1).mean(), 1.0, 2)

    # Make sure predictions are correct
    X1 = np.array([[1., 0.]])
    preds = model.predict(X1)
    np.testing.assert_almost_equal(preds[0][0] + 1.0, 0.0, 2)
    X2 = np.array([[0., 1.]])
    preds = model.predict(X2)
    np.testing.assert_almost_equal(preds[0][0] - 1.0, 0.0, 2)

    # Make sure importances are correct
    importances = model.get_importances(X)
    print(importances)
    sign1 = np.sign(importances[0][0] - importances[0][1])
    sign2 = np.sign(importances[-1][0] - importances[-1][1])
    sign3 = np.sign(importances[499][0] - importances[499][1])
    sign4 = np.sign(importances[500][0] - importances[500][1])

    assert sign1 != sign2
    assert sign1 == sign3
    assert sign2 == sign4
    
    # Assert that scale is 0.1
    #np.testing.assert_almost_equal(model.get_scales(), 0.1, 2)
    assert np.abs(model.get_scales()-.1).sum() < 0.01

    # Test learnable scales
    model = PrototypeModel(
        n_prototypes=2, scale=.1, reg_constant=0.0,
        learning_rate=0.1, epochs=10, batch_size=256,
        verbose=False, restart=False, trainable_scales=True
    )

    X += np.random.normal(0, 0.5, X.shape)
    model.fit(X, y)
    print(model._model.summary())
    # Make sure scales are different from 0.1
    print(model.get_scales())
    assert np.abs(model.get_scales()-.1).sum() > 0.01

    assert model.get_prototype_values().shape[0] == model.get_prototypes().shape[0]


def test_init_means_and_values():
    model = PrototypeModel(
        n_prototypes=2, scale=1, reg_constant=0.0,
        learning_rate=0.01, epochs=1, batch_size=256,
        verbose=False, restart=False
    )

    X = np.array([[1., 0.]]*500 + [[0., 1.]]*500)
    y = np.array([[-1.]]*500 + [[1.]]*500)

    X += np.random.normal(0, 0.1, X.shape)
    y += np.random.normal(0, 0.1, y.shape)

    model.fit(X, y)
    init_means = model.get_initial_prototypes()

    model.fit(X, y)
    init_means2 = model.get_initial_prototypes()

    np.testing.assert_almost_equal(init_means, init_means2, 2)
