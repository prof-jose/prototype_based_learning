import json
from protolearn.loaders import Loader
import numpy as np


def test_loader():
    loader = Loader('protolearn/tests/test_config.json')

    assert len(loader._config['attributes']) == 2
    assert len(loader._data.columns) == 4


def test_load_as_dict():
    with open('protolearn/tests/test_config.json') as f:
        config = json.load(f)

    loader = Loader(config)
    assert len(loader._config['attributes']) == 2
    assert len(loader._data.columns) == 4


def test_split():
    loader = Loader('protolearn/tests/test_config.json')
    X_train, X_test, y_train, y_test = loader.get_splits()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(loader._data)
    diff = len(X_train) - len(X_test)
    assert diff == 1 or diff == 0 or diff == -1

    # Test autoscale
    loader = Loader('protolearn/tests/test_config_autoscale.json')
    X_train, X_test, y_train, y_test = loader.get_splits()

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)

    for i in range(0, 2):
        np.testing.assert_almost_equal(x_mean[i], 0., 1)
        np.testing.assert_almost_equal(x_std[i], 1., 1)

    percentile1 = np.percentile(y_train, 5)
    percentile2 = np.percentile(y_train, 95)
    old_shape = y_train.shape[0]

    # Modify config to include percentiles
    loader = Loader('protolearn/tests/test_config_outliers.json')
    X_train, X_test, y_train, y_test = loader.get_splits()

    # Assert that the percentiles are not in the training set
    assert y_train.shape[0] < old_shape
    np.testing.assert_almost_equal(y_train.shape[0]/old_shape, 0.9, 2)
    np.testing.assert_almost_equal(y_train.min(), percentile1, 2)
    np.testing.assert_almost_equal(y_train.max(), percentile2, 2)
    np.testing.assert_almost_equal(y_test.min(), percentile1, 2)
    np.testing.assert_almost_equal(y_test.max(), percentile2, 2)


def test_header():
    loader = Loader('protolearn/tests/test_banknote.json')
    X_train, X_test, y_train, y_test = loader.get_splits()
    assert X_train.shape[1] == 4
    assert np.unique(y_train).shape[0] == 2
    np.testing.assert_almost_equal(y_train.min(), 0., 2)
    np.testing.assert_almost_equal(y_train.max(), 1., 2)

    loader._config['attributes'] = [0, 1]
    X_train, X_test, y_train, y_test = loader.get_splits()
    assert X_train.shape[1] == 2
    np.testing.assert_almost_equal(y_train.min(), 0., 2)
    np.testing.assert_almost_equal(y_train.max(), 1., 2)


def test_source():
    loder = Loader('protolearn/tests/test_config_source.json')
    X_train, X_test, y_train, y_test = loder.get_splits()
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[0] == y_test.shape[0]
