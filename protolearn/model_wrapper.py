"""
sklearn wrapper of the model.
"""

from .model import init_means_and_values, get_rlvq_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.base import RegressorMixin, BaseEstimator


class PrototypeModel(BaseEstimator, RegressorMixin):

    def __init__(self, n_prototypes=3, scale=0.1, reg_constant=0.001,
                 learning_rate=0.001, epochs=100, batch_size=256,
                 verbose=False, restart=False, validation_data=None,
                 init_method="kmeans", variant="full", trainable_scales=False,
                 projection=None):
        self.n_prototypes = n_prototypes
        self.scale = scale
        self.reg_constant = reg_constant
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.restart = restart
        self._fitted = False
        self.validation_data = validation_data
        self.init_method = init_method
        self.variant = variant
        self.trainable_scales = trainable_scales
        self.projection = projection

    def fit(self, X, y):
        return self._fit_full(X, y)

    def _fit_full(self, X, y):

        if not (self._fitted) or self.restart:

            means, vals = init_means_and_values(
                X, y, self.n_prototypes, self.init_method
                )

            self._initial_means = means
            self._initial_values = vals

            self._model = get_rlvq_model(
                dim=X.shape[1],
                n_prototypes=self.n_prototypes,
                means=means,
                values=vals,
                reg_constant=self.reg_constant,
                scale=self.scale,
                trainable_scales=self.trainable_scales,
                projection=self.projection
                )

            self._model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.learning_rate),
                metrics='mse'
                )

            # Auxiliary model for getting the importances
            self._importance_model = Model(
                inputs=self._model.input,
                outputs=self._model.layers[-2].output
            )

        self._training_log = self._model.fit(
            X, y,
            epochs=self.epochs,
            verbose=self.verbose,
            batch_size=self.batch_size,
            validation_data=self.validation_data
        )
        self._fitted = True

        return self

    def predict(self, X):
        if not self._fitted:
            raise Exception("Model not fitted yet")
        return self._model.predict(X)

    def get_prototypes(self):
        return self._model.layers[1].weights[0].numpy()

    def get_scales(self):
        return self._model.layers[1].weights[1].numpy()

    def get_initial_prototypes(self):
        return self._initial_means

    def get_prototype_values(self):
        return self._model.layers[-1].weights[0].numpy().flatten()
    
    def get_importances(self, X):
        return self._importance_model.predict(X)
