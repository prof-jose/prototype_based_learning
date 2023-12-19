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
                 init_method="kmeans", trainable_scales=False):
        """
        Implementation of prototype-based model as a sklearn-like class.

        Parameters
        ----------
        n_prototypes : int
            Number of prototypes to use.
        scale : float
            Scale of the multivariate normal distribution.
        reg_constant : float
            Regularization constant.
        learning_rate : float
            Learning rate for the Adam optimizer.
        epochs : int
            Number of epochs to train.
        batch_size : int
            Batch size for training.
        verbose : bool
            Whether to print training progress.
        restart : bool
            Whether each call to fit should restart the training or continue
            from the previous state.
        validation_data : tuple
            Validation data to use during training.
        init_method : str
            Method to use for initializing the prototypes.
            Either "kmeans" or "random_pick".
        trainable_scales : bool
            Whether to train the scales.
            If True, the value given in scale is the initializer.
        """

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
        self.trainable_scales = trainable_scales

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Target values.

        Returns
        -------
        self : PrototypeModel
            A reference to the fitted model.
        """
        return self._fit_full(X, y)

    def _fit_full(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Target values.

        Returns
        -------
        self : PrototypeModel
            A reference to the fitted model.
        """

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
                trainable_scales=self.trainable_scales
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
        """
        Predict the target values for the given data.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        y : array-like
            Predicted target values.
        """
        if not self._fitted:
            raise Exception("Model not fitted yet")
        return self._model.predict(X)

    def get_prototypes(self):
        """Get the prototypes of the model."""
        return self._model.layers[1].weights[0].numpy()

    def get_scales(self):
        """Get the scales of the model."""
        return self._model.layers[1].weights[1].numpy()

    def get_initial_prototypes(self):
        """Get the initial prototypes of the model."""
        return self._initial_means

    def get_prototype_values(self):
        """Get the prototype values of the model."""""
        return self._model.layers[-1].weights[0].numpy().flatten()

    def get_importances(self, X):
        """
        Get the importances of the prototypes for the given data.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        importances : array-like
            Importance of each prototype for each data point.
        """
        return self._importance_model.predict(X)
