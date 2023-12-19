"""
Data loader class.
"""


import json
import pandas as pd
from sklearn import preprocessing as pre
import os
import numpy as np


class Loader():
    def __init__(self, config, working_dir="", seed=0):
        """
        Data loader class.
        """

        self._working_dir = working_dir
        self._seed = seed
        if type(config) == str:
            with open(config) as f:
                self._config = json.load(f)
        else:
            self._config = config

        header = 0
        if 'header' in self._config:
            if self._config['header'] == "None":
                header = None

        # Make a path concatenating working_dir and source
        path_to_csv = os.path.join(self._working_dir, self._config['source'])

        self._data = pd.read_csv(
            path_to_csv,
            header=header,
            usecols=self._get_necessary_cols()
        )

        target_col = self._config["target"]["column"]
        if 'attributes' not in self._config:
            self._config['attributes'] = self._data.columns.drop(target_col)

        # Filter outliers from target
        if "outliers" in self._config["target"]:
            alpha = self._config["target"]["outliers"]
            value1 = np.percentile(self._data[target_col], alpha)
            value2 = np.percentile(self._data[target_col], 100-alpha)
            condition1 = self._data[target_col] > value1
            condition2 = self._data[target_col] < value2
            self._data = self._data[condition1 & condition2]

        # Perform summary of the data before autoscaling
        print(self._data.describe().transpose()[['min', 'max', 'mean', 'std']])

        # Autoscale if needed
        self._data = self._autoscale(self._data)

    def _autoscale(self, data):
        output = data.copy()
        if 'autoscale' in self._config:
            if self._config['autoscale'] == "True":
                print("Autoscaling data")
                # Apply standard scaling to all columns

                for col in self._config['attributes']:
                    print("*** Scaling column: ", col)
                    output[col] = pre.scale(
                        data[col]
                    )
        return output

    def _get_necessary_cols(self):
        # TODO: Consider the case where attributes are not specified,
        # but target is
        if 'attributes' not in self._config:
            return None
        necessary_cols = self._config['attributes'][:]
        if 'split' in self._config:
            if self._config['split']['type'] == 'sequential':
                necessary_cols.append(self._config['split']['column'])
        if 'target' in self._config:
            print("Target column: ", self._config['target']['column'])
            necessary_cols.append(self._config['target']['column'])
        return necessary_cols

    def get_splits(self):
        if 'split' in self._config:
            # Sequential split
            if self._config['split']['type'] == 'sequential':
                sorted_data = self._data.sort_values(
                    self._config['split']['column']
                    )
            # Random split
            elif self._config['split']['type'] == 'random':
                sorted_data = self._data.sample(
                    frac=1,
                    replace=False,
                    random_state=self._seed
                    )
            # Explicitly indicate test split
            elif self._config['split']['type'] == 'source':
                # TODO: Refactor this
                header = 0
                if 'header' in self._config:
                    header = self._config['header']
                print(self._get_necessary_cols())
                path_to_source = os.path.join(
                    self._working_dir,
                    self._config['split']['source']
                    )
                test_source = pd.read_csv(
                    path_to_source,
                    header=header,
                    usecols=self._get_necessary_cols()
                )
            else:
                raise NotImplementedError(
                    "Unknown split type: ", self._config['split']['type']
                    )
            if self._config['split']['type'] == 'source':
                train_split = self._data
                test_split = self._autoscale(test_source)
            else:
                fraction = self._config['split']['fraction']
                n_items = len(sorted_data)
                n_train = int(n_items * fraction)

                train_split = sorted_data.iloc[:n_train, :]
                test_split = sorted_data.iloc[n_train:, :]

            X_train = train_split[self._config['attributes']]
            X_test = test_split[self._config['attributes']]
            print(train_split.columns)
            y_train = train_split[self._config['target']['column']]
            y_test = test_split[self._config['target']['column']]

            # Apply transformation if exists
            if 'transformation' in self._config['target']:
                # Log
                if self._config['target']['transformation'] == 'log':
                    y_train = np.log(y_train)
                    y_test = np.log(y_test)
                else:
                    raise NotImplementedError(
                        "Unknown transformation: ",
                        self._config['target']['transformation']
                        )

            # Apply normalizer if exists
            if 'normalizer' in self._config['target']:
                y_train = y_train / self._config['target']['normalizer']
                y_test = y_test / self._config['target']['normalizer']

        return X_train, X_test, y_train, y_test
