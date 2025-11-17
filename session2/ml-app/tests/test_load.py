import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from src.data_loader import (
    load_iris_data,
    get_feature_names,
    get_target_names,
    load_iris_as_dataframe,
    get_dataset_info
)


class TestLoadIrisData:
    def test_load_iris_data_default_params(self):
        """Test loading data with default parameters"""
        X_train, X_test, y_train, y_test = load_iris_data()

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == 4
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

    def test_load_iris_data_custom_test_size(self):
        """Test loading data with custom test size"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.3)

        total_samples = X_train.shape[0] + X_test.shape[0]
        test_ratio = X_test.shape[0] / total_samples

        assert 0.25 < test_ratio < 0.35  # Allow some tolerance

    def test_load_iris_data_reproducibility(self):
        """Test that same random_state produces same split"""
        X_train1, X_test1, y_train1, y_test1 = load_iris_data(random_state=42)
        X_train2, X_test2, y_train2, y_test2 = load_iris_data(random_state=42)

        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)

    def test_load_iris_data_different_random_state(self):
        """Test that different random_state produces different split"""
        X_train1, X_test1, y_train1, y_test1 = load_iris_data(random_state=42)
        X_train2, X_test2, y_train2, y_test2 = load_iris_data(random_state=123)

        assert not np.array_equal(X_train1, X_train2)

    def test_load_iris_data_class_distribution(self):
        """Test that stratification maintains class distribution"""
        X_train, X_test, y_train, y_test = load_iris_data()

        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)

        assert len(train_classes) == 3
        assert len(test_classes) == 3
        assert set(train_classes) == set(test_classes)

    def test_load_iris_data_feature_count(self):
        """Test that data has correct number of features"""
        X_train, X_test, y_train, y_test = load_iris_data()

        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4

    def test_load_iris_data_target_range(self):
        """Test that target values are in valid range"""
        X_train, X_test, y_train, y_test = load_iris_data()

        assert all(0 <= y <= 2 for y in y_train)
        assert all(0 <= y <= 2 for y in y_test)


class TestGetFeatureNames:
    def test_get_feature_names_count(self):
        """Test that correct number of feature names returned"""
        feature_names = get_feature_names()
        assert len(feature_names) == 4

    def test_get_feature_names_content(self):
        """Test that feature names contain expected keywords"""
        feature_names = get_feature_names()
        feature_str = ' '.join(feature_names).lower()

        assert 'sepal' in feature_str
        assert 'petal' in feature_str
        assert 'length' in feature_str
        assert 'width' in feature_str

    def test_get_feature_names_type(self):
        """Test that feature names are returned as list of strings"""
        feature_names = get_feature_names()
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)


class TestGetTargetNames:
    def test_get_target_names_count(self):
        """Test that correct number of target names returned"""
        target_names = get_target_names()
        assert len(target_names) == 3

    def test_get_target_names_content(self):
        """Test that target names are correct Iris species"""
        target_names = get_target_names()
        expected_names = ['setosa', 'versicolor', 'virginica']

        assert all(name in expected_names for name in target_names)

    def test_get_target_names_type(self):
        """Test that target names are returned as list or array of strings"""
        target_names = get_target_names()
        assert isinstance(target_names, (list, np.ndarray))
        assert all(isinstance(str(name), str) for name in target_names)


class TestLoadIrisAsDataFrame:
    def test_load_iris_as_dataframe_type(self):
        """Test that DataFrame is returned"""
        df = load_iris_as_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_load_iris_as_dataframe_shape(self):
        """Test DataFrame has correct shape"""
        df = load_iris_as_dataframe()
        assert df.shape[0] == 150
        assert df.shape[1] == 6  # 4 features + target + species

    def test_load_iris_as_dataframe_columns(self):
        """Test DataFrame has expected columns"""
        df = load_iris_as_dataframe()

        assert 'target' in df.columns
        assert 'species' in df.columns
        assert len(df.columns) == 6

    def test_load_iris_as_dataframe_target_range(self):
        """Test target column has valid values"""
        df = load_iris_as_dataframe()

        assert df['target'].min() == 0
        assert df['target'].max() == 2
        assert set(df['target'].unique()) == {0, 1, 2}

    def test_load_iris_as_dataframe_species_column(self):
        """Test species column contains string names"""
        df = load_iris_as_dataframe()

        assert df['species'].dtype == object
        assert len(df['species'].unique()) == 3


class TestGetDatasetInfo:
    def test_get_dataset_info_type(self):
        """Test that dict is returned"""
        info = get_dataset_info()
        assert isinstance(info, dict)

    def test_get_dataset_info_keys(self):
        """Test that all expected keys are present"""
        info = get_dataset_info()
        expected_keys = [
            'feature_names',
            'target_names',
            'n_samples',
            'n_features',
            'n_classes',
            'class_distribution'
        ]

        for key in expected_keys:
            assert key in info

    def test_get_dataset_info_values(self):
        """Test that info values are correct"""
        info = get_dataset_info()

        assert info['n_samples'] == 150
        assert info['n_features'] == 4
        assert info['n_classes'] == 3

    def test_get_dataset_info_class_distribution(self):
        """Test that class distribution is balanced"""
        info = get_dataset_info()
        class_dist = info['class_distribution']

        assert len(class_dist) == 3
        assert all(count == 50 for count in class_dist.values())

    def test_get_dataset_info_feature_names_length(self):
        """Test feature names in info"""
        info = get_dataset_info()
        assert len(info['feature_names']) == 4

    def test_get_dataset_info_target_names_length(self):
        """Test target names in info"""
        info = get_dataset_info()
        assert len(info['target_names']) == 3