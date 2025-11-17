import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from src.predict import main


class TestPredictScript:
    @patch('src.predict.IrisClassifier')
    @patch('src.predict.get_target_names')
    def test_main_loads_model(self, mock_get_names, mock_classifier):
        """Test that main function attempts to load the model"""
        mock_instance = MagicMock()
        mock_classifier.return_value = mock_instance
        mock_get_names.return_value = ['setosa', 'versicolor', 'virginica']

        mock_instance.predict.return_value = np.array([0])
        mock_instance.model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])

        try:
            main()
        except:
            pass

        mock_instance.load_model.assert_called_once_with('models/iris_classifier.pkl')

    @patch('builtins.print')
    @patch('src.predict.IrisClassifier')
    def test_main_handles_missing_model(self, mock_classifier, mock_print):
        """Test that main function handles missing model file gracefully"""
        mock_instance = MagicMock()
        mock_instance.load_model.side_effect = FileNotFoundError()
        mock_classifier.return_value = mock_instance

        main()

        # Check that error message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('not found' in call.lower() for call in print_calls)

    @patch('src.predict.IrisClassifier')
    @patch('src.predict.get_target_names')
    def test_main_makes_predictions(self, mock_get_names, mock_classifier):
        """Test that main function makes predictions on example data"""
        mock_instance = MagicMock()
        mock_classifier.return_value = mock_instance
        mock_get_names.return_value = ['setosa', 'versicolor', 'virginica']

        mock_instance.predict.return_value = np.array([0])
        mock_instance.model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])

        try:
            main()
        except:
            pass

        # Should be called 3 times for 3 examples
        assert mock_instance.predict.call_count == 3

    @patch('src.predict.IrisClassifier')
    @patch('src.predict.get_target_names')
    def test_main_uses_correct_examples(self, mock_get_names, mock_classifier):
        """Test that main function uses the correct example data"""
        mock_instance = MagicMock()
        mock_classifier.return_value = mock_instance
        mock_get_names.return_value = ['setosa', 'versicolor', 'virginica']

        mock_instance.predict.return_value = np.array([0])
        mock_instance.model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])

        try:
            main()
        except:
            pass

        # Check first call has setosa-like features
        first_call = mock_instance.predict.call_args_list[0]
        first_example = first_call[0][0][0]

        assert len(first_example) == 4
        assert all(isinstance(x, (int, float)) for x in first_example)

    @patch('builtins.print')
    @patch('src.predict.IrisClassifier')
    @patch('src.predict.get_target_names')
    def test_main_prints_probabilities(self, mock_get_names, mock_classifier, mock_print):
        """Test that main function prints prediction probabilities"""
        mock_instance = MagicMock()
        mock_classifier.return_value = mock_instance
        mock_get_names.return_value = ['setosa', 'versicolor', 'virginica']

        mock_instance.predict.return_value = np.array([0])
        mock_instance.model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])

        main()

        # Check that probabilities were printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Probabilities' in call or 'prob' in call.lower() for call in print_calls)

    @patch('src.predict.get_target_names')
    @patch('src.predict.IrisClassifier')
    def test_main_gets_target_names(self, mock_classifier, mock_get_names):
        """Test that main function retrieves target names"""
        mock_instance = MagicMock()
        mock_classifier.return_value = mock_instance
        mock_get_names.return_value = ['setosa', 'versicolor', 'virginica']

        mock_instance.predict.return_value = np.array([0])
        mock_instance.model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])

        try:
            main()
        except:
            pass

        mock_get_names.assert_called_once()

    @patch('builtins.print')
    @patch('src.predict.IrisClassifier')
    @patch('src.predict.get_target_names')
    def test_main_outputs_predictions_for_all_examples(self, mock_get_names,
                                                       mock_classifier, mock_print):
        """Test that predictions are output for all 3 examples"""
        mock_instance = MagicMock()
        mock_classifier.return_value = mock_instance
        mock_get_names.return_value = ['setosa', 'versicolor', 'virginica']

        mock_instance.predict.return_value = np.array([0])
        mock_instance.model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])

        main()

        # Check that all 3 examples were printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        example_calls = [call for call in print_calls if 'Example' in call]

        assert len(example_calls) >= 3