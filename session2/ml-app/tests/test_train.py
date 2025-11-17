import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import patch, MagicMock
from src.train import main


class TestTrainScript:
    def test_main_executes_without_error(self, tmp_path):
        """Test that main function executes without errors"""
        # Change to temp directory for model saving
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        os.makedirs('models', exist_ok=True)

        try:
            main()
            assert os.path.exists('models/iris_classifier.pkl')
        finally:
            os.chdir(original_dir)

    @patch('src.train.load_iris_data')
    def test_main_loads_data(self, mock_load_data):
        """Test that main function calls load_iris_data"""
        import numpy as np

        mock_load_data.return_value = (
            np.random.rand(100, 4),
            np.random.rand(50, 4),
            np.random.randint(0, 3, 100),
            np.random.randint(0, 3, 50)
        )

        with patch('src.train.IrisClassifier'):
            with patch('src.train.plot_confusion_matrix'):
                with patch('src.train.plot_feature_importance'):
                    try:
                        main()
                    except:
                        pass

                    mock_load_data.assert_called_once()

    @patch('src.train.IrisClassifier')
    @patch('src.train.load_iris_data')
    def test_main_trains_model(self, mock_load_data, mock_classifier):
        """Test that main function trains the classifier"""
        import numpy as np

        mock_load_data.return_value = (
            np.random.rand(100, 4),
            np.random.rand(50, 4),
            np.random.randint(0, 3, 100),
            np.random.randint(0, 3, 50)
        )

        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = (0.95, "Report")
        mock_instance.predict.return_value = np.array([0, 1, 2])
        mock_classifier.return_value = mock_instance

        with patch('src.train.plot_confusion_matrix'):
            with patch('src.train.plot_feature_importance'):
                try:
                    main()
                except:
                    pass

                mock_instance.train.assert_called_once()

    @patch('src.train.IrisClassifier')
    @patch('src.train.load_iris_data')
    def test_main_evaluates_model(self, mock_load_data, mock_classifier):
        """Test that main function evaluates the model"""
        import numpy as np

        mock_load_data.return_value = (
            np.random.rand(100, 4),
            np.random.rand(50, 4),
            np.random.randint(0, 3, 100),
            np.random.randint(0, 3, 50)
        )

        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = (0.95, "Report")
        mock_instance.predict.return_value = np.array([0, 1, 2])
        mock_classifier.return_value = mock_instance

        with patch('src.train.plot_confusion_matrix'):
            with patch('src.train.plot_feature_importance'):
                try:
                    main()
                except:
                    pass

                mock_instance.evaluate.assert_called_once()

    @patch('src.train.IrisClassifier')
    @patch('src.train.load_iris_data')
    def test_main_saves_model(self, mock_load_data, mock_classifier):
        """Test that main function saves the model"""
        import numpy as np

        mock_load_data.return_value = (
            np.random.rand(100, 4),
            np.random.rand(50, 4),
            np.random.randint(0, 3, 100),
            np.random.randint(0, 3, 50)
        )

        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = (0.95, "Report")
        mock_instance.predict.return_value = np.array([0, 1, 2])
        mock_classifier.return_value = mock_instance

        with patch('src.train.plot_confusion_matrix'):
            with patch('src.train.plot_feature_importance'):
                try:
                    main()
                except:
                    pass

                mock_instance.save_model.assert_called_once()

    @patch('src.train.plot_confusion_matrix')
    @patch('src.train.plot_feature_importance')
    @patch('src.train.IrisClassifier')
    @patch('src.train.load_iris_data')
    def test_main_generates_plots(self, mock_load_data, mock_classifier,
                                  mock_feature_plot, mock_confusion_plot):
        """Test that main function generates visualization plots"""
        import numpy as np

        mock_load_data.return_value = (
            np.random.rand(100, 4),
            np.random.rand(50, 4),
            np.random.randint(0, 3, 100),
            np.random.randint(0, 3, 50)
        )

        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = (0.95, "Report")
        mock_instance.predict.return_value = np.array([0, 1, 2])
        mock_classifier.return_value = mock_instance

        try:
            main()
        except:
            pass

        mock_confusion_plot.assert_called_once()
        mock_feature_plot.assert_called_once()