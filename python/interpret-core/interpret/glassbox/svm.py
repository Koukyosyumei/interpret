from abc import abstractmethod

import colorlover as cl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
from sklearn.svm import SVC as SKSVMClassifier
from sklearn.svm import SVR as SKSVMRegressor

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils import (
    gen_global_selector,
    gen_local_selector,
    gen_name_from_class,
    gen_perf_dicts,
    hist_per_column,
    unify_data,
)


class SVM2DExplanation(FeatureValueExplanation):
    """ Visualizes specifically for SVM methods. """

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        """Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """

        super(SVM2DExplanation, self).__init__(
            explanation_type,
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from ..visual.plot import svm_boundary_plot

        data_dict = self._internal_obj["specific"]
        if self.explanation_type == "global":
            figure = svm_boundary_plot(
                data_dict["X"],
                data_dict["y"],
                data_dict["z"],
                data_dict["xx"],
                data_dict["yy"],
                data_dict["h"],
                0.5,
            )

            return figure

        return super().visualize(key)


class LinearSVMExplanation(FeatureValueExplanation):
    """ Visualizes specifically for SVM methods. """

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        """Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """

        super(LinearSVMExplanation, self).__init__(
            explanation_type,
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from ..visual.plot import (
            get_explanation_index,
            get_sort_indexes,
            mli_plot_horizontal_bar,
            mli_sort_take,
            plot_horizontal_bar,
            sort_take,
        )

        if isinstance(key, tuple) and len(key) == 2:
            provider, key = key
            if (
                "mli" == provider
                and "mli" in self.data(-1)
                and self.explanation_type == "global"
            ):
                explanation_list = self.data(-1)["mli"]
                explanation_index = get_explanation_index(
                    explanation_list, "global_feature_importance"
                )
                scores = explanation_list[explanation_index]["value"]["scores"]
                sort_indexes = get_sort_indexes(
                    scores, sort_fn=lambda x: -abs(x), top_n=15
                )
                sorted_scores = mli_sort_take(
                    scores, sort_indexes, reverse_results=True
                )
                sorted_names = mli_sort_take(
                    self.feature_names, sort_indexes, reverse_results=True
                )
                return mli_plot_horizontal_bar(
                    sorted_scores,
                    sorted_names,
                    title="Overall Importance:<br>Coefficients",
                )
            else:  # pragma: no cover
                raise RuntimeError("Visual provider {} not supported".format(provider))
        else:
            data_dict = self.data(key)
            if data_dict is None:
                return None

            if self.explanation_type == "global" and key is None:
                data_dict = sort_take(
                    data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
                )
                figure = plot_horizontal_bar(
                    data_dict, title="Overall Importance:<br>Coefficients"
                )
                return figure

        return super().visualize(key)


class BaseSVM:

    available_explanations = ["local", "global"]
    explainer_type = "model"

    def __init__(
        self,
        feature_names=None,
        feature_types=None,
        svm_class=SKSVMClassifier,
        **kwargs,
    ):
        """Initializes class.
        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            svm_class: A scikit-learn svm class.
            **kwargs: Kwargs pass to linear class at initialization time.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.svm_class = svm_class
        self.kwargs = kwargs

    @abstractmethod
    def _model(self):
        # This method should be overridden.
        return None

    def fit(self, X, y):
        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )
        self.X = X
        self.y = y

        sk_model_ = self._model()
        sk_model_.fit(X, y)

        self.X_mins_ = np.min(X, axis=0)
        self.X_maxs_ = np.max(X, axis=0)
        self.categorical_uniq_ = {}

        for i, feature_type in enumerate(self.feature_types):
            if feature_type == "categorical":
                self.categorical_uniq_[i] = list(sorted(set(X[:, i])))

        self.global_selector = gen_global_selector(
            X, self.feature_names, self.feature_types, None
        )
        self.bin_counts_, self.bin_edges_ = hist_per_column(X, self.feature_types)

        return self

    def predict(self, X):
        """Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        return self._model().predict(X)

    def explain_local(self, X, y=None, name=None):
        """Provides local explanations for provided instances.
        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.
        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        if name is None:
            name = gen_name_from_class(self)
        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        sk_model_ = self._model()

        is_classification = is_classifier(self)
        if is_classification:
            predictions = self.predict_proba(X)[:, 1]
            intercept = sk_model_.intercept_[0]
            coef = sk_model_.coef_[0]
        else:
            predictions = self.predict(X)
            intercept = sk_model_.intercept_
            coef = sk_model_.coef_

        data_dicts = []
        scores_list = []
        perf_list = []
        perf_dicts = gen_perf_dicts(predictions, y, is_classification)
        for i, instance in enumerate(X):
            scores = list(coef * instance)
            scores_list.append(scores)
            data_dict = {}
            data_dict["data_type"] = "univariate"

            # Performance related (conditional)
            perf_dict_obj = None if perf_dicts is None else perf_dicts[i]
            data_dict["perf"] = perf_dict_obj
            perf_list.append(perf_dict_obj)

            # Names/scores
            data_dict["names"] = self.feature_names
            data_dict["scores"] = scores

            # Values
            data_dict["values"] = instance

            data_dict["extra"] = {
                "names": ["Intercept"],
                "scores": [intercept],
                "values": [1],
            }
            data_dicts.append(data_dict)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "local_feature_importance",
                    "value": {
                        "scores": scores_list,
                        "intercept": intercept,
                        "perf": perf_list,
                    },
                }
            ],
        }
        internal_obj["mli"].append(
            {
                "explanation_type": "evaluation_dataset",
                "value": {"dataset_x": X, "dataset_y": y},
            }
        )

        selector = gen_local_selector(data_dicts, is_classification=is_classification)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )

    def _explain_global_2d(self, name=None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        if name is None:
            name = gen_name_from_class(self)

        h = 0.3  # (self.X.max() - self.X.min()) / 50

        x_min = self.X[:, 0].min() - 0.5
        x_max = self.X[:, 0].max() + 0.5
        y_min = self.X[:, 1].min() - 0.5
        y_max = self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        sk_model_ = self._model()
        if hasattr(sk_model_, "decision_function"):
            Z = sk_model_.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = sk_model_.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        data_dict = {"X": self.X, "y": self.y, "xx": xx, "yy": yy, "z": Z, "h": h}
        internal_obj = {
            "overall": None,
            "specific": data_dict,
        }
        return SVM2DExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )

    def _explain_global_linear(self, name=None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        if name is None:
            name = gen_name_from_class(self)

        sk_model_ = self._model()
        if is_classifier(self):
            intercept = sk_model_.intercept_[0]
            coef = sk_model_.coef_[0]
        else:
            intercept = sk_model_.intercept_
            coef = sk_model_.coef_

        overall_data_dict = {
            "names": self.feature_names,
            "scores": list(coef),
            "extra": {"names": ["Intercept"], "scores": [intercept]},
        }

        specific_data_dicts = []
        for index, feature in enumerate(self.feature_names):
            feat_min = self.X_mins_[index]
            feat_max = self.X_maxs_[index]
            feat_coef = coef[index]

            feat_type = self.feature_types[index]

            if feat_type == "continuous":
                # Generate x, y points to plot from coef for continuous features
                grid_points = np.linspace(feat_min, feat_max, 30)
            else:
                grid_points = np.array(self.categorical_uniq_[index])

            y_scores = feat_coef * grid_points

            data_dict = {
                "names": grid_points,
                "scores": y_scores,
                "density": {
                    "scores": self.bin_counts_[index],
                    "names": self.bin_edges_[index],
                },
            }

            specific_data_dicts.append(data_dict)

        internal_obj = {
            "overall": overall_data_dict,
            "specific": specific_data_dicts,
            "mli": [
                {
                    "explanation_type": "global_feature_importance",
                    "value": {"scores": list(coef), "intercept": intercept},
                }
            ],
        }
        return LinearSVMExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )

    def explain_global(self, name=None, mode="coef"):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.
            mode: coef or boundary

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        if mode == "coef":
            return self._explain_global_linear(name=name)
        elif mode == "boundary":
            return self._explain_global_2d(name=name)


class SVMRegressor(BaseSVM, RegressorMixin, ExplainerMixin):
    """SVM regression.

    Currently wrapper around svm models in scikit-learn: https://github.com/scikit-learn/scikit-learn
    """

    def __init__(
        self, feature_names=None, feature_types=None, svm_class=SKSVMRegressor, **kwargs
    ):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            svm_class: A scikit-learn linear class.
            **kwargs: Kwargs pass to linear class at initialization time.
        """
        super().__init__(feature_names, feature_types, svm_class, **kwargs)

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        self.sk_model_ = self.svm_class(**self.kwargs)
        return super().fit(X, y)


class SVMClassifier(BaseSVM, ClassifierMixin, ExplainerMixin):
    """SVM classifier.

    Currently wrapper around svm models in scikit-learn: https://github.com/scikit-learn/scikit-learn
    """

    def __init__(
        self,
        feature_names=None,
        feature_types=None,
        svm_class=SKSVMClassifier,
        **kwargs,
    ):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            svm_class: A scikit-learn svm class.
            **kwargs: Kwargs pass to linear class at initialization time.
        """
        super().__init__(feature_names, feature_types, svm_class, **kwargs)

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        self.sk_model_ = self.svm_class(**self.kwargs)
        return super().fit(X, y)

    def predict_proba(self, X):
        """Probability estimates on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Probability estimate of instance for each class.
        """
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        return self._model().predict_proba(X)
