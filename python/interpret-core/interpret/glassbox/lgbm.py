from abc import abstractmethod

import colorlover as cl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils import (gen_global_selector, gen_local_selector,
                     gen_name_from_class, gen_perf_dicts, hist_per_column,
                     unify_data)


def gen_global_selector3(
    X, feature_names, feature_types, feature_idxs, importance_scores, round=3
):
    """Custom selector which can deal with pairwise features."""
    records = []

    for f_name, f_type, f_idx in zip(feature_names, feature_types, feature_idxs):
        record = {}
        record["Name"] = f_name
        record["Type"] = f_type

        if f_idx < X.shape[1]:
            col_vals = X[:, f_idx]
            nz_count = np.count_nonzero(col_vals)
            col_vals = col_vals.astype("U")
            record["# Unique"] = len(np.unique(col_vals))
            record["% Non-zero"] = nz_count / X.shape[0]
        else:
            record["# Unique"] = np.nan
            record["% Non-zero"] = np.nan

        # if importance_scores is None:
        #     record["Importance"] = np.nan
        # else:
        #     record["Importance"] = importance_scores[feat_idx]

        records.append(record)

    # columns = ["Name", "Type", "# Unique", "% Non-zero", "Importance"]
    columns = ["Name", "Type", "# Unique", "% Non-zero"]
    df = pd.DataFrame.from_records(records, columns=columns)
    if round is not None:
        return df.round(round)
    else:  # pragma: no cover
        return df


def extract_gain_from_treeinfo(tree_info, feature_names):
    """Extract gain obtained by each feature.

    Args:
        tree_info: tree_info of lgbm class
        feature_names: list of the feature names

    Returns:
        return a dictionary whose keys are the names of feature including pairwise features,
        and values are the gain obtained by each feature.
    """
    num_trees = len(tree_info)
    features2gain = {}

    for idx in range(num_trees):
        tree_struct = tree_info[idx]["tree_structure"]

        gain = 0
        if tree_info[idx]["num_leaves"] == 2:
            features = feature_names[tree_struct["split_feature"]]
            gain += tree_struct["split_gain"]
        else:
            feature_1 = feature_names[tree_struct["split_feature"]]
            if "split_feature" in tree_struct["left_child"]:
                feature_2 = feature_names[tree_struct["left_child"]["split_feature"]]
                gain += tree_struct["split_gain"]
            else:
                feature_2 = feature_names[tree_struct["right_child"]["split_feature"]]
                gain += tree_struct["split_gain"]
            if feature_1 != feature_2:
                features = f"{feature_1} x {feature_2}"
            else:
                features = str(feature_1)

        if features not in features2gain:
            features2gain[features] = gain
        else:
            features2gain[features] += gain

    return features2gain


def convert_treeinfo_to_features2treeidx(tree_info):
    """Convert a tree_info of LGBM to a dictionary whose key is a "idx" of a feaure or
    "idx x idx" (for a pairwise feature), and value is a list which contains the trees
    that uses the corresponding feature.

    Args:
        tree_info: tree_info of lgbm class

    Returns:
        return a dictionary
    """
    num_trees = len(tree_info)
    features2tree = {}

    for idx in range(num_trees):
        tree_struct = tree_info[idx]["tree_structure"]
        if tree_info[idx]["num_leaves"] == 2:
            features = str(tree_struct["split_feature"])
        else:
            feature_1 = tree_struct["split_feature"]
            if "split_feature" in tree_struct["left_child"]:
                feature_2 = tree_struct["left_child"]["split_feature"]
            else:
                feature_2 = tree_struct["right_child"]["split_feature"]
            if feature_1 != feature_2:
                features = f"{feature_1} x {feature_2}"
            else:
                features = str(feature_1)

        if features not in features2tree:
            features2tree[features] = [idx]
        else:
            features2tree[features].append(idx)

    return features2tree


def update_feature_names_and_types(features2tree, feature_names, feature_types):
    """Create a new feature_names and feature_types which contains pairwise features

    Args:
        feature2tree: a dictionary whose keys are the idx or "idx x idx" of each features
        feature_names: a list of feature names
        feature_types: a list of feature types

    Returns:
        feature_names: a list of feature names including pairwise features
        feature_types: a list of feature types including pairwise features
        feature_idxs: ith value indicates which of original features ith feature is made up of
                      (Ex. if ith feature is "1 x 2", then feature_idxs[i] = [1, 2])
    """
    feature_idxs = [[i] for i in range(len(feature_names))]
    for key in features2tree.keys():
        features = key.split(" x ")
        if len(features) == 2:
            tmp_name = (
                f"{feature_names[int(features[0])]} x {feature_names[int(features[1])]}"
            )

            feature_names.append(tmp_name)
            feature_types.append("interaction")
            feature_idxs.append([int(features[0]), int(features[1])])

    return feature_names, feature_types, feature_idxs


class BaseLGBM:

    available_explanations = ["local", "global"]
    explainer_type = "model"

    def __init__(
        self,
        feature_names=None,
        feature_types=None,
        lgbm_class=LGBMClassifier,
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
        self.lgbm_class = lgbm_class
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

        model_ = self._model()
        model_.fit(X, y)

        self.X_mins_ = np.min(X, axis=0)
        self.X_maxs_ = np.max(X, axis=0)
        self.categorical_uniq_ = {}

        for i, feature_type in enumerate(self.feature_types):
            if feature_type == "categorical":
                self.categorical_uniq_[i] = list(sorted(set(X[:, i])))

        self.bin_counts_, self.bin_edges_ = hist_per_column(X, self.feature_types)

        # extract the tree structure
        self.tree_info = model_._Booster.dump_model()["tree_info"]
        # convert the tree structure to a dictionary which shows the
        # assignment relationship between features and trees.
        self.feature2tree = convert_treeinfo_to_features2treeidx(self.tree_info)
        # add pairwise features to faeture_names and feature_types
        (
            self.feature_names_withpairwise,
            self.feature_types_withpairwise,
            self.feature_idxs_withpairwise,
        ) = update_feature_names_and_types(
            self.feature2tree, self.feature_names, self.feature_types
        )

        self.global_selector = gen_global_selector(
            X, self.feature_names_withpairwise, self.feature_types_withpairwise, None
        )

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

    def tree_importance(self):
        """Returns a dictionary which shows the importance of each tree"""
        return extract_gain_from_treeinfo(
            self.tree_info, self.feature_names_withpairwise
        )

    def _get_grid_points(self, feat_idxs):
        feat_min = self.X_mins_[feat_idxs]
        feat_max = self.X_maxs_[feat_idxs]
        feat_type = self.feature_types[feat_idxs]

        if feat_type == "continuous":
            # Generate x, y points to plot from coef for continuous features
            grid_points = np.linspace(feat_min, feat_max, 30)
        else:
            grid_points = np.array(self.categorical_uniq_[feat_idxs])

        return grid_points

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

        model_ = self._model()

        is_classification = is_classifier(self)
        if is_classification:
            predictions = self.predict_proba(X)[:, 1]
        else:
            predictions = self.predict(X)

        data_dicts = []
        scores_list = []
        perf_list = []
        perf_dicts = gen_perf_dicts(predictions, y, is_classification)

        feature_names_used = []
        feature_types_used = []
        feature_idxs_used = []
        feature_groups = []
        for index, feat_idxs in enumerate(self.feature_idxs_withpairwise):
            feature_group_idxs = " x ".join([str(i) for i in feat_idxs])
            if feature_group_idxs not in self.feature2tree:
                continue

            feature_groups.append(feature_group_idxs)
            feature_names_used.append(self.feature_names_withpairwise[index])
            feature_types_used.append(self.feature_types_withpairwise[index])
            feature_idxs_used.append(index)

        for i, instance in enumerate(X):
            scores = []
            for feature_group_idxs in feature_groups:
                score = 0
                for tree_idx in self.feature2tree[feature_group_idxs]:
                    score += model_.predict(
                        instance.reshape(1, -1),
                        raw_score=True,
                        start_iteration=tree_idx,
                        num_iteration=1,
                    )[0]
                scores.append(score)
            scores = np.array(scores)

            scores_list.append(scores)
            data_dict = {}
            data_dict["data_type"] = "univariate"

            # Performance related (conditional)
            perf_dict_obj = None if perf_dicts is None else perf_dicts[i]
            data_dict["perf"] = perf_dict_obj
            perf_list.append(perf_dict_obj)

            # Names/scores
            data_dict["names"] = feature_names_used
            data_dict["scores"] = scores

            # Values
            # new_instance = np.zeros(len(scores))
            # new_instance[: len(instance)] = instance
            # data_dict["values"] = new_instance

            data_dicts.append(data_dict)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "local_feature_importance",
                    "value": {
                        "scores": scores_list,
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
            feature_names=feature_names_used,
            feature_types=feature_types_used,
            name=name,
            selector=selector,
        )

    def explain_global(self, name=None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        if name is None:
            name = gen_name_from_class(self)

        # get importance of each tree
        tree_importance = self.tree_importance()
        overall_data_dict = {
            "names": list(tree_importance.keys()),
            "scores": list(tree_importance.values()),
        }

        _model = self._model()

        specific_data_dicts = []
        feature_names_used = []
        feature_types_used = []
        feature_idxs_used = []
        for index, feat_idxs in enumerate(self.feature_idxs_withpairwise):
            feature_group_idxs = " x ".join([str(i) for i in feat_idxs])

            # if the trained LGBM does not use this feature,
            if feature_group_idxs not in self.feature2tree:
                continue

            feature_names_used.append(self.feature_names_withpairwise[index])
            feature_types_used.append(self.feature_types_withpairwise[index])
            feature_idxs_used.append(index)

            # single feature
            if len(feat_idxs) == 1:
                grid_points = self._get_grid_points(index)
                grids = np.zeros((grid_points.shape[0], self.X.shape[1]))
                grids[:, feat_idxs[0]] = grid_points

                scores = np.zeros(grid_points.shape[0])
                # get the sum of predictions made by the trees whihc use the specified feature
                for tree_idx in self.feature2tree[feature_group_idxs]:
                    scores += _model.predict(
                        grids, raw_score=True, start_iteration=tree_idx, num_iteration=1
                    )

                data_dict = {
                    "type": self.feature_types_withpairwise[index],
                    "names": grid_points,
                    "scores": scores,
                    "density": {
                        "scores": self.bin_counts_[index],
                        "names": self.bin_edges_[index],
                    },
                }
            # pairwise feature
            else:
                grid_points_x = self._get_grid_points(feat_idxs[0])
                grid_points_y = self._get_grid_points(feat_idxs[1])
                xx, yy = np.meshgrid(grid_points_x, grid_points_y)
                xx_ravel = xx.ravel()
                yy_ravel = yy.ravel()
                grids = np.zeros((xx_ravel.shape[0], self.X.shape[1]))
                grids[:, feat_idxs[0]] = xx_ravel
                grids[:, feat_idxs[1]] = yy_ravel

                scores = np.zeros(xx_ravel.shape[0])
                for tree_idx in self.feature2tree[feature_group_idxs]:
                    scores += _model.predict(
                        grids, raw_score=True, start_iteration=tree_idx, num_iteration=1
                    )

                lower_bound = np.min(scores - 0.0)
                upper_bound = np.max(scores + 0.0)
                bounds = (lower_bound, upper_bound)

                data_dict = {
                    "type": "interaction",
                    "left_names": xx_ravel,
                    "right_names": yy_ravel,
                    "scores": scores,
                    "scores_range": bounds,
                }

            specific_data_dicts.append(data_dict)

        self.global_selector = gen_global_selector3(
            self.X,
            feature_names_used,
            feature_types_used,
            feature_idxs_used,
            None,
        )

        internal_obj = {
            "overall": overall_data_dict,
            "specific": specific_data_dicts,
            "mli": [
                {
                    "explanation_type": "global_feature_importance",
                    "value": {"scores": list(self.tree_importance().values())},
                }
            ],
        }

        assert len(feature_names_used) == len(feature_types_used)
        assert len(feature_names_used) == len(feature_idxs_used)

        return FeatureValueExplanation(
            "global",
            internal_obj,
            feature_names=feature_names_used,
            feature_types=feature_types_used,
            name=name,
            selector=self.global_selector,
        )


class LGBMClassifier(BaseLGBM, ClassifierMixin, ExplainerMixin):
    """LGBM classifier."""

    def __init__(
        self,
        feature_names=None,
        feature_types=None,
        lgbm_class=LGBMClassifier,
        **kwargs,
    ):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            lgbm_class: A LGBM class.
            **kwargs: Kwargs pass to linear class at initialization time.

        Examples:
            >>> clf = LGBMClassifier(num_iterations=200, max_depth=2, num_leaves=3)
            >>> clf.fit(X_train, y_train)
            >>> clf.score(X_test, y_test)
            0.8316831683168316
            >>> clf_global = clf.explain_global(name='LGBM')
            >>> clf_local = clf.explain_local(X_test[:5], name='LGBM')
            >>> show([clf_global, clf_local])
        """
        super().__init__(feature_names, feature_types, lgbm_class, **kwargs)
        if kwargs["max_depth"] == 1:
            pass
        elif kwargs["max_depth"] == 2:
            if kwargs["num_leaves"] == 3:
                pass
            else:
                raise ValueError("if max_depth = 2, num_leaves should be 3")
        else:
            raise ValueError("max_depth should be less than 3")

    def _model(self):
        return self.lgbm_class_

    def fit(self, X, y):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        self.lgbm_class_ = self.lgbm_class(**self.kwargs)
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
