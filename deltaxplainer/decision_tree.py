from typing import List, Tuple, Dict, Union
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from .utils import extract_rules, compress_rule

def get_rules_decision_tree(
    X_delta: pd.DataFrame,
    y_delta: pd.DataFrame,
    params: Dict[str, Union[int, float, str]] = {
        "max_depth": 4,
        "criterion": "gini",
        "min_samples_leaf": 10,
        "min_impurity_decrease": 0,
    },
    class_diff: int = 1
) -> Tuple[DecisionTreeClassifier, List[str]]:
    """
    Extract rules from a decision tree.

    Args:
        X_delta (pd.DataFrame): Input data for deltamodels
        y_delta (pd.DataFrame): Disagreeing labels
        params (dict, optional): Decision tree parameters. Defaults to {"max_depth":4, "criterion":"gini", "min_samples_leaf":10,"min_impurity_decrease": 0}.
        class_diff (int, optional): Class to consider as disagreeing. Defaults to 1.

    Returns:
        Tuple[DecisionTreeClassifier, List[str]]: Trained deltamodel and list of segments where models make different predictions
    """
    max_depth = params["max_depth"]
    criterion = params["criterion"]
    min_samples_leaf = params["min_samples_leaf"]
    min_impurity_decrease = params["min_impurity_decrease"]

    delta = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease
    )
    delta.fit(X_delta, y_delta)
    raw_rules = extract_rules(delta, feature_names=X_delta.columns, class_names=["0", "1"], class_diff=class_diff)
    segments = [compress_rule(rule) for rule in raw_rules]
    return delta, segments
