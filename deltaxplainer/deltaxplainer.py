from typing import Optional, Dict, List, Union
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from .decision_tree import get_rules_decision_tree

class DeltaXplainer(BaseEstimator, RegressorMixin):
    def __init__(self) -> None:
        self.X_delta: Optional[pd.DataFrame] = None
        self.segments: List[str] = []
        self.model = None
        self.f = None
        self.g = None

    def fit(
        self,
        Xf: pd.DataFrame,
        f: ClassifierMixin,
        g: ClassifierMixin,
        Xg: Optional[pd.DataFrame] = None,
        yg: Optional[pd.Series] = None,
        params: Dict[str, Union[int, float, str]] = {
            "max_depth": 8,
            "criterion": "gini",
            "min_samples_leaf": 1,
            "min_impurity_decrease": 0,
        },
        class_diff: int = 1,
        
    ) -> 'DeltaXplainer':
        """
        Fits the DeltaModel as a Decision Tree and generates decision rules

        Args:
            Xf (pd.DataFrame): Feature set of the first group
            f (ClassifierMixin) : Model f - First model to compare
            g (ClassifierMixin) : Model g - Second model to compare
            Xg (pd.DataFrame, optional): Feature set of the second group. Defaults to None.
            yg (pd.Series, optional): Target variable of the second group. Defaults to None.
            params (dict, optional): Parameters for the decision tree. Defaults to {"max_depth": 4, "criterion": "gini", "min_samples_leaf": 10, "min_impurity_decrease": 0}.
            class_diff (int, optional): Class difference. Defaults to 1.

        Returns:
            DeltaXplainer: Fitted DeltaXplainer instance
        """
        
        self.f = f
        self.g = g
        if Xg is not None and yg is not None:
            self.X_delta = pd.concat([Xf, Xg])
        else:
            self.X_delta = Xf
            
        self.y_delta=(self.f.predict(self.X_delta) != self.g.predict(self.X_delta)).astype(int)
        
        self.model,self.segments = get_rules_decision_tree(self.X_delta, self.y_delta, params=params, class_diff=class_diff)
        return self

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, None]:
        """
        Predicts the target variable for the given features

        Args:
            X (pd.DataFrame): Feature set for prediction

        Returns:
            Union[pd.Series, None]: Predicted target variable
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit method first.")
        return self.model.predict(X)
    
    def get_segments(self) -> List[str]:
        """
        Returns the generated segments/rules

        Returns:
            List[str]: List of segments/rules
        """
        return self.segments
