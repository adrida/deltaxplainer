import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from  ..deltaxplainer import DeltaXplainer 

class TestDeltaXplainer(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_moons(n_samples=250, noise=0.1, random_state=1, shuffle=True)
        self.X, self.y  = pd.DataFrame(self.X),pd.DataFrame(self.y)
        self.model_f = DecisionTreeClassifier(max_depth=3, random_state=42).fit(self.X,self.y)
        self.model_g = RandomForestClassifier(n_estimators=50, random_state=0).fit(self.X,self.y)
    def test_fit_predict(self):

        model = DeltaXplainer()
        model.fit(self.X, self.model_f, self.model_g)
        print(model.segments)
        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), 250)  # Add more assertions as needed
        self.assertEqual(len(model.segments), 3)  # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
