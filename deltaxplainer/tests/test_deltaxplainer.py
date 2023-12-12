import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from  ..deltaxplainer import DeltaXplainer 

class TestDeltaXplainer(unittest.TestCase):
    def setUp(self):
        # Set up your test data
        self.X_train = pd.DataFrame(np.random.rand(100, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
        self.y_train = pd.Series(np.random.randint(0, 2, 100))
        self.X_test = pd.DataFrame(np.random.rand(20, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])

    def test_fit_predict(self):
        model = DeltaXplainer()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), 20)  # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
