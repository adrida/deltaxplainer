Usage Guide
===========

DeltaXplainer is a Python package designed to explain the differences between two binary classifiers using decision tree-based explanations.

Installation
------------

Install the `deltaxplainer` package using:

.. code-block:: bash

    pip install deltaxplainer

Getting Started
---------------

In this guide, we'll cover the basic steps to use DeltaXplainer. For a more detailed example, refer to the [Getting Started Notebook](https://github.com/adrida/deltaxplainer/blob/master/notebooks/get_started.ipynb).

1. Import the necessary modules:

.. code-block:: python

    from deltaxplainer import DeltaXplainer

2. Create two classifiers `classifier_a` and `classifier_b`:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_train_a, X_train_b, y_train_a, y_train_b = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create classifiers
    classifier_a = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier_b = RandomForestClassifier(n_estimators=100, random_state=42)

    classifier_a.fit(X_train_a, y_train_a)
    classifier_b.fit(X_train_b, y_train_b)

3. Create a DeltaXplainer instance and generate explanations:

.. code-block:: python

    delta_model = DeltaXplainer(X_train_a, classifier_a, classifier_b)
    delta_model.fit()

    # Print segments where the two models differ
    print(delta_model.get_segments())

Parameters for Decision Tree
-----------------------------

The decision tree used by DeltaXplainer can be customized using various parameters. Here are the key parameters:

- ``max_depth``: Maximum depth of the tree.
- ``criterion``: The function to measure the quality of a split.
- ``min_samples_leaf``: The minimum number of samples required to be at a leaf node.
- ``min_impurity_decrease``: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

You can customize these parameters when fitting the DeltaXplainer instance:

.. code-block:: python

    delta_model.fit(X_train_a, classifier_a, classifier_b, params={
        "max_depth": 8,
        "criterion": "gini",
        "min_samples_leaf": 1,
        "min_impurity_decrease": 0,
    })

For more information, refer to the [scikit-learn decision tree documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

For additional functionalities and advanced usage, please refer to the official documentation and examples.

