# DeltaXplainer, XAI model comparison and explanations

[![Open In Jupyter Notebook](https://img.shields.io/badge/Open%20in-Jupyter%20Notebook-orange.svg)](https://github.com/adrida/deltaxplainer/blob/master/notebooks/get_started.ipynb) [![Documentation Status](https://readthedocs.org/projects/deltaxplainer/badge/?version=latest)](https://deltaxplainer.readthedocs.io/en/latest/?badge=latest) (documentation not up to date)


Package for DeltaXplainer model implemented from the paper Dynamic Interpretability for Model Comparison via Decision Rules, A Rida, MJ Lesot, X Renard, C Marsala, DynXAI workshop at ECML PKDD 2023, https://arxiv.org/pdf/2309.17095.pdf

DeltaXplainer is an algortihm aiming at explaining differences between two black box binary classifiers.

![DeltaXplainer Schema](https://github.com/adrida/deltaxplainer/blob/master/assets/delta.png?raw=true)

The models takes as input the two models to compare and generate explanations. The package is originally built to support comparison of sklearn models but any object with a `predict` method doing binary classification should work.

The explanations are provided using decision rules. We propose to answer to the question "Why are the models different?" by showing "Where" they differ.

The explanations are a list of segments where the two black box models make different predictions.

Ideas for future improvements include considering other explanations format and ways to extract knowledge from the delta model.

### Installation

`pip install deltaxplainer`

### Getting Started

In order to have a hands on example please refer to this [notebook](https://github.com/adrida/deltaxplainer/blob/master/notebooks/get_started.ipynb)

#### Generating Explanations

Assuming you want to explain differences between `classifer_a` and `classifer_b` trained on `X_a, y_a` and `X_b, y_b`

For more details on how the method works please refer to the original paper.

`from deltaxplainer import DeltaXplainer`

`X_delta_train = pd.concat([X_a, X_b])`

`delta_model = DeltaXplainer(X_delta_train, model_a, model_b).fit()`

`print(delta_model.segments)`

This last line gives you a list of segments where the two models differ.



[under construction]

