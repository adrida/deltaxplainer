# deltaxplainer

Package for DeltaXplainer model implemented from the paper https://arxiv.org/pdf/2309.17095.pdf

DeltaXplainer is an algortihm aiming at explaining differences between two black box binary classifiers. 

The models takes as input the two models to compare and generate explanations. The package is originally built to support comparison of sklearn models but any object with a `predict` method doing binary classification should work.

The explanations are provided using decision rules. We propose to answer to the question "Why are the models different?" by showing "Where" they differ.

The explanations are a list of segments where the two black box models make different predictions.


### Installation


`pip install deltaxplainer`


### Getting Started

In order to have a hands on example


[under construction]

