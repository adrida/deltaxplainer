from setuptools import setup

setup(
    name='deltaxplainer',
    version='1',
    packages=['deltaxplainer'],
    install_requires=[
    'numpy>=1.20',
    'scikit-learn>=0.24',
    'pandas>=1.3',
    'pytest'
    ],
    author='Adam Rida',
    description='A package for DeltaXplainer model implemented from the paper https://arxiv.org/pdf/2309.17095.pdf',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
