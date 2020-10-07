# python setup.py install
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='RNN implementation on functional magnetic resonance imaging (fMRI) data to predict brain states during anticipation of aversive and neutral events.',
    author='Chirag Limbachia',
    license='MIT',
)
