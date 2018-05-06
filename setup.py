"""
A setuptools-based setup module.
"""

from setuptools import setup

setup(
    name='wavenet',
    version='0.1.0',
    description='An implementation of WaveNet for TensorFlow.',
    url='https://github.com/unixpickle/wavenet',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    packages=['wavenet'],
    install_requires=[
        'numpy>=1.0.0,<2.0.0'
    ],
    extras_require={
        "tf": ["tensorflow>=1.4.0"],
        "tf_gpu": ["tensorflow-gpu>=1.4.0"]
    }
)
