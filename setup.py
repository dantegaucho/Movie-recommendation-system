from setuptools import setup, find_packages

setup(
    name='recommender_app',
    version='0.1.0',
    description='A hybrid movie recommender system using collaborative filtering, SVD, and CNN',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    py_modules=['recommender_app'],
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'scikit-surprise',  # for the `surprise` library
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
