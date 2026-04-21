from setuptools import setup, find_packages

setup(
    name="breast_cancer_wisconsin_classifier",
    version="0.1.0",
    description="Binary classifier for the Breast Cancer Wisconsin dataset",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
