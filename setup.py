# setup.py (place at /Users/pabloherrero/sabat/RaTagging/RaTag/setup.py)

from setuptools import setup, find_packages

setup(
    name="ratag",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "data*"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        # Add other dependencies
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ]
    }
)