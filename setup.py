#!/usr/bin/env python3

import setuptools
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.1"

REPO_NAME = "ML-PREDICITNG-DRUG-RESPONSE-CURVES"
AUTHOR_USER_NAME = "Pangani"
AUTHOR_EMAIL = "roypangani@gmail.com"

HYPHEN = "-e ."

def get_required(file_path: str) -> List[str]:
    requirements = []
    with open(file_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

        if HYPHEN in requirements:
            requirements.remove(HYPHEN)
    return requirements

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A project that uses the GDSC dataset to train Sparse MOGP to predict the multiple drug sensitivities on curve",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=setuptools.find_packages(),
    install_requires=get_required('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)