#!/usr/bin/env python
import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="tamer",
    version="0.1.0",
    description="Tree-Aware Transformer for Handwritten Mathematical Expression Recognition",
    author="Jianhua Zhu",
    author_email="zhujianhuapku@pku.edu.cn",
    url="https://github.com/PKU-ICST-MLNLP/TAMER",
    python_requires=">=3.10",
    install_requires=[
        "einops>=0.7.0",
        "editdistance>=0.6.2",
        "pytorch-lightning>=2.2.0",
        "torchmetrics>=1.2.0",
        "jsonargparse[signatures]>=4.27.0",
        "typer>=0.9.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "pillow>=10.0.0",
        "pandas>=2.0.0",
    ],
    packages=find_packages(),
)
