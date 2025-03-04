import os.path
from setuptools import find_packages, setup

# the directory containing this file
ROOT = os.path.dirname(__file__)

# the text of the README file
with open(os.path.join(ROOT, "README.md"), "r") as f:
    README = f.read()

setup(
    name="pylinsql",
    version="0.1.18",
    description="Language-Integrated SQL Queries in Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/hunyadi/pylinsql",
    author="Levente Hunyadi",
    author_email="hunyadi@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 7 - Inactive",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["asyncpg", "json_strong_typing"],
)
