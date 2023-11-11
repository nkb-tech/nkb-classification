from typing import List

from setuptools import find_packages, setup

from nkb_classification import __version__


def get_requirements(path: str) -> List[str]:
    with open(path, "r") as file:
        requirements = file.read().split()

    return requirements


url = "https://github.com/nkb-tech/nkb-classification.git"

setup(
    name="nkb_classification",
    version=__version__,
    url=url,
    description="Repository of common tools of NKB team",
    author="Aleksandr Nevarko",
    packages=find_packages(),
    python_requires=">=3.7",
    keywords=["pytorch", "classification"],
    install_requires=get_requirements("requirements/main.txt"),
    extras_require={
        "optional": get_requirements("requirements/optional.txt"),
    },
)
