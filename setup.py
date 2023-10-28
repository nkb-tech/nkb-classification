from setuptools import setup

with open('requirements.txt', 'r') as file:
    install_requires = file.read().split()

setup(
    name="nkb_classification",
    version="1.0.0",
    description="Repository of common tools of NKB team",
    author="Aleksandr Nevarko",
    packages=["nkb_classification"],
    install_requires=install_requires,
)
