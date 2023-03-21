from setuptools import setup

setup(
    name="aurora_classification",
    version="1.0.0",
    description="Repository of common tools of Aurorai team",
    author="Aleksandr Nevarko",
    packages=["aurora_classification"],
    install_requires=[
        "numpy>=1.21.6",
        "Pillow>=9.2.0",
        "torch>=1.12",
        "tqdm>=4.64.0",
        "opencv-python",
        "matplotlib>=3.2.2",
        "scipy>=1.4.1",
        "albumentations>=1.0.3",
        "pandas>=1.1.4"
    ],
)
