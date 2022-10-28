from setuptools import setup

setup(
    name="sk2torch",
    version="1.2.0",
    description="Convert scikit-learn models to PyTorch modules",
    long_description="Convert scikit-learn models to PyTorch modules",
    packages=["sk2torch"],
    install_requires=["scikit-learn", "torch"],
    author="Alex Nichol",
    author_email="unixpickle@gmail.com",
    url="https://github.com/unixpickle/sk2torch",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
