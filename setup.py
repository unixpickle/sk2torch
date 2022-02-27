from setuptools import setup

setup(
    name="sk2torch",
    py_modules=["sk2torch"],
    install_requires=["scikit-learn", "torch"],
)
