from setuptools import setup

setup(
    name="dbread",
    version="0.1.0",
    author="Vineet Tiruvadi",
    author_email="virati@gmail.com",
    packages=["dbread"],
    package_dir={"": "src"},
    url="http://pypi.python.org/pypi/DBRead/",
    license="LICENSE.txt",
    description="Multimodal, Model-based DBS Analysis",
    long_description=open("README.md").read(),
    install_requires=["numpy", "jaxlib", "jax"],
)
