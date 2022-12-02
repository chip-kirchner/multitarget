from setuptools import setup, find_packages

setup(
    name="multicapture",
    version="0.1",
    description="Mutli-Target Capture Environment",
    author="Charles Kirchner",
    url="",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy==1.23.5", "gym==0.25.2", "pyglet==1.5.27"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
