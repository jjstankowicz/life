from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="life",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    # entry_points={"console_scripts": ["my_package=my_package.__main__:main"]},
    author="James Stankowicz",
    author_email="jj.stankowicz@gmail.com",
    description="Make music of the game of life",
    url="https://github.com/jjstankowicz/life",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
