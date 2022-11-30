from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='smarthealing',
      version="0.0.1",
      description="Mdical sick leave prediction",
      license="",
      author="The Dream Team",
      author_email="jaimesoroa@gmail.com",
      url="",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      )