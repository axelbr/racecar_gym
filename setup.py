from setuptools import setup, find_packages

setup(
    name="racecar_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=['pybullet', 'scipy', 'numpy', 'gym', 'yamldataclassconfig', 'nptyping'],
    author='Axel Brunnbauer',
    author_email='axel.brunnbauer@gmx.at',
    description='A gym environment for a miniature racecar using the pybullet physics engine.',
    url='https://github.com/axelbr/racecar_gym',
)