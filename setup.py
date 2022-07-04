from setuptools import setup, find_packages

setup(
    name="racecar_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=['pybullet==3.1.7',
                      'scipy', # version automatically choosen w.r.t numpy
                      'numpy==1.22.3',
                      'gym==0.21.0',
                      'yamldataclassconfig==1.5.0',
                      'nptyping<2.0',
                      'pettingzoo==1.18.1'
                      ],
    author='Axel Brunnbauer',
    author_email='axel.brunnbauer@gmx.at',
    description='An RL environment for a miniature racecar using the pybullet physics engine.',
    url='https://github.com/axelbr/racecar_gym',
)