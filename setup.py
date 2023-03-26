from setuptools import setup, find_packages

with open('requirements.txt') as f:
    lines = f.read().split('\n')
    requirements = []
    for line in lines:
        if line.startswith('git+'):
            link, package = line.split('#egg=')
            requirements.append(f'{package} @ {link}#{package}')
        else:
            requirements.append(line)


setup(
    name="racecar_gym",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    author='Axel Brunnbauer',
    author_email='axel.brunnbauer@gmx.at',
    description='An RL environment for a miniature racecar using the pybullet physics engine.',
    url='https://github.com/axelbr/racecar_gym',
)