import pybullet
from gym.envs.registration import register
from pybullet_utils.bullet_client import BulletClient

from .envs.multi_race_car_env import MultiRaceCarScenario

register(id='f1tenth-berlin-two-v0',
         entry_point='racecar_gym.envs:MultiRaceCarEnv',
         kwargs={
             'client_factory': lambda: BulletClient(pybullet.DIRECT),
             'scenario': MultiRaceCarScenario(
                 start_gui=True,
                 no_of_cars=2,
                 map='./models/tracks/berlin/berlin.yml',
                 cars='./models/cars/racecar_differential.yml'
         )})

register(id='f1tenth-berlin-two-gui-v0',
         entry_point='racecar_gym.envs:MultiRaceCarEnv',
         kwargs={
             'client_factory': lambda: BulletClient(pybullet.GUI),
             'scenario': MultiRaceCarScenario(
                 start_gui=True,
                 no_of_cars=2,
                 map='./models/tracks/berlin/berlin.yml',
                 cars='./models/cars/racecar_differential.yml'
         )})

register(id='f1tenth-porto-two-gui-v0',
         entry_point='racecar_gym.envs:MultiRaceCarEnv',
         kwargs={
             'client_factory': lambda: BulletClient(pybullet.GUI),
             'scenario': MultiRaceCarScenario(
                 start_gui=True,
                 no_of_cars=2,
                 map='./models/tracks/porto/porto.yml',
                 cars='./models/cars/racecar_differential.yml')
         })

register(id='f1tenth-porto-two-v0',
         entry_point='racecar_gym.envs:MultiRaceCarEnv',
         kwargs={
             'client_factory': lambda: BulletClient(pybullet.DIRECT),
             'scenario': MultiRaceCarScenario(
                 start_gui=True,
                 no_of_cars=2,
                 map='./models/tracks/porto/porto.yml',
                 cars='./models/cars/racecar_differential.yml')
         })