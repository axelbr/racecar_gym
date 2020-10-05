from time import time, sleep

import gym
import racecar_gym
from agents.gap_follower import GapFollower

import pybullet as p

p.connect(p.GUI)

id = p.loadURDF('models/vehicles/f1tenth_car/f1tenth_car.urdf')

for i in range(p.getNumJoints(id)):
    print(p.getJointInfo(id, i))

while True:
    p.stepSimulation()