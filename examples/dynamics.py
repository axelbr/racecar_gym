from time import sleep

import numpy as np
import pybullet as p
import pybullet_data


def get_image(carId):
    width, height = 320, 320
    position, orientation = p.getBasePositionAndOrientation(carId)
    orientation = p.getEulerFromQuaternion(orientation)
    r, pich, y = orientation
    pich = np.math.radians(80.0)
    orientation = p.getQuaternionFromEuler((r, pich, y))
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    camera_vector = rot_matrix.dot([1, 0, 0])
    up_vector = rot_matrix.dot([0, 0, 1])
    x, y, z = position
    position = x - 0.2, y, z + 0.4
    target = position + 1.0 * camera_vector
    view_matrix = p.computeViewMatrix(position, target, up_vector)
    aspect_ratio = float(width) / height
    proj_matrix = p.computeProjectionMatrixFOV(60, aspect_ratio, 0.01, 1.0)
    (_, _, px, _, _) = p.getCameraImage(width=width,
        height=height,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix)


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
p.setAdditionalSearchPath('../models')
carId = p.loadURDF('vehicles/racecar/racecar.urdf', [0, 0, 0.07], useFixedBase=False)
car = carId

for wheel in range(p.getNumJoints(car)):
    p.setJointMotorControl2(car,
        wheel,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0)
    p.getJointInfo(car, wheel)

    # p.setJointMotorControl2(car,10,p.VELOCITY_CONTROL,targetVelocity=1,force=10)
c = p.createConstraint(car,
    9,
    car,
    11,
    jointType=p.JOINT_GEAR,
    jointAxis=[0, 1, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=1, maxForce=10000)

c = p.createConstraint(car,
    10,
    car,
    13,
    jointType=p.JOINT_GEAR,
    jointAxis=[0, 1, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, maxForce=10000)

c = p.createConstraint(car,
    9,
    car,
    13,
    jointType=p.JOINT_GEAR,
    jointAxis=[0, 1, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, maxForce=10000)

c = p.createConstraint(car,
    16,
    car,
    18,
    jointType=p.JOINT_GEAR,
    jointAxis=[0, 1, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=1, maxForce=10000)

c = p.createConstraint(car,
    16,
    car,
    19,
    jointType=p.JOINT_GEAR,
    jointAxis=[0, 1, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, maxForce=10000)

steeringLinks = [0, 2]
maxForce = 0.5
nMotors = 2
motorizedwheels = [15]
speedMultiplier = 25.
steeringMultiplier = 0.5

force = 0.2
targetVelocity = 14.0 * speedMultiplier
# print("targetVelocity")
# print(targetVelocity)
steeringAngle = 0.0 * steeringMultiplier


# print("steeringAngle")
# print(steeringAngle)
# print("maxForce")
# print(maxForce)

def get_velocity(body_id):
    v_linear, v_rotation = p.getBaseVelocity(body_id)
    position, orientation = p.getBasePositionAndOrientation(body_id)
    rot = p.getMatrixFromQuaternion(orientation)
    rot = np.reshape(rot, (-1, 3)).transpose()
    v_linear = rot.dot(v_linear)
    v_rotation = rot.dot(v_rotation)
    return np.append(v_linear, v_rotation)


for steer in steeringLinks:
    p.setJointMotorControl2(carId, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)

prev_v = 0
p.setRealTimeSimulation(False)
i = 0
target = 1.0
while True:
    i += 1

    motorizedwheels = [8]
    for motor in motorizedwheels:
        p.setJointMotorControl2(carId,
            motor,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=target * 30.0,
            force=50.0)
    if False:
        get_image(carId)
    p.stepSimulation()
    v = get_velocity(carId)[0]
    if v >= target - 0.5:
        target += 1.0

    accel_v = v - prev_v
    prev_v = v
    print(f'velocity: {v:.2f}, acceleration: {accel_v:.2f}')
    sleep(0.01)
