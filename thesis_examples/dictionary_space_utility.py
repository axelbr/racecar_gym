import gymnasium
import math
import numpy as np
from numpy import inf
import torch as th


#create a function to edit structure of observation space before it goes into sb3
def unwrap_obs_space(parallel_env):

    new_dict = dict.fromkeys(parallel_env.observation_space.keys())
    for key in parallel_env.observation_space.keys():
        #don't give time to sb for now
        pose = ('x','y','z','roll','pitch','yaw')
        accel = ('acc_x_trans','acc_y_trans','acc_z_trans','acc_x_rot','acc_y_rot','acc_z_rot')
        vel = ('vel_x_trans','vel_y_trans','vel_z_trans','vel_x_rot','vel_y_rot','vel_z_rot')
        vel_val = (gymnasium.spaces.Box(low=-14,high=14,shape=(1,), dtype=np.float64),gymnasium.spaces.Box(low=-14,high=14,shape=(1,),dtype=np.float64),gymnasium.spaces.Box(low=-14,high=14,shape=(1,),dtype=np.float64),
                        gymnasium.spaces.Box(low=-6,high=6,shape=(1,),dtype=np.float64), gymnasium.spaces.Box(low=-6,high=6,shape=(1,),dtype=np.float64),  gymnasium.spaces.Box(low=-6,high=6,shape=(1,),dtype=np.float64))
        accel_val = gymnasium.spaces.Box(low=-inf,high=inf,shape=(1,),dtype=np.float64)
        pose_val = (gymnasium.spaces.Box(low=-100,high=100,shape=(1,), dtype=np.float64),gymnasium.spaces.Box(low=-100,high=100,shape=(1,),dtype=np.float64),gymnasium.spaces.Box(low=-100,high=100,shape=(1,),dtype=np.float64),
                        gymnasium.spaces.Box(low=-3,high=3,shape=(1,),dtype=np.float64), gymnasium.spaces.Box(low=-math.pi,high=math.pi,shape=(1,),dtype=np.float64),
                    gymnasium.spaces.Box(low=-math.pi,high=math.pi,shape=(1,),dtype=np.float64))

        pose_dict = dict(zip(pose,pose_val))
        accel_dict = dict.fromkeys(accel,accel_val)
        vel_dict = dict(zip(vel,vel_val))

        spaces_tmp = pose_dict | accel_dict
        spaces = spaces_tmp | vel_dict

        new_dict[key] = gymnasium.spaces.Dict(spaces)

    return new_dict

#not used --> delete later
def rewrap(parallel_env):

    for key in parallel_env.observation_space.keys():
        pose = 'pose'
        accel = 'acceleration'
        vel = 'velocity'
        lidar = 'lidar'
        time = 'time'

        accel_val = gymnasium.spaces.Box(low = -inf,high=inf,shape=(6,),dtype=np.float64)
        pose_val = gymnasium.spaces.Box(low = np.array([-100,-100,-3,-math.pi,-math.pi,-math.pi]), high=np.array([100,100,3,math.pi,math.pi,math.pi]),shape=(6,),dtype=np.float64)
        vel_val = gymnasium.spaces.Box(low = np.array([-14, -14, -14, -6,-6,-6]), high = np.array([14,14,14,6,6,6]),shape=(6,), dtype=np.float64)
        lid_val = gymnasium.spaces.Box(low = 0.25, high = 15.25, shape = (1080,), dtype=np.float64)
        tim_val = gymnasium.spaces.Box(low = 0.0, high = 1.0, shape = (), dtype = np.float32)

        pose_dict = dict.fromkeys(pose, pose_val)
        accel_dict = dict.fromkeys(accel, accel_val)
        vel_dict = dict.fromkeys(vel, vel_val)
        lid_dict = dict.fromkeys(lidar,lid_val)
        tim_dict = dict.fromkeys(time,tim_val)

        spaces_tmp = pose_dict | accel_dict | vel_dict | lid_dict | tim_dict
        spaces = spaces_tmp

        parallel_env.observation_space[key] = gymnasium.spaces.Dict(spaces)

        return parallel_env



    #a function that takes the original (as defined in racecarenvs documentation) observation and fits it into a new observation space that is compatible with stablebaselines3
    #For now, assume this is an observation for a SINGLE agent

def refit_obs(obs):
    pose = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
    accel = ('acc_x_trans', 'acc_y_trans', 'acc_z_trans', 'acc_x_rot', 'acc_y_rot', 'acc_z_rot')
    vel = ('vel_x_trans', 'vel_y_trans', 'vel_z_trans', 'vel_x_rot', 'vel_y_rot', 'vel_z_rot')

    #this is not very programatic or modular --> change in the future if there is time
    #obs are just regular dictionaries NOT gymnasium spaces --> as returned from the reset function in multi_agent_race.py
    #pose_val = th.from_numpy(obs['pose'])
    pose_val = obs['pose']
    #print(pose_val.shape)
    #pose_val = th.reshape(pose_val, (6,))
    #accel_val = th.from_numpy(obs['acceleration'])
    accel_val = obs['acceleration']
    #accel_val = th.reshape(accel_val, (6,))
    #vel_val = th.from_numpy(obs['velocity'])
    vel_val = obs['velocity']
    #vel_val = th.reshape(vel_val, (6,))

    #convert numpy arrays to torch tensors because that is the datatype MultiInputPolicy network expects apparently
    pose_dict = dict(zip(pose,pose_val))
    accel_dict = dict(zip(accel,accel_val))
    vel_dict = dict(zip(vel,vel_val))


    spaces_tmp = pose_dict | accel_dict
    spaces = spaces_tmp | vel_dict

    for key in spaces.keys():
        #rehaping such that all the entries have shape (1,)
        spaces[key] = np.reshape(spaces[key], (1,))

    new_dict = spaces
    print(new_dict['x'].shape)

    #returning the newly formed observations and the simulation time from the original observation
    return new_dict, obs['time']


#returns a flattened observation space
def flatten_obs_space():

    #low_b = [pose,vel,accel]

    low_b = np.array([-100,-100,-3,-math.pi,-math.pi,-math.pi,-14, -14, -14, -6,-6,-6,-inf,-inf,-inf,-inf,-inf,-inf])
    high_b = np.array([100,100,3,math.pi,math.pi,math.pi,14,14,14,6,6,6,inf,inf,inf,inf,inf,inf])

    flat_space = gymnasium.spaces.Box(low = low_b, high = high_b, shape = (18,), dtype = np.float64)

    return flat_space

def flatten_acts_space():

    low_b = np.array([-1,-1])
    high_b = np.array([1,1])

    #action_space = [motor, steering]
    flat_space = gymnasium.spaces.Box(low = low_b, high = high_b, shape = (2,), dtype = np.float32)

    return flat_space

#returns a flattened observation
def flatten_obs(obs):
    flat_obs = obs['pose']
    keys = ['velocity','acceleration']
    for key in keys:
        flat_obs = np.append(flat_obs,obs[key])
    flat_obs = np.reshape(flat_obs,(18,))

    return flat_obs


# returns the flattented action **
def flatten_acts(act):
    flat_act = act['motor']
    keys = ['motor', 'steering']

    for key in keys:
        flat_act = np.append(flat_act, act[key])

    return flat_act


#given a flattened action space, return it unflattened in the style of racecar env
def unflatten_acts(acts):

    unflat_act = {}

    #assuming the actions from rllib will come as dictionaries with form agentID:act
    for key in acts.keys():
        ray_acts = acts[key]
        motor = ray_acts["motor"]
        steering = ray_acts["steering"]

        tmp_dict = {"motor":motor,"steering":steering}
        unflat_act[key] = tmp_dict

    return unflat_act

#preprocessing for obs before using .policy functions such as predict_values
def policyprep(obs,sb3_env):
    #old shape was [18,]
    obs = th.from_numpy(obs)
    #new shape is [1,18]
    obs = obs.reshape((-1,) + sb3_env.observation_space.shape)

    return obs









