# Racecar Gym

![berlin track](docs/racecar_gym.gif)

A gym environment for a miniature, [F1Tenth](https://f1tenth.org/)-like racecar using the bullet physics engine with pybullet.
## Installation

> **Warning**
> We now migrated to *gymnasium*. If you need to use the old gym api, you can either use some of the [compatibility wrappers](https://gymnasium.farama.org/content/gym_compatibility/) of gymnasium
> or you can use older versions of this gym. The last version supporting the old gym api is tagged as **gym-api**:
>
>`git checkout gym-api; pip install -e .`

You can install ``racecar_gym`` with the following commands:

```shell_script
git clone https://github.com/axelbr/racecar_gym.git
cd racecar_gym
pip install -e .
```

On the first use of this gym, the tracks are downloaded automatically.

Of course, you can download them manually too. Here is how you can do this from the command line:
```shell_script
cd ./models/scenes
VERSION=v1.0.0
wget https://github.com/axelbr/racecar_gym/releases/download/tracks-${VERSION}/all.zip
unzip all.zip
```

## Environments

The observation and action space is a `Dict` holding the agents and their id's. The observation and action space for a single agent 
is also a `Dict`, which is described in more detail below. In general, observations are obtained through sensors and commands
are executed by actuators. Vehicles can have multiple sensors and actuators. Those are described in the vehicle configuration
(e.g. [differential racecar](models/vehicles/racecar/racecar.yml)). Agents, which consist of a vehicle and an assigned task,
are specified in the scenario file (e.g. [austria.yml](scenarios/multi_austria.yml)). In this file, agents are described by the
sensors to use (note that they must be available in the vehicle configuration) and the corresponding task. Have a look at
[tasks.py](racecar_gym/core/tasks.py) to see all available tasks.

**Example:**
```yaml
world:
  name: austria
agents:
  - id: A
    vehicle:
      name: racecar
      sensors: [lidar, pose, velocity, acceleration]
      actuators: [motor, steering]
      color: blue # default is blue, one of red, green, blue, yellow, magenta or random
    task:
      task_name: maximize_progress
      params: {laps: 1, time_limit: 120.0, terminate_on_collision: False}
```

This example specifies a scenario on the [Austria](models/scenes/austria/austria.yml) track.
One agent with id **A** is specified. The agent controls the differential drive racecar defined in [differential racecar](models/vehicles/racecar/racecar.yml), identified by its name.
The scenario tells the agent to use only the specified sensors (lidar, pose, velocity, acceleration). 
Optionally, one can also specify a color for the car. The default color is blue. Available colors are listed above.

The task which is assigned to this agent is also identified by name (implementations can be found in [tasks.py](racecar_gym/core/tasks.py)). Task parameters are passed by the dict *params*.

### Observations

Observations are obtained by (possibly noisy) sensors. Parameters for the sensors as well as the level of noise, can be
configured in the corresponding vehicle configuration (e.g.  [differential racecar](models/vehicles/racecar/racecar.yml)). 
In the scenario specification, one can specify which of the available  sensors should be actually used. 
The observation space is a dictionary where the names of the sensors are the keys 
which map to the actual measurements. Currently, five sensors are implemented:
pose, velocity, acceleration, LiDAR and RGB Camera. Further, the observation space also includes the current simulation time.

|Key|Space|Defaults|Description|
|---|---|---|---|
|pose|`Box(6,)`| |Holds the position (`x`, `y`, `z`) and the orientation (`roll`, `pitch`, `yaw`) in that order.|
|velocity|`Box(6,)`| |Holds the `x`, `y` and `z` components of the translational and rotational velocity.|
|acceleration|`Box(6,)`| |Holds the `x`, `y` and `z` components of the translational and rotational acceleration.|
|lidar|`Box(<scans>,)`|`scans: 1080`|Lidar range scans.|
|rgb_camera|`Box(<height>, <width>, 3)`|`height: 240, width: 320`|RGB image of the front camera.|

### Actions
The action space for a single agent is a defined by the actuators of the vehicle. 
**By default**, [differential racecar](models/vehicles/racecar/racecar.yml) defines two actuators: motor and steering. 
The action space is therefore a dictionary with keys `motor` and `steering`.

Alternatevely, the agent can control the target speed and steering, but must be defined in the scenario specification.
In this case, the action space is a dictionary with keys `speed` and `steering`.

Note, that the action space of the car is normalized between -1 and 1.
The action space can include the following actuators:

| Key      |Space| Description                                                                 |
|----------|---|-----------------------------------------------------------------------------|
| motor    |`Box(low=-1, high=1, shape=(1,))`| Throttle command. If negative, the car accelerates backwards.               |
| speed    |`Box(low=-1, high=1, shape=(1,))`| Normalized target speed. |
| steering |`Box(low=-1, high=1, shape=(1,))`| Normalized steering angle.                                                  |

### State
In addition to observations obtained by sensors, the environment passes back the true state of each vehicle in each
step (the state is returned as the *info* dictionary). The state is a dictionary, where the keys are the ids of all agents.
Currently, the state looks like this:

|Key|Type|Description|
|---|---|---|
|wall_collision|`bool`|True if the vehicle collided with the wall.|
|opponent_collisions|`List[str]`|List of opponent id's which are involved in a collision with the agent.|
|pose|`NDArray[6]`|Ground truth pose of the vehicle (x, y, z, roll, pitch, yaw).|
|acceleration|`NDArray[6]`|Ground truth acceleration of the vehicle (x, y, z, roll, pitch, yaw).|
|velocity|`NDArray[6]`|Ground truth velocity of the vehicle (x, y, z, roll, pitch, yaw).|
|progress|`float`|Current progress in this lap. Interval: [0, 1]|
|time|`float`|Simulation time.|
|checkpoint|`int`|Tracks are subdivided into checkpoints to make sure agents are racing in clockwise direction. Starts at 0.|
|lap|`int`|Current lap.|
|rank|`int`|Current rank of the agent, based on lap and progress.|
|wrong_way|`bool`|Indicates wether the agent goes in the right or wrong direction.|
|observations|`Dict`|The most recent observations of the agent.

## Available API's

`racecar_gym` currently implements the [Gym API](https://www.gymlibrary.ml/) and the [PettingZoo API](https://www.pettingzoo.ml/).

### Gym API
To use the *Gym API* you can either instantiate environments with the standard keys or by loading custom scenarios.
In either case, you have to load the `gym_api` module from this package:
```python
import gymnasium
import racecar_gym.envs.gym_api

# For predefined environments:
env = gymnasium.make(
    id='SingleAgentAustria-v0',
    render_mode='human'
)

# For custom scenarios:
env = gymnasium.make(
    id='SingleAgentRaceEnv-v0', 
    scenario='path/to/scenario',
    render_mode='rgb_array_follow', # optional
    render_options=dict(width=320, height=240, agent='A') # optional
)

done = False
reset_options = dict(mode='grid')
obs, info = env.reset(options=reset_options)

while not done:
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, states = env.step(action)
    done = terminated or truncated

env.close()
```
The predefined env-strings are of the form
```
<Multi|Single>Agent<track>-v0
e.g.: MultiAgentAustria-v0
```

For further documentation on available gym environments, please refer to the [Gym Documentation](./docs/gym.md) (under construction)
and the examples shown in [examples/gym_examples/](./examples/gym_examples).

### PettingZoo API
For multi-agent races, we also implement the [PettingZoo API](https://www.pettingzoo.ml/). To create a PettingZoo environment,
you can use the `pettingzoo_api` module:
```python
from racecar_gym.envs import pettingzoo_api
env = pettingzoo_api.env(scenario='path/to/scenario')
```
For further documentation on available PettingZoo environments, please refer to the [PettingZoo Documentation](./docs/pettingzoo.md) (under construction)
and the examples shown in [examples/pettingzoo_examples/](./examples/pettingzoo_examples).


## Maps

Currently available maps are listed below. The gridmaps are originally from the [F1Tenth](https://github.com/f1tenth) repositories.

| Image                                 | Name     |
|---------------------------------------|----------|
| ![austria](docs/tracks/austria.png)   | Austria  |                                                                                                   |
| ![berlin](docs/tracks/berlin.png)     | Berlin   |
| ![montreal](docs/tracks/montreal.png) | Montreal |
| ![torino](docs/tracks/torino.png)     | Torino   |
| ![circle](docs/tracks/circle.png)     | Circle   |
| ![plechaty](docs/tracks/plechaty.png)   | Plechaty |



## Notes
Please note that this is work in progress, and interfaces might change. Also more detailed documentation and additional scenarios will follow.
