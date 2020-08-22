# Racecar Gym
A gym environment for a miniature racecar using the pybullet physics engine.
## Installation
Clone the repositroy.
You can install ``racecar_gym`` with the following commands:

```shell_script
git clone https://github.com/axelbr/racecar_gym.git
cd racecar_gym
pip install -e .
```

## Environments

The observation space is a `n-tuple` of `Dict`, where `n` is the number of agents. An observation for a single agent has the following form.

|Key|Space|Defaults|Description|
|---|---|---|---|
|pose|`Box(6,)`||Holds the position (`x`, `y`, `z`) and the orientation (`roll`, `pitch`, `yaw`) in that order.|
|velocity|`Box(6,)`||Holds the translational velocity (`x`, `y`, `z`) and the rotational velocity around the `x`, `y` and `z` axis, in that order.|
|lidar|`Box(<scans>,)`|`scans: 100`|Lidar range scans.|
|lap|`Discrete(<laps>)`|`laps: 2`|The current lap of the vehicle. `laps` is a parameter for the simulation.|
|time|`Box(6,)`||Passed time since the start of the race.|
|collision|`Discrete(2)`||Indicates if an agent is involved in a collision with the wall or an opponent.|

Currently two maps are available and a total of four scenarios are specified.

 Name | Description |
| --- | --- |


## Environment Options

* `env_name` - Name of environment, or comma-separate list of environment names to instantiate as each env in the VecEnv.
* `num_levels=0` - The number of unique levels that can be generated. Set to 0 to use unlimited levels.
* `start_level=0` - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
* `paint_vel_info=False` - Paint player velocity info in the top left corner. Only supported by certain games.
* `use_generated_assets=False` - Use randomly generated assets in place of human designed assets.
* `debug=False` - Set to `True` to use the debug build if building from source.
* `debug_mode=0` - A useful flag that's passed through to procgen envs. Use however you want during debugging.
* `center_agent=True` - Determines whether observations are centered on the agent or display the full level. Override at your own risk.
* `use_sequential_levels=False` - When you reach the end of a level, the episode is ended and a new level is selected.  If `use_sequential_levels` is set to `True`, reaching the end of a level does not end the episode, and the seed for the new level is derived from the current level seed.  If you combine this with `start_level=<some seed>` and `num_levels=1`, you can have a single linear series of levels similar to a gym-retro or ALE game.
* `distribution_mode="hard"` - What variant of the levels to use, the options are `"easy", "hard", "extreme", "memory", "exploration"`.  All games support `"easy"` and `"hard"`, while other options are game-specific.  The default is `"hard"`.  Switching to `"easy"` will reduce the number of timesteps required to solve each game and is useful for testing or when working with limited compute resources.
* `use_backgrounds=True` - Normally games use human designed backgrounds, if this flag is set to `False`, games will use pure black backgrounds.
* `restrict_themes=False` - Some games select assets from multiple themes, if this flag is set to `True`, those games will only use a single theme.
* `use_monochrome_assets=False` - If set to `True`, games will use monochromatic rectangles instead of human designed assets. best used with `restrict_themes=True`.

Here's how to set the options:

```
import gym
env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=1)
```

Since the gym environment is adapted from a gym3 environment, early calls to `reset()` are disallowed and the `render()` method does not do anything.  To render the environment, pass `render_mode="human"` to the constructor, which will send `render_mode="rgb_array"` to the environment constructor and wrap it in a `gym3.ViewerWrapper`.  If you just want the frames instead of the window, pass `render_mode="rgb_array"`.

For the gym3 vectorized environment:

```
from procgen import ProcgenGym3Env
env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=0, num_levels=1)
```

To render with the gym3 environment, pass `render_mode="rgb_array"`.  If you wish to view the output, use a `gym3.ViewerWrapper`.
