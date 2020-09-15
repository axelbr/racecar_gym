from typing import TypeVar, Tuple, Dict

Action = TypeVar('Action')
State = TypeVar('State')
Observation = TypeVar('Observation')
Reward = float
StepReturn = Tuple[Observation, Reward, bool, Dict]