from typing import Tuple

Position = Tuple[float, float, float] # x, y, z coordinates
Orientation = Tuple[float, float, float] # euler angles
Pose = Tuple[Position, Orientation]
Quaternion = Tuple[float, float, float, float]
Velocity = Tuple[float, float, float, float, float, float]