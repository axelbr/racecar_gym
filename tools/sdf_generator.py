import sys

import pybullet
import glob
import os

def generate_sdf(directory, map_name, wall_name, section_prefix):
    segments = glob.glob(f'{directory}*.obj')
    segments.extend(glob.glob(''))
    segments = sorted(segments)

    sdf = """
    <?xml version="1.0" encoding="UTF-8"?>
      <sdf version="1.6">
      <world name="berlin">
        <gravity>0 0 -9.8</gravity>"""

    for i, segment in enumerate(segments):
        filename = os.path.basename(segment)
        sdf += f'''
        <model name="{section_prefix}{i}">
          <static>1</static>
          <pose frame="">0 0 0 0 0 0</pose>
          <link name="link_d0">
            <inertial>
              <mass>0</mass>
              <inertia>
                <ixx>0.166667</ixx>
                <ixy>0</ixy>
                <ixz>0</ixz>
                <iyy>0.166667</iyy>
                <iyz>0</iyz>
                <izz>0.166667</izz>
              </inertia>
            </inertial>
            <collision concave="yes" name="collision_0">
              <geometry>
                <mesh>
                  <scale>1 1 1</scale>
                  <uri>{segment}</uri>
                </mesh>
              </geometry>
            </collision>
            <visual name="visual">
              <geometry>
                <mesh>
                  <scale>1 1 1</scale>
                  <uri>{segment}</uri>
                </mesh>
              </geometry>
            </visual>
          </link>
        </model>
        '''

    sdf += """
      </world>
    </sdf>

    """

    with open(f'{directory}/{map_name}.sdf', 'w') as f:
        f.write(sdf)
        f.close()

if __name__ == '__main__':
    directory = sys.argv[1]
    map_name = sys.argv[2]
    wall_name = sys.argv[3]
    section_prefix = sys.argv[4]
    generate_sdf(directory, map_name, wall_name, section_prefix)
