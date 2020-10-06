import bpy
import os

"""
Blender script for generating obj files and the corresponding SDF file in the directory of the blender file.
"""

track_name = 'track1'

sdf = f"""
<?xml version="1.0" encoding="UTF-8"?>
  <sdf version="1.6">
  <world name="{track_name}">
    <gravity>0 0 -9.8</gravity>"""

basedir = bpy.path.abspath('//')
bpy.ops.object.select_all(action='DESELECT')
scene = bpy.context.scene
for ob in scene.objects:
    path = os.path.join(basedir, ob.name + '.obj')
    ob.select_set(True)
    if ob.type == 'MESH':
        bpy.ops.export_scene.obj(
            use_normals=True,
            use_mesh_modifiers=True,
            use_uvs=True,
            use_materials=True,
            use_triangles=True,
            use_blen_objects=True,
            path_mode='COPY',
            axis_up='Z',
            axis_forward='X',
            filepath=os.path.join(basedir, ob.name + '.obj'),
            use_selection=True,
        )
        sdf += f'''
            <model name="{ob.name}">
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
                      <uri>{ob.name + '.obj'}</uri>
                    </mesh>
                  </geometry>
                </collision>
                <visual name="visual">
                  <geometry>
                    <mesh>
                      <scale>1 1 1</scale>
                      <uri>{ob.name + '.obj'}</uri>
                    </mesh>
                  </geometry>
                </visual>
              </link>
            </model>
            '''
    ob.select_set(False)


sdf += """
  </world>
</sdf>
"""

with open(f'{basedir}{track_name}.sdf', 'w') as f:
    f.write(sdf)
    f.close()