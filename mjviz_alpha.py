import mujoco
import mujoco.viewer

# x = mujoco.MjModel.from_xml_path("alpha_urdf/alpha.urdf")
x = mujoco.MjModel.from_xml_path("alpha_urdf/newmodel.xml")
d = mujoco.MjData(x)

mujoco.viewer.launch(x,d)

# mujoco.viewer.launch(x,d)
# import mujoco
# x = mujoco.MjSpec.from_file("alpha_urdf/alpha.urdf")
# x.compile() # first compilation ignores URDF inertias and infers from geoms
# for body in x.bodies:
#     body.explicitinertial = True
# x.compiler.inertiafromgeom = 2
# mjmodel = x.compile() # not sure if you need this line, please try with and without :)
# print(x.to_xml())
