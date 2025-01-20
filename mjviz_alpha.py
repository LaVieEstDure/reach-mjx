import mujoco
import mujoco.viewer

x = mujoco.MjModel.from_xml_path("alpha_urdf/alpha.urdf")
d = mujoco.MjData(x)

mujoco.viewer.launch(x,d)
