import mujoco
import mujoco.viewer

from time import sleep
import numpy as np

m = mujoco.MjModel.from_xml_path("alpha_model/alpha.mjcf")
d = mujoco.MjData(m)

def key_callback(key):
    print(f"Key pressed: {key}")

class ReachEnvViz:
    def __init__(self, m, d, key_callback):
        self.m = m
        self.d = d
        self.key_callback = key_callback
        

    def loop(self, actions):
        with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:

            # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
            viewer.sync()

            while viewer.is_running():
                # Step the physics.
                mujoco.mj_step(m, d)

                # Add a 3x3x3 grid of variously colored spheres to the middle of the scene.
                viewer.user_scn.ngeom = 0
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0],
                    pos=np.array([0.3, 0, 0]),
                    mat=np.eye(3).flatten(),
                    rgba=0.5 * np.array([1, 1, 1, 2]),
                )
                viewer.user_scn.ngeom = 1
                viewer.sync()
                sleep(0.01)
