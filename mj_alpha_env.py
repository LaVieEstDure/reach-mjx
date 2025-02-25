import mujoco
import mujoco.viewer

from brax.envs.base import PipelineEnv, State
import jax.numpy as jp
import jax
from brax.io import mjcf

# x = mujoco.MjModel.from_xml_path("alpha_urdf/alpha.urdf")
# d = mujoco.MjData(x)

# mujoco.viewer.launch(x,d)

class ReachArm(PipelineEnv):
    def __init__(self, backend='positional', **kwargs):
        mj_model = mujoco.MjModel.from_xml_path("alpha_urdf/alpha.urdf") # pyright:ignore 
        self.sys = mjcf.load_model(mj_model)
        super().__init__(sys=self.sys, backend=backend, **kwargs)
        # data = mujoco.MjData(x)

    def reset(self, rng: jax.Array) -> State:
        q = (
            self.sys.init_q
            + jax.random.uniform(rng, (self.sys.q_size(),), minval=-0.01, maxval=0.01)
            # + jp.array([0.0, jp.pi])
        )
        qd = jp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}
        return State(pipeline_state, obs, reward, done, metrics)
    
    def _get_obs(self, pipeline_state):
        return pipeline_state.pipeline_state.q

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        reward = jp.cos(pipeline_state.q[1]) - jp.abs(pipeline_state.qd[0])
        done = 0.0
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

test = ReachArm()
