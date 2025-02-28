import importlib.resources

import mujoco
import mujoco.viewer
from brax.io import html

from brax.envs.base import PipelineEnv, State, base
import jax.numpy as jp
import jax
from brax.io import mjcf

from jax.random import PRNGKey

from jax import config

config.update("jax_enable_x64", True)


class ReachArm(PipelineEnv):
    def __init__(self, rng, backend="mjx", **kwargs):
        package = importlib.resources.files(__package__)
        fn = str(package / "alpha_model" / "alpha.mjcf")
        mj_model = mujoco.MjModel.from_xml_path(fn)  # pyright:ignore
        self.sys = mjcf.load_model(mj_model)
        # self.target_pos = jp.array([0.0, 0.0, 0.0])
        super().__init__(sys=self.sys, backend=backend, **kwargs)
        # self.target_pos = jax.random.uniform(rng, (3,), minval=-0.5, maxval=0.5)
        self.target_pos = jp.array([0.3, 0, 0])
        # data = mujoco.MjData(x)

    def reset(self, rng: jax.Array) -> State:
        q = (
            self.sys.init_q
            + jax.random.uniform(rng, (self.sys.q_size(),), minval=-0.2, maxval=0.2)
            # + jp.array([0.0, jp.pi])
        )
        qd = jp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}
        return State(pipeline_state, obs, reward, done, metrics)

    @property
    def action_size(self):
        return self.sys.act_size()

    def _get_obs(self, pipeline_state):
        # print(pipeline_state)
        return pipeline_state.q

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action / 10)
        obs = self._get_obs(pipeline_state)
        if state.pipeline_state is None:
            raise ValueError("pipeline_state is None")
        reward = -self.cost(state.pipeline_state)
        done = 0.0
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def cost(self, state: base.State):
        return 10*jp.linalg.norm(state.x.pos[-1] - self.target_pos)
