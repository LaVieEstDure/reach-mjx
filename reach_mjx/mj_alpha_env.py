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
        fn = str(package / 'alpha_urdf' / 'newmodel.xml')
        mj_model = mujoco.MjModel.from_xml_path(fn)  # pyright:ignore
        self.sys = mjcf.load_model(mj_model)
        self.target_pos = jp.array([0.0, 0.0, 0.0])
        super().__init__(sys=self.sys, backend=backend, **kwargs)
        self.target_pos = jax.random.uniform(rng, (3,), minval=-0.5, maxval=0.5)
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

    @property
    def action_size(self):
        return self.sys.act_size()

    def _get_obs(self, pipeline_state):
        # print(pipeline_state)
        return pipeline_state.q

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        if state.pipeline_state is None:
            raise ValueError("pipeline_state is None")
        reward = -self.reward(state.pipeline_state)
        done = 0.0
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def reward(self, state: base.State):
        return jp.linalg.norm(state.x.pos[-1] - self.target_pos)


# def rollout_us(step_env, state, us):
#     def step(state, u):
#         state = step_env(state, u)
#         return state, (state.reward, state.pipeline_state)

#     _, (rews, pipline_states) = jax.lax.scan(step, state, us)
#     return rews, pipline_states

# def render_us(step_env, sys, state, us):
#     rollout = []
#     # rew_sum = 0.0
#     Hsample = us.shape[0]
#     for i in range(Hsample):
#         rollout.append(state.pipeline_state)
#         state = step_env(state, us[i])
#         # rew_sum += state.reward
#     # rew_mean = rew_sum / (Hsample)
#     # print(f"evaluated reward mean: {rew_mean:.2e}")
#     return html.render(sys, rollout)

#

# if __name__ == "__main__":
#     rng = PRNGKey(42)
#     test = ReachArm()
#     state = jax.jit(test.reset)(rng)
#     step_env = jax.jit(test.step)
#     step_env(state, jp.zeros((5,)))


#     html_val = rollout_us(step_env, state, jp.zeros((100, 5)))
#     print(html_val)
