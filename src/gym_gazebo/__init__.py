import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

register(
	id='GazeboPool-v0',
	entry_point='gym_gazebo.envs.pool_gym_env:GazeboPoolv0Env',
	max_episode_steps=3000,
)

