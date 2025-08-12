import os
import json
import gymnasium as gym
import numpy as np
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def make_env(rank, env_config):
    def _init():
        env = RedGymEnv(env_config)
        return env
    return _init

if __name__ == "__main__":
    num_cpu = 8  # Number of processes to use
    ep_length = 128

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path) as f:
        env_config = json.load(f)

    try:
        env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
        env = VecTransposeImage(env)

        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./checkpoints/", name_prefix="ppo_model")

        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=checkpoint_callback)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
