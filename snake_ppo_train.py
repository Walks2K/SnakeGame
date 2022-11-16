"""
PPO training agent for Snake using Stable Baselines 3
"""


import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from snake_env import SnakeEnv

USE_LATEST_MODEL = True

MODELS_DIR = f"models/PPO-{int(time.time())}"
LOG_DIR = f"logs/PPO-{int(time.time())}"

NUM_ENVS = 8
NUM_TIMESTEPS = 1_000_000
SAVE_FREQ = 25000 // NUM_ENVS


def latest_model():
    """
    Find newest file in all subdirs of 'models'
    """
    newest_file = None
    for root, _dirs, files in os.walk("models"):
        for file in files:
            if newest_file is None or os.path.getmtime(
                os.path.join(root, file)
            ) > os.path.getmtime(newest_file):
                newest_file = os.path.join(root, file)

    # set models_dir and log_dir to the same as the latest model
    global MODELS_DIR  # pylint: disable=global-statement
    global LOG_DIR  # pylint: disable=global-statement
    MODELS_DIR = f"models/{os.path.basename(os.path.dirname(newest_file))}"
    LOG_DIR = f"logs/{os.path.basename(MODELS_DIR)}"

    print(newest_file)
    return newest_file


if __name__ == "__main__":
    env = make_vec_env(SnakeEnv, n_envs=8)
    if USE_LATEST_MODEL:
        model = PPO.load(latest_model(), env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ, save_path=MODELS_DIR, name_prefix="PPO"
    )

    model.learn(
        total_timesteps=NUM_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    model.save(f"{MODELS_DIR}/PPO")
