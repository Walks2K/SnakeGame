"""
PPO training agent for Snake using Stable Baselines 3
"""


import os
import time

from stable_baselines3 import PPO
from snake_env import SnakeEnv

STEPS = 100
TIMESTEPS = 10000
MODELS_DIR = f"models/PPO-{int(time.time())}"
LOGDIR = f"logs/PPO-{int(time.time())}"
RESUME_TRAINING = False


def latest_model():
    """
    Find newest file in all subdirs of 'models'
    """
    newest_file = None
    for root, _dirs, files in os.walk('models'):
        for file in files:
            if newest_file is None or os.path.getmtime(os.path.join(root, file)) > \
                    os.path.getmtime(newest_file):
                newest_file = os.path.join(root, file)
    print(newest_file)
    return newest_file


def main():
    """
    Main function
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    env = SnakeEnv()
    if RESUME_TRAINING:
        model = PPO.load(latest_model(), env=env,
                         verbose=1, tensorboard_log=LOGDIR)
    else:
        model = PPO('MlpPolicy', env, verbose=1,
                    tensorboard_log=LOGDIR)

    for i in range(1, STEPS):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{MODELS_DIR}/snake_ppo_{TIMESTEPS * i}")


if __name__ == "__main__":
    main()
