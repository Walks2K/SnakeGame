"""
PPO training agent for Snake using Stable Baselines 3
"""


import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from snake_env import SnakeEnv

MODELS_DIR = f"models/PPO-{int(time.time())}"
LOGDIR = f"logs/PPO-{int(time.time())}"


def main():
    """
    Main function
    """
    env = SnakeEnv()
    model = PPO('MlpPolicy', env, verbose=1,
                tensorboard_log=LOGDIR)

    chkpt_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODELS_DIR,
        name_prefix='PPO')

    model.learn(total_timesteps=1e5,
                callback=chkpt_callback, tb_log_name="PPO")
    model.save(f"{MODELS_DIR}/snake_ppo_final")


if __name__ == "__main__":
    main()
