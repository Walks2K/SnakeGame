"""
PPO test agent for Snake using Stable Baselines 3
"""


import os

from stable_baselines3 import PPO
from snake_env import SnakeEnv


EPISODES = 10


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
    print(newest_file)
    return newest_file


def main():
    """
    Main function
    """
    env = SnakeEnv()
    model = PPO.load(latest_model(), env=env)

    for episode in range(1, EPISODES + 1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, _info = env.step(action)
            score += rewards
            env.render()

        print(f"Episode: {episode}, Score: {score}")


if __name__ == "__main__":
    main()
