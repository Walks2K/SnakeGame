"""
Test snake env
"""


from stable_baselines3.common.env_checker import check_env

from snake_env import SnakeEnv


# Use stable_baselines3.common.env_checker.check_env to check if the environment is correct
env = SnakeEnv()
check_env(env)


# Run some test games
TEST_GAMES = 10
for i in range(TEST_GAMES):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
