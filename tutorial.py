#!/usr/bin/env python3
"""
Source: https://www.youtube.com/watch?v=Mut_u40Sqz4
"""
import os
import gym  # https://github.com/openai/gym 
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    # Create environment with OpenAI's RL gym for agent to
    # act in. 
    environment_name = "CartPole-v0"
    env = gym.make(environment_name)

    episodes = 100
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False 
        score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward 
        print(f"Episode:{episode} Score:{score}")
    env.close()

    exit(0)


if __name__ == "__main__":
    main()
