#!/usr/bin/env python3
import argparse 

from env import Env
from agent import Agent


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def train(env: Env, agent: Agent) -> None:
    pass


def main() -> None:
    args = cli()
    
    env = Env(r"C1CCCCC1")
    agent = Agent()

    train(env, agend)

    exit(0)


if __name__ == "__main__":
    main()
