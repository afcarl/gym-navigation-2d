import gym
import sys

from baselines import deepq
import gym_navigation_2d


def main(policy_pkl_file):
    env = gym.make('State-Based-Navigation-2d-Map0-Goal0-v0')
    act = deepq.load(policy_pkl_file)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage: python3 run_dqn_on_state.py policy.pkl")
        sys.exit(1)
    
    main(sys.argv[1])
