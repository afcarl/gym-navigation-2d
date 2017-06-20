import gym

from gym import spaces
from baselines import deepq
import gym_navigation_2d
import types

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100000000 #and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def main():
    env = gym.make('State-Based-MDP-Navigation-2d-Map0-Goal0-KnownGoalPosition-v0')
    #env = gym.make('Image-Based-Navigation-2d-Map0-Goal0-v0')
    env.action_space = spaces.Discrete(100)
    model = deepq.models.mlp([64])

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=1,
        callback=callback
    )

    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
