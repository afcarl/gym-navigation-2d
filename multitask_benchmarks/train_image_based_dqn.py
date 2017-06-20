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
    env = gym.make('Image-Based-Navigation-2d-Map0-Goal0-v0')
    env.action_space = spaces.Discrete(100)

    num_output_filters = 8
    kernel_size = (5,5)
    stride = 1

    convs = [(num_output_filters, kernel_size, stride)]
    hidden_fcns = [64]
    model = deepq.models.cnn_to_mlp(convs, hidden_fcns, dueling=False)

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
