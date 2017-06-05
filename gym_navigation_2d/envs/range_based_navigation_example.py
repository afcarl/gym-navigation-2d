import gym
import gym_navigation_2d

from env_generator import EnvironmentGenerator, Environment, Obstacle
import numpy as np

env = gym.make('Limited-Range-Based-Navigation-2d-v0')


"""
world_collection_pickle_filename = 'AAA.pkl'
worlds = EnvironmentCollection(world_collection_pickle_filename)
env.world = worlds.map_collection[0]
"""

print 'Creating world'
eg = EnvironmentGenerator(x_range=[0, 640], y_range=[0, 480], width_range=[10, 30], height_range=[10,50])
#centers, widths, heights = eg.sample_axis_aligned_rectangles(density=0.0001)
#obstacles = eg.merge_rectangles_into_obstacles(centers, widths, heights, epsilon=1)
obstacles = {0: Obstacle(np.array([100,200]), 50, 100) }

print 'Done Creating world'


env.world = Environment(eg.x_range, eg.y_range, obstacles)
env.set_initial_position(np.array([-20.0, -20.0]))
env.set_destination(np.array([700.0, 520.0]))
env.max_observation_range = 20.0
env.destination_tolerance_range = 20.0

assert (env.viewer is None)

observation = env.reset()
for t in range(100):
    print 'time', t
    print 'A'
    env.render()
    print 'B'
    
    action = env.action_space.sample()

    print 'C'
    
    observation, reward, done, info = env.step(action)

    print 'D'
    
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
