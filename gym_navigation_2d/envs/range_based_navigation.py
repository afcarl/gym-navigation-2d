import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .env_generator import Environment, EnvironmentCollection
from gym.envs.classic_control import rendering
from gym.spaces import Box, Tuple

from math import pi, cos, sin
import numpy as np

import os

class LimitedRangeBasedPOMDPNavigation2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 worlds_pickle_filename=os.path.join(os.path.dirname(__file__), "assets", "worlds_640x480_v2.pkl"),
                 world_idx=0,
                 initial_position = np.array([-20.0, -20.0]),
                 destination = np.array([520.0, 400.0]),
                 max_observation_range = 100.0,
                 destination_tolerance_range=20.0,
                 add_self_position_to_observation=False,
                 add_goal_position_to_observation=False,
                 add_map_to_observation=False):

        worlds = EnvironmentCollection()
        worlds.read(worlds_pickle_filename)

        self.world = worlds.map_collection[world_idx]
        self.destination = destination
        
        assert not (self.destination is None)
        self.init_position = initial_position
        self.state = self.init_position.copy()
        
        self.max_observation_range = max_observation_range
        self.destination_tolerance_range = destination_tolerance_range
        self.viewer = None
        self.num_beams = 16
        self.max_speed = 5

        self.add_self_position_to_observation = add_self_position_to_observation
        self.add_goal_position_to_observation = add_goal_position_to_observation
        self.add_map_to_observation = add_map_to_observation
        
        low = np.array([0.0, 0.0])
        high = np.array([self.max_speed, 2*pi])
        self.action_space = Box(low, high)#Tuple( (Box(0.0, self.max_speed, (1,)), Box(0.0, 2*pi, (1,))) )
        low = [-1.0] * self.num_beams
        high = [self.max_observation_range] * self.num_beams
        if add_self_position_to_observation:
            low.extend([-10000., -10000.]) # x and y coords
            high.extend([10000., 10000.])
        if add_goal_position_to_observation:
            low.extend([-10000., -10000.]) # x and y coords
            high.extend([10000., 10000.])


        self.observation_space = Box(np.array(low), np.array(high))
        self.observation = []
        

    def _get_observation(self, state):
        delta_angle = 2*pi/self.num_beams
        ranges = [self.world.raytrace(self.state,
                                      i * delta_angle,
                                      self.max_observation_range) for i in range(self.num_beams)]

        ranges = np.array(ranges)
        if self.add_self_position_to_observation:
            ranges = np.concatenate([ranges, self.state])
        if self.add_goal_position_to_observation:
            ranges = np.concatenate([ranges, self.destination])
        return ranges

    def _dynamics(self, action, old_state):

        if type(action) == type(np.int64(0)):
            assert (action < 100 and action >= 0)
            
            i = int(action) / 10
            j = int(action) % 10

            dv = self.max_speed / 10.0
            dtheta = 2*pi / 10.0

            v = i * dv
            theta = j * dtheta
    
        else:
            v = action[0]
            theta = action[1]

        dx = v*cos(theta)
        dy = v*sin(theta)

        new_state = old_state + np.array([dx, dy])        
        return new_state
        
        
    def _step(self, action):
        old_state = self.state.copy()
        self.state = self._dynamics(action, old_state)
        
        reward = -1 # minus 1 for every timestep you're not in the goal
        done = False
        info = {}

        if np.linalg.norm(self.destination - self.state) < self.destination_tolerance_range:
            reward = 20 # for reaching the goal
            done = True

        if not self.world.point_is_in_free_space(self.state[0], self.state[1], epsilon=0.25):
            reward = -5 # for hitting an obstacle

        if not self.world.segment_is_in_free_space(old_state[0], old_state[1],
                                                   self.state[0], self.state[1],
                                                   epsilon=0.25):
            reward = -5 # for hitting an obstacle

        self.observation = self._get_observation(self.state)
        return self.observation, reward, done, info


    def _reset(self):
        self.state = self.init_position
        return self._get_observation(self.state)

    def _plot_state(self, viewer, state):
        polygon = rendering.make_circle(radius=5, res=30, filled=True)
        state_tr = rendering.Transform(translation=(state[0], state[1]))
        polygon.add_attr(state_tr)
        viewer.add_onetime(polygon)


    def _plot_observation(self, viewer, state, observation):
        delta_angle = 2*pi/self.num_beams
        for i in range(len(observation)):
            r = observation[i]
            if r < 0:
                r = self.max_observation_range

            theta = i*delta_angle
            start = (state[0], state[1])
            end = (state[0] + r*cos(theta), state[1] + r*sin(theta))

            line = rendering.Line(start=start, end=end)
            line.set_color(.5, 0.5, 0.5)
            viewer.add_onetime(line)

    def _append_elements_to_viewer(self, viewer,
                                   screen_width,
                                   screen_height,
                                   obstacles,
                                   destination=None,
                                   destination_tolerance_range=None):

        viewer.set_bounds(left=-100, right=screen_width+100, bottom=-100, top=screen_height+100)

        L = len(obstacles)
        for i in range(L):

            obs = obstacles[i]
            for c,w,h in zip(obs.rectangle_centers, obs.rectangle_widths, obs.rectangle_heights):
                l = -w/2.0
                r = w/2.0
                t = h/2.0
                b = -h/2.0

                rectangle = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                tr = rendering.Transform(translation=(c[0], c[1]))
                rectangle.add_attr(tr)
                rectangle.set_color(.8,.6,.4)
                viewer.add_geom(rectangle)


        if not (destination is None):
            tr = rendering.Transform(translation=(destination[0], destination[1]))
            polygon = rendering.make_circle(radius=destination_tolerance_range, res=30, filled=True)
            polygon.add_attr(tr)
            polygon.set_color(1.0, 0., 0.)
            viewer.add_geom(polygon)

    def _render(self, mode='human', close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None
            return

        screen_width = (self.world.x_range[1] - self.world.x_range[0])
        screen_height = (self.world.y_range[1] - self.world.y_range[0])

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self._append_elements_to_viewer(self.viewer,
                                            screen_width,
                                            screen_height,
                                            obstacles=self.world.obstacles,
                                            destination=self.destination,
                                            destination_tolerance_range=self.destination_tolerance_range)

        self._plot_state(self.viewer, self.state)
        self._plot_observation(self.viewer, self.state, self.observation)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class StateBasedMDPNavigation2DEnv(LimitedRangeBasedPOMDPNavigation2DEnv):
    
    def __init__(self, *args, **kwargs):
        LimitedRangeBasedPOMDPNavigation2DEnv.__init__(self, *args, **kwargs)
        
        self.rectangles = [(c[0], c[1], w, h) for obs in self.world.obstacles for c, w, h in zip(obs.rectangle_centers,
                                                                                                 obs.rectangle_widths,
                                                                                                 obs.rectangle_heights)]

        self.rectangle_obs_vector = [el for rect in self.rectangles for el in rect]

        
        inf = 1000000.0
        low = [-inf, -inf, 0.0, 0.0] 
        high = [inf, inf, inf, 2*pi]

        if self.add_map_to_observation:
            low.extend([-inf, -inf, self.world.x_range[0], self.world.y_range[0]] * len(self.rectangles))
            high.extend([inf,  inf, self.world.x_range[1], self.world.y_range[1]] * len(self.rectangles))

        
        if self.add_goal_position_to_observation:
            low.extend([-inf, -inf]) # x and y coords
            high.extend([inf, inf])

        
        self.observation_space = Box(np.array(low), np.array(high))

        
    def _plot_observation(self, viewer, state, observation):
        pass

    
    def _get_observation(self, state):
        dist_to_closest_obstacle, absolute_angle_to_closest_obstacle = self.world.range_and_bearing_to_closest_obstacle(state[0], state[1])
        obs =  [state[0], state[1], dist_to_closest_obstacle, absolute_angle_to_closest_obstacle] 

        if self.add_map_to_observation:
            obs.extend(self.rectangle_obs_vector)

        obs = np.array(obs)

        if self.add_goal_position_to_observation:
            obs = np.concatenate([obs, self.destination])
            
        return obs
