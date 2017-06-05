import gym
from gym import error, spaces, utils
from gym.utils import seeding
from env_generator import Environment, EnvironmentCollection
from gym.envs.classic_control import rendering
from gym.spaces import Box, Tuple

from math import pi
import numpy as np

class LimitedRangeBasedNavigation2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.world = None
        self.destination = None
        self.state = np.array([0.0, 0.0])
        self.init_position = self.state.copy()
        self.max_observation_range = 10
        self.destination_tolerance_range = 5
        self.viewer = None
        self.num_beams = 16
        self.max_speed = 5
        self.action_space = Tuple( (Box(0.0, self.max_speed, (1,)), Box(0.0, 2*pi, (1,))) ) 
        self.observation_space = Box(-1.0, self.max_observation_range, (self.num_beams,)) 
        
    def set_initial_position(self, init_position):
        self.init_position = init_position
        self.state = self.init_position.copy()

    def set_destination(self, destination):
        self.destination = destination
    
    def _get_observation(self, state):
        delta_angle = 2*pi/self.num_beams
        ranges = [self.world.raytrace(self.state,
                                      i * delta_angle,
                                      self.max_observation_range) for i in xrange(self.num_beams)]
        return ranges
        
    def _step(self, action):
        old_state = self.state
        self.state += np.array([ action[0][0], action[1][0] ])
        
        reward = 0
        done = False
        info = {}

        if np.linalg.norm(self.destination - self.state) < self.destination_tolerance_range:
            reward = 1
            done = True
            
        if not self.world.point_is_in_free_space(self.state[0], self.state[1], epsilon=0.25):
            reward = -1

        if not self.world.segment_is_in_free_space(old_state[0], old_state[1],
                                                   self.state[0], self.state[1],
                                                   epsilon=0.25):
            reward = -1
            
        self.observation = self._get_observation(self.state)
        return self.observation, reward, done, info

    
    def _reset(self):
        self.state = self.init_position

        
    def _render(self, mode='human', close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None
            return

        screen_width = self.world.x_range[1] - self.world.x_range[0]
        screen_height = self.world.y_range[1] - self.world.y_range[0]

        if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height)

            for i in xrange(len(self.world.obstacles)):
                obs = self.world.obstacles[i]
                for c,w,h in zip(obs.rectangle_centers, obs.rectangle_widths, obs.rectangle_heights):
                    l = -w/2.0
                    r = w/2.0
                    t = h/2.0
                    b = -h/2.0

                    rectangle = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                    tr = rendering.Transform(translation=(c[0], c[1]))
                    rectangle.add_attr(tr)
                    self.viewer.add_geom(rectangle)
                    rectangle.set_color(.8,.6,.4)

            
        
