from gym.envs.registration import register


register(
    id='Limited-Range-Based-Navigation-2d-v0',
    entry_point='gym_navigation_2d.envs:LimitedRangeBasedNavigation2DEnv',
)

register(
    id='Image-Based-Navigation-2d-v0',
    entry_point='gym_navigation_2d.envs:ImageBasedNavigation2DEnv',
)

