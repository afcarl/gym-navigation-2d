from gym.envs.registration import register


register(
    id='Limited-Range-Based-POMDP-Navigation-2d-v0',
    entry_point='gym_navigation_2d.envs:LimitedRangeBasedPOMDPNavigation2DEnv',
)

register(
    id='State-Based-MDP-Navigation-2d-v0',
    entry_point='gym_navigation_2d.envs:StateBasedMDPNavigation2DEnv',
)

register(
    id='Image-Based-Navigation-2d-v0',
    entry_point='gym_navigation_2d.envs:ImageBasedNavigation2DEnv',
)

