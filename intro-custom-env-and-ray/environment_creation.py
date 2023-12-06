# fmt: off
"""
Make your own custom environment
================================

This documentation overviews creating new environments and relevant
useful wrappers, utilities and tests included in Gymnasium designed for
the creation of new environments. You can clone gym-examples to play
with the code that is presented here. We recommend that you use a virtual environment:

.. code:: console

   git clone https://github.com/Farama-Foundation/gym-examples
   cd gym-examples
   python -m venv .env
   source .env/bin/activate
   pip install -e .

Subclassing gymnasium.Env
-------------------------

Before learning how to create your own environment you should check out
`the documentation of Gymnasium’s API </api/env>`__.

We will be concerned with a subset of gym-examples that looks like this:

.. code:: sh

   gym-examples/
     README.md
     setup.py
     gym_examples/
       __init__.py
       envs/
         __init__.py
         grid_world.py
       wrappers/
         __init__.py
         relative_position.py
         reacher_weighted_reward.py
         discrete_action.py
         clip_reward.py

To illustrate the process of subclassing ``gymnasium.Env``, we will
implement a very simplistic game, called ``GridWorldEnv``. We will write
the code for our custom environment in
``gym-examples/gym_examples/envs/grid_world.py``. The environment
consists of a 2-dimensional square grid of fixed size (specified via the
``size`` parameter during construction). The agent can move vertically
or horizontally between grid cells in each timestep. The goal of the
agent is to navigate to a target on the grid that has been placed
randomly at the beginning of the episode.

-  Observations provide the location of the target and agent.
-  There are 4 actions in our environment, corresponding to the
   movements “right”, “up”, “left”, and “down”.
-  A done signal is issued as soon as the agent has navigated to the
   grid cell where the target is located.
-  Rewards are binary and sparse, meaning that the immediate reward is
   always zero, unless the agent has reached the target, then it is 1.

An episode in this environment (with ``size=5``) might look like this:

where the blue dot is the agent and the red square represents the
target.

Let us look at the source code of ``GridWorldEnv`` piece by piece:
"""

# %%
# Declaration and Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our custom environment will inherit from the abstract class
# ``gymnasium.Env``. You shouldn’t forget to add the ``metadata``
# attribute to your class. There, you should specify the render-modes that
# are supported by your environment (e.g. ``"human"``, ``"rgb_array"``,
# ``"ansi"``) and the framerate at which your environment should be
# rendered. Every environment should support ``None`` as render-mode; you
# don’t need to add it in the metadata. In ``GridWorldEnv``, we will
# support the modes “rgb_array” and “human” and render at 4 FPS.
#
# The ``__init__`` method of our environment will accept the integer
# ``size``, that determines the size of the square grid. We will set up
# some variables for rendering and define ``self.observation_space`` and
# ``self.action_space``. In our case, observations should provide
# information about the location of the agent and target on the
# 2-dimensional grid. We will choose to represent observations in the form
# of dictionaries with keys ``"agent"`` and ``"target"``. An observation
# may look like ``{"agent": array([1, 0]), "target": array([0, 3])}``.
# Since we have 4 actions in our environment (“right”, “up”, “left”,
# “down”), we will use ``Discrete(4)`` as an action space. Here is the
# declaration of ``GridWorldEnv`` and the implementation of ``__init__``:

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class CarTrackEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.track_length = 10
        self.track_width = 3
        self.agent_fov = 3 # side len of the squares in which agent will see car_ids and car_actions

        '''
        actions:
            0: stay still
            1: go straight 1 unit
            2: go up 1 unit then left 1 unit
            3: go up 1 unit then right 1 unit
            4: go left 1 unit
            5: go right 1 unit
            6: go left 1 unit and then up 1 unit
            7: go right 1 unit and then up 1 unit
        . action priority will also be in this order with most priority at 0.
        '''
        self.action_space = spaces.Discrete(8)

        '''
        --------
            x(00,00,00)x
		++++x(00,00,00)x++++    <- goal line
		    x(00,00,00)x
		    x(00,10,10)x
		    x(00,00,00)x <- dummy car turns left once more here
		    x(00,00,00)x <- dummy car turns left here
		    x(10,10,00)x
		    x(00,00,00)x
	    ----x(00,20,11)x----    <- starting line
		    x(00,00,00)x
        -------
        Map above is complete starting state of the track; agent can only see within its field of view (fov).
        . car_ids: car positions where
            0: no car
            1: non-agent car
            2: agent car
        . car_actions: actions of non-agent car that indicates next step behavior; agent car has a dummy 0
            (shown in action_space def)
        . goal_info: 2 goal-related variable that agent needs to know
            dist_to_goal: track length distance between agent and goal line
            goal_lane: the lane_id that agent needs to land on when it hits the goal line.
                0: left-most lane
                1: middle lane
                2: right-most lane
        '''
        self.observation_space = spaces.Dict(
            {
                "car_ids": spaces.Box(0, 2, shape=(self.agent_fov**2,), dtype=int),
                "car_actions": spaces.Box(0, 7, shape=(self.agent_fov**2,), dtype=int),
                "goal_info": spaces.Box(0, np.array([7, 2]), shape=(2,), dtype=int)
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # current state info for entire track; to be intialized in reset()
        self.id_state = np.array([[]])      # car ids matrix
        self.action_state = np.array([[]])  # car actions matrix
        self.dist_to_goal = -1              # track-len distance bw agent and goal line
        self.goal_lane_id = -1              # lane id agent is supposed to lane on when it reaches goal line.

        # single crash coordinate info
        self.crash_coor = ()



    def _get_obs(self):
        agent_pos = self.dist_to_goal + 1
        return {
            "car_ids": self.id_state[agent_pos-1:agent_pos+2].flatten(),
            "car_actions": self.action_state[agent_pos-1:agent_pos+2].flatten(),
            "goal_info": np.array([self.dist_to_goal, self.goal_lane_id])
        }

    def _get_info(self):
        # return the track coordinate (row, column) of the crash where (0,0) is top left.
        return {"crash_coordinate": self.crash_coor}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Fix goal_lane and dist_to_goal
        self.goal_lane_id = 2
        self.dist_to_goal = 7

        # intialize car ids and actions state.
        # self.id_state = np.array(
        #     [[0,0,0]] * 3 + [[0,1,1]] + [[0,0,0]] * 2 + [[1,1,0]] + [[0,0,0]] + [[1,2,1]] + [[1,1,1]]
        # )
        # self.action_state = np.array(
        #     [[0,0,0]] * 8 + [[1,0,1]] + [[1,1,1]]
        # )
        self.id_state = np.array(
            [[0,0,0]] * 3 + [[0,1,1]] + [[0,0,0]] * 2 + [[1,1,0]] + [[0,0,0]] + [[0,2,1]] + [[0,0,0]]
        )
        self.action_state = np.array(
            [[0,0,0]] * 8 + [[0,0,1]] + [[0,0,0]]
        )

        #render
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    '''
    Get dumb AI action for the moving non-agent car
        . AI will ignore agent car (treat as if agent is not there), but the system will prevent AI from 
            crashing to another car or walls.
        . AI car can only go forward, left, or right.
        . AI tries to move forward if possible, then it tries to move in previous direction, then it tries to
            move at all, then it tries to keep previous action when stuck.
    Args: 
        (r,c):  car's current coordinate
        prev_a: car's previous action
        nam:    next_action_matrix
    Returns: Car's action for current coordinate
    '''

    def _get_dumbAI_action(self, r, c, prev_a, nam):
        if r == 0:
            ret_a = 1
        else:
            if self.id_state[r-1][c] != 1 or (self.id_state[r-1][c] == 1 and nam[r-1][c] != 0):
                # go forward if it can
                ret_a = 1
            else:
                # possible actions: left or right; now try to move and keep previous action

                left_possible = self._can_go_left(r, c, nam)
                right_possible = self._can_go_right(r, c, nam)
                if prev_a == 1:
                    if left_possible and right_possible:
                        if np.random.choice([0,1]) == 0:
                            ret_a = 4
                        else:
                            ret_a = 5
                    elif left_possible:
                        ret_a = 4
                    elif right_possible:
                        ret_a = 5
                    else:
                        # can't go in any 3 directions
                        ret_a = 1
                elif prev_a == 4:
                    if left_possible:
                        ret_a = 4
                    elif right_possible:
                        ret_a = 5
                    else:
                        ret_a = 4
                elif prev_a == 5:
                    if right_possible:
                        ret_a = 5
                    elif left_possible:
                        ret_a = 4
                    else:
                        ret_a = 5
                else:
                    # unknown prev_a
                    ret_a = prev_a

        return ret_a

    # returns if AI can go left without getting blocked by any non-agent car.
    # note: agent can only move up, left, right; and right has lower priority than left.
    # if there's a car blocking the agent's preferred moved, but the car is not stationary, then agent ignores
    #   that car's behavior and takes the chance that it could move to where it wants.
    def _can_go_left(self, r, c, nam):
        if c == 0:
            ret_bool = False
        elif c >= 1 and (self.id_state[r][c-1] == 1 and nam[r][c-1] == 0 and self.action_state[r][c-1] == 0 or 
                         r < self.track_length - 1 and self.id_state[r+1][c-1] == 1 and self.action_state[r+1][c-1] == 1):
            # c >= 1 -> check if there's a stationary car to left or there's a car on bottom left trying to move forward.
            ret_bool = False
        elif c >= 2 and (self.id_state[r][c-2] == 1 and self.action_state[r][c-2] == 5):
            # c >= 2 -> check if there's a car 2 unit to your left trying to go right.
            ret_bool = False
        else:
            ret_bool = True

        return ret_bool


    # returns if AI can go right without getting blocked by any non-agent car.
    # note: agent can only move up, left , right; and left has higher priority than right.
    # if there's a car blocking the agent's preferred move, but the car is not stationary, then agent ignores
    #   that car's behavior and takes the chance that it could move to where it wants.
    def _can_go_right(self, r, c, nam):
        max_c = self.track_width - 1
        if c == max_c:
            ret_bool = False
        elif c <= max_c - 1 and (self.id_state[r][c+1] == 1 and nam[r][c+1] == 0 and self.action_state[r][c+1] == 0 or 
                                 r < self.track_length - 1 and self.id_state[r+1][c+1] == 1 and self.action_state[r+1][c+1] == 1):
            # c >= 1 -> check if there's a stationary car to right or there's a car on bottom right trying to move forward.
            ret_bool = False
        elif c <= max_c - 2 and (self.id_state[r][c+2] == 1 and nam[r][c+2] == 4):
            # c >= 2 -> check if there's a car 2 unit to your right trying to go left.
            ret_bool = False
        else:
            ret_bool = True

        return ret_bool

    '''
    Perform cars' actions row by row from top down, then set these cars' actions to 0 and save its next
        action to be updated once all cars' actions have been performed.
    . In each row, cars' actions have same priority as action encodings with 0 having highest priority.
    . Non-agent cars are 'smart' in that if they would crash, they will just take action 0 (stay still) 
        instead, and keep their previous action as next action.
    . Reward: -100 if agent goes off track or crashes to other car, -10 if agent hits goal line at wrong lane,
        -1 for the rest.
    . Termination: if agent goes off track or crashes or if agent hits goal line.
    '''
    def step(self, action):
        terminated = False
        reward = -1         #default reward
        potential_crash_coor = () # potential crash location (no crash if it's non-agent car)
        potential_goal_lane = -1    # updated whenever agent makes one step closer to goal line

        # save next actions since current action state will be updated to 0.
        next_action_matrix = np.array([[0]* self.track_width] * self.track_length)

        # add agent car's action to action_state
        agent_pos = self.dist_to_goal + 1
        for c in range(self.track_width):
            if self.id_state[agent_pos][c] == 2:
                self.action_state[agent_pos][c] = action
                break

        # perform cars' actions (update car's id_state and dist_to_goal only)
        for r in range(self.track_length):
            # for each row
            while np.max(self.action_state[r]) != 0:
                # must still contain car's actions to be processed.

                # get highest priority non-0 action
                tmp_list = []
                for c in range(self.track_width):
                    action_value = self.action_state[r][c]
                    if action_value != 0:
                        tmp_list.append(action_value)
                min_action = min(tmp_list) # current action to prioritize

                # perform action
                for c in range(self.track_width):

                    # for more intuitive behavior, action 5 and 7 should be performed from right to left
                    if min_action == 5 or min_action == 7:
                        c = self.track_width - 1 - c

                    if self.action_state[r][c] != min_action:
                        # no car or car with action 0 or action of lower priority
                        continue
                    else:
                        # must be car_id at [r][c] with min_action

                        crashed = False
                        car_id = self.id_state[r][c]
                        car_action = self.action_state[r][c]

                        # remove current id, action of car from state
                        self.id_state[r][c] = 0
                        self.action_state[r][c] = 0

                        # min_action can't be 0 here
                        if min_action == 1:
                            # up 1 unit
                            if r == 0:
                                # non-agent car goes beyond the track -> car disappears from state
                                pass
                            elif self.id_state[r-1][c] == 1 or self.id_state[r-1][c] == 2:
                                # r >= 1; car is agent or non-agent
                                crashed = True
                                potential_crash_coor = (r-1, c)    
                            else:
                                # id[r-1][c] must be 0, aka no crash

                                # update car's new location
                                self.id_state[r-1][c] = car_id

                                if car_id == 2:
                                    self.dist_to_goal -= 1
                                    potential_goal_lane = c
                                else:
                                    # must be non-agent car (id=1) -> get and save car next action
                                    next_action_matrix[r-1][c] = self._get_dumbAI_action(r-1, c, car_action, next_action_matrix)
                        elif min_action == 2:
                            # up 1 then left 1 unit

                            if c == 0 and r == 0:
                                # crash to left wall at track end
                                crashed = True
                                potential_crash_coor = (r-1, c-1)
                            elif r == 0:
                                # goes beyond map
                                pass
                            elif self.id_state[r-1][c] == 1 or self.id_state[r-1][c] == 2:
                                # crash to car in front
                                crashed = True
                                potential_crash_coor = (r-1, c)
                            elif c == 0:
                                # crash to left wall
                                crashed = True
                                potential_crash_coor = (r-1, c-1)
                            elif self.id_state[r-1][c-1] == 1 or self.id_state[r-1][c-1] == 2:
                                # crash to car on top left
                                crashed = True
                                potential_crash_coor = (r-1, c-1)
                            else:
                                # no crash

                                # update car's new location
                                self.id_state[r-1][c-1] = car_id

                                if car_id == 2:
                                    self.dist_to_goal -= 1
                                    potential_goal_lane = c - 1
                                else:
                                    # must be non-agent car (id=1) -> get and save car next action
                                    next_action_matrix[r-1][c-1] = self._get_dumbAI_action(r-1, c-1, car_action, next_action_matrix)

                        elif min_action == 3:
                            # up 1 then right 1 unit

                            if c == self.track_width - 1 and r == 0:
                                # crash to right wall at track end
                                crashed = True
                                potential_crash_coor = (r-1, c+1)
                            elif r == 0:
                                # goes beyond map
                                pass
                            elif self.id_state[r-1][c] == 1 or self.id_state[r-1][c] == 2:
                                # crash to car in front
                                crashed = True
                                potential_crash_coor = (r-1, c)
                            elif c == self.track_width - 1:
                                # crash to right wall
                                crashed = True
                                potential_crash_coor = (r-1, c+1)
                            elif self.id_state[r-1][c+1] == 1 or self.id_state[r-1][c+1] == 2:
                                # crash to car on top right
                                crashed = True
                                potential_crash_coor = (r-1, c+1)
                            else:
                                # no crash

                                # update car's new location
                                self.id_state[r-1][c+1] = car_id

                                if car_id == 2:
                                    self.dist_to_goal -= 1
                                    potential_goal_lane = c + 1
                                else:
                                    # must be non-agent car (id=1) -> get and save car next action
                                    next_action_matrix[r-1][c+1] = self._get_dumbAI_action(r-1, c+1, car_action, next_action_matrix)

                        elif min_action == 4:
                            # left 1 unit

                            if c == 0:
                                # crash to left wall
                                crashed = True
                                potential_crash_coor = (r, c-1)
                            elif self.id_state[r][c-1] == 1 or self.id_state[r][c-1] == 2:
                                # crash to car on left
                                crashed = True
                                potential_crash_coor = (r, c-1)
                            else:
                                # no crash

                                # update car's new location
                                self.id_state[r][c-1] = car_id

                                if car_id == 2:
                                    pass
                                else:
                                    # must be non-agent car (id=1) -> get and save car next action
                                    next_action_matrix[r][c-1] = self._get_dumbAI_action(r, c-1, car_action, next_action_matrix)

                        elif min_action == 5:
                            # right 1 unit

                            if c == self.track_width - 1:
                                # crash to right wall
                                crashed = True
                                potential_crash_coor = (r, c+1)
                            elif self.id_state[r][c+1] == 1 or self.id_state[r][c+1] == 2:
                                # crash to car on right
                                crashed = True
                                potential_crash_coor = (r, c+1)
                            else:
                                # no crash

                                # update car's new location
                                self.id_state[r][c+1] = car_id

                                if car_id == 2:
                                    pass
                                else:
                                    # must be non-agent car (id=1) -> get and save car next action
                                    next_action_matrix[r][c+1] = self._get_dumbAI_action(r, c+1, car_action, next_action_matrix)
                        elif min_action == 6:
                            # left 1 then up 1 unit

                            if c == 0:
                                # crash to left wall
                                crashed = True
                                potential_crash_coor = (r, c-1)
                            elif self.id_state[r][c-1] == 1 or self.id_state[r][c-1] == 2:
                                # crash to car on left
                                crashed = True
                                potential_crash_coor = (r, c-1)
                            elif r == 0:
                                # non-agent car goes beyond map -> disappears from state
                                pass
                            elif self.id_state[r-1][c-1] == 1 or self.id_state[r-1][c-1] == 2:
                                # crash to car on top left
                                crashed = True
                                potential_crash_coor = (r-1, c-1)
                            else:
                                # no crash

                                # update car's new location
                                self.id_state[r-1][c-1] = car_id

                                if car_id == 2:
                                    self.dist_to_goal -= 1
                                    potential_goal_lane = c - 1
                                else:
                                    # must be non-agent car (id=1) -> get and save car next action
                                    next_action_matrix[r-1][c-1] = self._get_dumbAI_action(r-1, c-1, car_action, next_action_matrix)
                        else:
                            # min_action == 7
                            # right 1 then up 1 unit

                            if c == self.track_width - 1:
                                # crash to right wall
                                crashed = True
                                potential_crash_coor = (r, c+1)
                            elif self.id_state[r][c+1] == 1 or self.id_state[r][c+1] == 2:
                                # crash to car on right
                                crashed = True
                                potential_crash_coor = (r, c+1)
                            elif r == 0:
                                # non-agent car goes beyond map -> disappears from state
                                pass
                            elif self.id_state[r-1][c+1] == 1 or self.id_state[r-1][c+1] == 2:
                                # crash to car on top right
                                crashed = True
                                potential_crash_coor = (r-1, c+1)
                            else:
                                # no crash

                                # update car's new location
                                self.id_state[r-1][c+1] = car_id

                                if car_id == 2:
                                    self.dist_to_goal -= 1
                                    potential_goal_lane = c + 1
                                else:
                                    # must be non-agent car (id=1) -> get and save car next action
                                    next_action_matrix[r-1][c+1] = self._get_dumbAI_action(r-1, c+1, car_action, next_action_matrix)

                        # if crashed car is non-agent -> 'smart' car
                        # if it is agent -> update terminated and reward
                        if crashed == True:
                            if car_id == 1:
                                # car discards current action and stays idle (action 0)
                                self.id_state[r][c] = 1
                                next_action_matrix[r][c] = car_action
                            elif car_id == 2:
                                terminated = True
                                reward = -100
                                self.crash_coor = potential_crash_coor

                                # crashed cars disappear from map
                                if potential_crash_coor[1] < 0 or potential_crash_coor[1] >= self.track_width:
                                    # crashed to wall
                                    pass
                                else:
                                    # must have crashed to car.
                                    self.id_state[potential_crash_coor[0]][potential_crash_coor[1]] = 0
                            else:
                                print("Unknown crashed car's id")
                                exit()


                        # episode termination by agent reaching goal line -> update terminated and reward
                        if terminated == False and self.dist_to_goal == 0:

                            # car_id must be 2 here since once terminated == True, it can't be False again
                            terminated = True

                            if potential_goal_lane == self.goal_lane_id:
                                # default reward of -1
                                pass
                            else:
                                # agent goal lane must not match required goal lane
                                reward = -10

        # renew next cars' actions (update car's action state)
        self.action_state = next_action_matrix

        # render
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        # if self.render_mode == "human":
        #     return self._render_frame()

        pass

    def _render_frame(self):

        for r in range(self.track_length):

            if r == 1:
                print("++++x ", end="")
            elif r == self.track_length - 2:
                print("----x ", end="")
            else:
                print("    x ", end="")

            for c in range(self.track_width):
                print(f'{self.id_state[r][c]}{self.action_state[r][c]} ', end="")

            if r == 1:
                print("x++++    <- Goal Line")
            elif r == self.track_length - 2:
                print("x----    <- Start Line")
            else:
                print("x    ")

        print()

    def close(self):
        pass



class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

# %%
# Constructing Observations From Environment States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since we will need to compute observations both in ``reset`` and
# ``step``, it is often convenient to have a (private) method ``_get_obs``
# that translates the environment’s state into an observation. However,
# this is not mandatory and you may as well compute observations in
# ``reset`` and ``step`` separately:

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

# %%
# We can also implement a similar method for the auxiliary information
# that is returned by ``step`` and ``reset``. In our case, we would like
# to provide the manhattan distance between the agent and the target:

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

# %%
# Oftentimes, info will also contain some data that is only available
# inside the ``step`` method (e.g. individual reward terms). In that case,
# we would have to update the dictionary that is returned by ``_get_info``
# in ``step``.

# %%
# Reset
# ~~~~~
#
# The ``reset`` method will be called to initiate a new episode. You may
# assume that the ``step`` method will not be called before ``reset`` has
# been called. Moreover, ``reset`` should be called whenever a done signal
# has been issued. Users may pass the ``seed`` keyword to ``reset`` to
# initialize any random number generator that is used by the environment
# to a deterministic state. It is recommended to use the random number
# generator ``self.np_random`` that is provided by the environment’s base
# class, ``gymnasium.Env``. If you only use this RNG, you do not need to
# worry much about seeding, *but you need to remember to call
# ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
# correctly seeds the RNG. Once this is done, we can randomly set the
# state of our environment. In our case, we randomly choose the agent’s
# location and the random sample target positions, until it does not
# coincide with the agent’s position.
#
# The ``reset`` method should return a tuple of the initial observation
# and some auxiliary information. We can use the methods ``_get_obs`` and
# ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

# %%
# Step
# ~~~~
#
# The ``step`` method usually contains most of the logic of your
# environment. It accepts an ``action``, computes the state of the
# environment after applying that action and returns the 5-tuple
# ``(observation, reward, terminated, truncated, info)``. See
# :meth:`gymnasium.Env.step`. Once the new state of the environment has
# been computed, we can check whether it is a terminal state and we set
# ``done`` accordingly. Since we are using sparse binary rewards in
# ``GridWorldEnv``, computing ``reward`` is trivial once we know
# ``done``.To gather ``observation`` and ``info``, we can again make
# use of ``_get_obs`` and ``_get_info``:

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else -1  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

# %%
# Rendering
# ~~~~~~~~~
#
# Here, we are using PyGame for rendering. A similar approach to rendering
# is used in many environments that are included with Gymnasium and you
# can use it as a skeleton for your own environments:

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

# %%
# Close
# ~~~~~
#
# The ``close`` method should close any open resources that were used by
# the environment. In many cases, you don’t actually have to bother to
# implement this method. However, in our example ``render_mode`` may be
# ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# %%
# In other environments ``close`` might also close files that were opened
# or release other resources. You shouldn’t interact with the environment
# after having called ``close``.

# %%
# Registering Envs
# ----------------
#
# In order for the custom environments to be detected by Gymnasium, they
# must be registered as follows. We will choose to put this code in
# ``gym-examples/gym_examples/__init__.py``.
#
# .. code:: python
#
#   from gymnasium.envs.registration import register
#
#   register(
#        id="gym_examples/GridWorld-v0",
#        entry_point="gym_examples.envs:GridWorldEnv",
#        max_episode_steps=300,
#   )

# %%
# The environment ID consists of three components, two of which are
# optional: an optional namespace (here: ``gym_examples``), a mandatory
# name (here: ``GridWorld``) and an optional but recommended version
# (here: v0). It might have also been registered as ``GridWorld-v0`` (the
# recommended approach), ``GridWorld`` or ``gym_examples/GridWorld``, and
# the appropriate ID should then be used during environment creation.
#
# The keyword argument ``max_episode_steps=300`` will ensure that
# GridWorld environments that are instantiated via ``gymnasium.make`` will
# be wrapped in a ``TimeLimit`` wrapper (see `the wrapper
# documentation </api/wrappers>`__ for more information). A done signal
# will then be produced if the agent has reached the target *or* 300 steps
# have been executed in the current episode. To distinguish truncation and
# termination, you can check ``info["TimeLimit.truncated"]``.
#
# Apart from ``id`` and ``entrypoint``, you may pass the following
# additional keyword arguments to ``register``:
#
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | Name                 | Type      | Default   | Description                                                                                                   |
# +======================+===========+===========+===============================================================================================================+
# | ``reward_threshold`` | ``float`` | ``None``  | The reward threshold before the task is  considered solved                                                    |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``nondeterministic`` | ``bool``  | ``False`` | Whether this environment is non-deterministic even after seeding                                              |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``max_episode_steps``| ``int``   | ``None``  | The maximum number of steps that an episode can consist of. If not ``None``, a ``TimeLimit`` wrapper is added |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``order_enforce``    | ``bool``  | ``True``  | Whether to wrap the environment in an  ``OrderEnforcing`` wrapper                                             |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``autoreset``        | ``bool``  | ``False`` | Whether to wrap the environment in an ``AutoResetWrapper``                                                    |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``kwargs``           | ``dict``  | ``{}``    | The default kwargs to pass to the environment class                                                           |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
#
# Most of these keywords (except for ``max_episode_steps``,
# ``order_enforce`` and ``kwargs``) do not alter the behavior of
# environment instances but merely provide some extra information about
# your environment. After registration, our custom ``GridWorldEnv``
# environment can be created with
# ``env = gymnasium.make('gym_examples/GridWorld-v0')``.
#
# ``gym-examples/gym_examples/envs/__init__.py`` should have:
#
# .. code:: python
#
#    from gym_examples.envs.grid_world import GridWorldEnv
#
# If your environment is not registered, you may optionally pass a module
# to import, that would register your environment before creating it like
# this - ``env = gymnasium.make('module:Env-v0')``, where ``module``
# contains the registration code. For the GridWorld env, the registration
# code is run by importing ``gym_examples`` so if it were not possible to
# import gym_examples explicitly, you could register while making by
# ``env = gymnasium.make('gym_examples:gym_examples/GridWorld-v0)``. This
# is especially useful when you’re allowed to pass only the environment ID
# into a third-party codebase (eg. learning library). This lets you
# register your environment without needing to edit the library’s source
# code.

# %%
# Creating a Package
# ------------------
#
# The last step is to structure our code as a Python package. This
# involves configuring ``gym-examples/setup.py``. A minimal example of how
# to do so is as follows:
#
# .. code:: python
#
#    from setuptools import setup
#
#    setup(
#        name="gym_examples",
#        version="0.0.1",
#        install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
#    )
#
# Creating Environment Instances
# ------------------------------
#
# After you have installed your package locally with
# ``pip install -e gym-examples``, you can create an instance of the
# environment via:
#
# .. code:: python
#
#    import gym_examples
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#
# You can also pass keyword arguments of your environment’s constructor to
# ``gymnasium.make`` to customize the environment. In our case, we could
# do:
#
# .. code:: python
#
#    env = gymnasium.make('gym_examples/GridWorld-v0', size=10)
#
# Sometimes, you may find it more convenient to skip registration and call
# the environment’s constructor yourself. Some may find this approach more
# pythonic and environments that are instantiated like this are also
# perfectly fine (but remember to add wrappers as well!).
#
# Using Wrappers
# --------------
#
# Oftentimes, we want to use different variants of a custom environment,
# or we want to modify the behavior of an environment that is provided by
# Gymnasium or some other party. Wrappers allow us to do this without
# changing the environment implementation or adding any boilerplate code.
# Check out the `wrapper documentation </api/wrappers/>`__ for details on
# how to use wrappers and instructions for implementing your own. In our
# example, observations cannot be used directly in learning code because
# they are dictionaries. However, we don’t actually need to touch our
# environment implementation to fix this! We can simply add a wrapper on
# top of environment instances to flatten observations into a single
# array:
#
# .. code:: python
#
#    import gym_examples
#    from gymnasium.wrappers import FlattenObservation
#
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#    wrapped_env = FlattenObservation(env)
#    print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}
#
# Wrappers have the big advantage that they make environments highly
# modular. For instance, instead of flattening the observations from
# GridWorld, you might only want to look at the relative position of the
# target and the agent. In the section on
# `ObservationWrappers </api/wrappers/#observationwrapper>`__ we have
# implemented a wrapper that does this job. This wrapper is also available
# in gym-examples:
#
# .. code:: python
#
#    import gym_examples
#    from gym_examples.wrappers import RelativePosition
#
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#    wrapped_env = RelativePosition(env)
#    print(wrapped_env.reset())     # E.g.  [-3  3], {}
