import numpy as np
# import pygame

import gymnasium as gym
from gymnasium import spaces

'''
Full observability for agent; no moving car.
rewards: -200 crashing, + 200 if hit goal lane at correct lane; reward is -1 otherwise
Termination: if crashing or hitting goal line.
'''


class CarTrack1Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.track_length = 10
        self.track_width = 3
        # self.agent_fov = 3 # side len of the squares in which agent will see car_ids and car_actions
        # chances of non-agent car taking a non-optimal action (optimal action determined by dumbAI algo).
        self.rand_action_prob = 0

        self.max_epi_len = 150
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
        . Cars' actions are prioritized from top row to bottom. On the same row, action priority will 
            be in this order with most priority at 0.
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
                "car_ids": spaces.Box(0, 2, shape=(self.track_length * self.track_width,), dtype=int),
                "car_actions": spaces.Box(0, 7, shape=(self.track_length * self.track_width,), dtype=int),
                "goal_info": spaces.Box(0, np.array([7, 2]), shape=(2,), dtype=int)
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # current state info for entire track; to be intialized in reset()
        self.id_state = np.array([[]])      # car ids matrix
        self.action_state = np.array([[]])  # car actions matrix
        self.dist_to_goal = -1              # track-len distance bw agent and goal line
        # lane id agent is supposed to lane on when it reaches goal line.
        self.goal_lane_id = -1

        # single crash coordinate info
        self.crash_coor = ()

    def _get_obs(self):
        agent_pos = self.dist_to_goal + 1
        return {
            "car_ids": self.id_state.flatten(),
            "car_actions": self.action_state.flatten(),
            "goal_info": np.array([self.dist_to_goal, self.goal_lane_id])
        }

    def _get_info(self):
        # return the track coordinate (row, column) of the crash where (0,0) is top left.
        return {"crash_coordinate": self.crash_coor}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_len = 0
        self.crash_coor = ()

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
        # full cars
        # self.id_state = np.array(
        #     [[0, 0, 0]] * 3 + [[0, 1, 1]] + [[0, 0, 0]] * 2 +
        #     [[1, 1, 0]] + [[0, 0, 0]] + [[1, 2, 1]] + [[1, 1, 1]]
        # )
        # self.action_state = np.array(
        #     [[0, 0, 0]] * 8 + [[1, 0, 1]] + [[1, 1, 1]]
        # )
        # debug
        # self.id_state = np.array(
        #     [[0, 0, 0]] * 3 + [[0, 1, 1]] + [[0, 0, 0]] * 2 +
        #     [[1, 1, 0]] + [[0, 0, 0]] + [[2, 1, 1]] + [[0, 0, 0]]
        # )
        # self.action_state = np.array(
        #     [[0, 0, 0]] * 7 + [[0, 0, 0]] + [[0, 5, 5]] + [[0, 0, 0]]
        # )

        '''
        Variable goal lane (on r=1), random location of 2 stationary cars on r=3 and r=6, random 
            location of agent on r=8, 3 random moving cars on r=7,8,9.
        '''
        # self.id_state = np.array([[0]*self.track_width]*self.track_length)
        # self.action_state = np.array([[0]*self.track_width]*self.track_length)
        # self.goal_lane_id = np.random.choice(self.track_width)
        # for i in np.random.choice(self.track_width, 2, replace=False):
        #     self.id_state[3][i] = 1
        # for i in np.random.choice(self.track_width, 2, replace=False):
        #     self.id_state[6][i] = 1
        # agent_col = np.random.choice(self.track_width)
        # self.id_state[8][agent_col] = 2
        # for i in np.random.choice(8, 3, replace=False):
        #     if i >= 3 + agent_col:
        #         i += 1
        #     if i < 3:
        #         self.id_state[7][i] = 1
        #         self.action_state[7][i] = 1
        #     elif i >= 3 and i < 6:
        #         self.id_state[8][i % self.track_width] = 1
        #         self.action_state[8][i % self.track_width] = 1
        #     else:
        #         # i must be 6,7, or 8
        #         self.id_state[9][i % self.track_width] = 1
        #         self.action_state[9][i % self.track_width] = 1

        # variable goal lane only
        # self.goal_lane_id = np.random.choice(self.track_width)
        # self.id_state = np.array(
        #     [[0, 0, 0]] * 3 + [[0, 1, 1]] + [[0, 0, 0]] * 2 +
        #     [[1, 1, 0]] + [[0, 0, 0]] + [[0, 2, 0]] + [[0, 0, 0]]
        # )
        # self.action_state = np.array(
        #     [[0, 0, 0]] * 7 + [[0, 0, 0]] + [[0, 0, 0]] + [[0, 0, 0]]
        # )

        # variation: goal lane, and stationary car in both rows
        # self.id_state = np.array([[0]*self.track_width]*self.track_length)
        # self.action_state = np.array([[0]*self.track_width]*self.track_length)
        # self.goal_lane_id = np.random.choice(self.track_width)
        # for i in np.random.choice(self.track_width, 2, replace=False):
        #     self.id_state[3][i] = 1
        # for i in np.random.choice(self.track_width, 2, replace=False):
        #     self.id_state[6][i] = 1
        # self.id_state[8][1] = 2

        # goal_lane variation + non-agent car at (8,1) moving deterministically.
        # self.goal_lane_id = np.random.choice(self.track_width)
        # self.id_state = np.array(
        #     [[0, 0, 0]] * 3 + [[0, 1, 1]] + [[0, 0, 0]] * 2 +
        #     [[1, 1, 0]] + [[0, 0, 0]] + [[0, 2, 1]] + [[0, 0, 0]]
        # )
        # self.action_state = np.array(
        #     [[0, 0, 0]] * 7 + [[0, 0, 0]] + [[0, 0, 1]] + [[0, 0, 0]]
        # )

        # goal_lane variation + 2 non-agent car at random around agent car, moving deterministically.
        self.id_state = np.array(
            [[0, 0, 0]]*3 + [[0, 1, 1]] + [[0, 0, 0]]*2 + [[1, 1, 0]]+[[0, 0, 0]]*3)
        self.action_state = np.array([[0]*self.track_width]*self.track_length)
        self.goal_lane_id = np.random.choice(self.track_width)
        agent_col = np.random.choice(self.track_width)  # agent_col = 1
        self.id_state[8][agent_col] = 2
        for i in np.random.choice(8, 2, replace=False):
            if i >= 3 + agent_col:
                i += 1
            if i < 3:
                self.id_state[7][i] = 1
                self.action_state[7][i] = 1
            elif i >= 3 and i < 6:
                self.id_state[8][i % self.track_width] = 1
                self.action_state[8][i % self.track_width] = 1
            else:
                # i must be 6,7, or 8
                self.id_state[9][i % self.track_width] = 1
                self.action_state[9][i % self.track_width] = 1

        # test compare
        # self.goal_lane_id = 1
        # self.id_state = np.array(
        #     [[0, 0, 0]] * 3 + [[0, 1, 1]] + [[0, 0, 0]] * 2 +
        #     [[1, 1, 0]] + [[0, 1, 0]] + [[0, 2, 0]] + [[0, 0, 1]]
        # )
        # self.action_state = np.array(
        #     [[0, 0, 0]] * 7 + [[0, 1, 0]] + [[0, 0, 0]] + [[0, 0, 1]]
        # )

        # render
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
                        if np.random.choice([0, 1]) == 0:
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

        # random action
        if np.random.rand() < self.rand_action_prob:
            a_list = [1, 4, 5]
            a_list.remove(prev_a)
            ret_a = a_list[np.random.choice(2)]

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
        truncated = False
        terminated = False
        crashed = False
        # reward should be set in loop or crashed or terminated
        reward = -1  # -self.dist_to_goal
        potential_crash_coor = ()  # potential crash location (no crash if it's non-agent car)
        potential_goal_lane = -1    # updated whenever agent makes one step closer to goal line

        # save next actions since current action state will be updated to 0.
        next_action_matrix = np.array(
            [[0] * self.track_width] * self.track_length)

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
                min_action = min(tmp_list)  # current action to prioritize

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
                                    next_action_matrix[r-1][c] = self._get_dumbAI_action(
                                        r-1, c, car_action, next_action_matrix)
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
                                    next_action_matrix[r-1][c-1] = self._get_dumbAI_action(
                                        r-1, c-1, car_action, next_action_matrix)

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
                                    next_action_matrix[r-1][c+1] = self._get_dumbAI_action(
                                        r-1, c+1, car_action, next_action_matrix)

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
                                    next_action_matrix[r][c-1] = self._get_dumbAI_action(
                                        r, c-1, car_action, next_action_matrix)

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
                                    next_action_matrix[r][c+1] = self._get_dumbAI_action(
                                        r, c+1, car_action, next_action_matrix)
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
                                    next_action_matrix[r-1][c-1] = self._get_dumbAI_action(
                                        r-1, c-1, car_action, next_action_matrix)
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
                                    next_action_matrix[r-1][c+1] = self._get_dumbAI_action(
                                        r-1, c+1, car_action, next_action_matrix)

                        # update new reward if aagent car took a non-0 action
                        # if car_id == 2:
                        #     reward = -self.dist_to_goal

                        # if crashed car is non-agent -> 'smart' car
                        # if it is agent -> update terminated and reward
                        if crashed == True:
                            if car_id == 1:
                                # car discards current action and stays idle (action 0)
                                self.id_state[r][c] = 1
                                next_action_matrix[r][c] = self._get_dumbAI_action(
                                    r, c, car_action, next_action_matrix)
                            elif car_id == 2:
                                terminated = True
                                reward = -200
                                self.crash_coor = potential_crash_coor

                                # crashed cars disappear from map
                                if potential_crash_coor[1] < 0 or potential_crash_coor[1] >= self.track_width:
                                    # crashed to wall
                                    pass
                                else:
                                    # must have crashed to car.
                                    self.id_state[potential_crash_coor[0]
                                                  ][potential_crash_coor[1]] = 0
                                    # self.action_state[potential_crash_coor[0]][potential_crash_coor[1]] = 0
                            else:
                                print(
                                    f'car_id,r,c,min_action,crashed,terminated: {car_id,r,c,min_action,crashed,terminated}')
                                print(f'id_state: {self.id_state}')
                                print(f'action_state: {self.action_state}')
                                print(f'nam: {next_action_matrix}')
                                print("Unknown crashed car's id")
                                print(self)
                                exit()

                        # episode termination by agent reaching goal line -> update terminated and reward
                        if terminated == False and self.dist_to_goal == 0:

                            # car_id must be 2 here since once terminated == True, it can't be False again
                            terminated = True

                            if potential_goal_lane == self.goal_lane_id:
                                # default reward of -1
                                reward = 200
                            else:
                                # agent goal lane must not match required goal lane
                                # dist_to_goal reward
                                pass

                    if crashed and terminated:
                        # must be agent car crashing -> freeze state and return
                        self.action_state += next_action_matrix
                        if self.render_mode == "human":
                            self._render_frame()
                        return self._get_obs(), reward, terminated, False, self._get_info()

        # renew next cars' actions (update car's action state)
        self.action_state = next_action_matrix

        # truncate episode if it's too long (to prevent too large negative reward build-up)
        self.episode_len += 1
        if self.episode_len == self.max_epi_len:
            truncated = True

        # render
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

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
                print(
                    f'{self.id_state[r][c]}{self.action_state[r][c]} ', end="")

            if r == 1:
                print("x++++    <- Goal Line")
            elif r == self.track_length - 2:
                print("x----    <- Start Line")
            else:
                print("x    ")

        print()

    def close(self):
        pass
