import numpy as np
from gym import spaces
from gym import Env

from . import register_env

# Mod from point_robot.py
@register_env('base-gridworld')
class BaseGridWorldEnv(Env):
    """
    control agent to move to a goal
    tasks (aka goals) are posisiton on the grid
    Basic gridworld, no windy, no obstacle, deterministic environment, start at (0,0)
    Grid with origin at top-left.
    State: [x,y] current coordinate of the agent
    Observation: current location and Mahattan distance to goal
    Action: 0-U, 1-R, 2-D, 3-L

     - tasks sampled uniformly for mxn grid
     - reward is 1 at goal, 0 elsewhere
    """

    def __init__(self, randomize_tasks=False, n_tasks=2, grid_size=(5,5)):
        self.grid_size = grid_size
        if randomize_tasks:
            np.random.seed(1337)
            # goals = [np.array([np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])]) for _ in range(n_tasks)] #Not unique
            goals_id = np.random.shuffle(np.arange(self.grid_size[0]*self.grid_size[1]))[:n_tasks]
            goals = [np.array([id//self.grid_size[0], id%self.grid_size[0]]) for id in goals_id]
        else:
            # some hand-coded goals for debugging. Optimal solution took 4 steps to any goal
            goals = [np.array([3, 1]),
                     np.array([1, 3]),
                     np.array([2, 2]),
                     np.array([4, 0]),
                     np.array([0, 4])
                     ]
        self.goals = goals
        self.reset_task(0)
        self.observation_space = spaces.Box(low=0, high=self.grid_size[0]+self.grid_size[1], shape=(3,), dtype=np.uint8) # If m != n => remember to check edge
        # self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8) # use spaces.Box instead of Discrete to easily integrate with current prj

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the grid
        # self._state = np.array([np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])])
        # reset to origin
        self._state = np.array([0, 0])
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        Mahattan_distance = np.abs(self._goal[0] - self._state[0]) + np.abs(self._goal[1] - self._state[1])
        return np.array([self._state[0], self._state[1], Mahattan_distance])
        # return np.copy(self._state)

    def _is_moveable(self, action):
        ''' not move if out of edges '''
        # override this when implement obstacle
        if action == 0 and self._state[1]==0:
            return False
        if action == 1 and self._state[0]==self.grid_size[0]-1:
            return False
        if action == 2 and self._state[1]==self.grid_size[1]-1:
            return False
        if action == 3 and self._state[0]==0:
            return False
        return True

    def step(self, action):
        action = int(np.rint(action))
        #Action: 0-U, 1-R, 2-D, 3-L
        if self._is_moveable(action):
            if action == 0:
                self._state = self._state + np.array([0,-1])
            elif action == 1:
                self._state = self._state + np.array([1,0])
            elif action == 2:
                self._state = self._state + np.array([0,1])
            elif action == 3:
                self._state = self._state + np.array([-1,0])
        reward = int(np.array_equal(self._goal, self._state))
        # reward = -self._get_obs()[-1]
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)
        #TODO: print the grid
        for j in range(self.grid_size[1]):
            for i in range(self.grid_size[0]):
                for potential_goal in self.goals:
                    if i==potential_goal[0] and j==potential_goal[1]:
                        c='P'
                if i==self._goal[0] and j==self._goal[1]:
                    c='G'
                if i==self._state[0] and j==self._state[1]:
                    c='X'
                print(" %s |"%c, end = '')
            print("\n")
            for i in range(self.grid_size[0]):
                print("----", end = '')
            print("\n")
