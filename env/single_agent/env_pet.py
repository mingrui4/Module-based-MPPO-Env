import numpy as np
import random
from gymnasium.utils import EzPickle
from scipy import stats

from env.env_utils.core import Box, World, Control
from env.env_utils.scenario import BaseScenario
from env.env_utils.simple_env import SimpleEnv, make_env

from pettingzoo.utils.conversions import parallel_wrapper_fn

ACTION_COST, FINISH_REWARD, GOAL_REWARD, COLLISION_REWARD, WAIT_REWARD, DEST_REWARD = -0.3, 30, 0, -5, -0.5, -0.3
EDGE = 1


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, agents, width, height, baseline=False, add_box=False, max_cycles=40, continuous_actions=False,
                 render_mode=None, vector_state=True):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            vector_state=vector_state,
            agents=agents,
            width=width,
            height=height,
            baseline=baseline,
            add_box=add_box
        )
        scenario = Scenario()
        world = scenario.make_world(agents, width, height, baseline, add_box)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.vector_state = vector_state
        self.metadata["name"] = "env_box"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self):
        self.num_box = 0
        self.start = []
        self.end = []
        self.colors = [
            np.array([200, 100, 100]),
            np.array([100, 100, 200]),
            np.array([100, 200, 100]),
            np.array([200, 200, 100]),
            np.array([100, 200, 200]),
        ]

    def make_world(self, agents, width, height, baseline, add_box):
        world = World()
        world.width = width
        world.height = height
        world.grid = np.zeros((world.width, world.height))
        world.baseline = baseline
        world.add_box = add_box

        self.num_box = agents['box']
        world.agents = [Control() for _ in range(agents['control'])]
        for i, agent in enumerate(world.agents):
            agent.name = agent.type + "_" + str(i)
            agent.num = int(world.width * world.height / len(world.agents))

        return world

    def reset_world(self, world, np_random):
        world.grid = np.zeros((world.width, world.height))
        world.boundary = []

        # random entrance and exit
        # self.edge = np.concatenate((np.array(range(0, world.width-1)), np.array(range(1, world.height-1)) * world.width,
        #                             np.array(range(2, world.width)) * world.height - 1,
        #                             np.array(range(1, world.height)) + world.width * (world.height - 1)))
        # random_indices = np_random.choice(range(len(self.edge)), len(self.edge) // 2, replace=False)
        # self.start = np.sort(np.array([self.edge[i] for i in random_indices]))
        # self.end = np.array([self.edge[i] for i in range(len(self.edge)) if i not in random_indices])

        # edge as the entrance and exit
        self.start = np.concatenate(
            (np.array(range(0, world.width - 1)), np.array(range(1, world.height - 1)) * world.width))
        self.end = np.concatenate((np.array(range(2, world.width)) * world.height - 1,
                                   np.array(range(1, world.height)) + world.width * (world.height - 1)))

        # 每次reset重置box数量
        world.boxes = [Box() for _ in range(self.num_box)]
        for b, box in enumerate(world.boxes):
            box.name = box.type + "_" + str(b)
            box.index = b + 1
            box.color = (self.colors[b % len(self.colors)] + b // len(self.colors) * 50) % 256

        box_buffer = np_random.choice(self.start, size=len(world.boxes), replace=False).tolist()
        dest_buffer = np_random.choice(self.end, size=len(world.boxes)).tolist()

        for b, box in enumerate(world.boxes):
            box.state.p_pos = np.array([box_buffer[b] // world.width, box_buffer[b] % world.width], dtype=int)
            box.state_last.p_pos = box.state.p_pos
            box.dest = np.array([dest_buffer[b] // world.width, dest_buffer[b] % world.width], dtype=int)
            box.route = 0
            box.path = [box.state.p_pos]
            box.finished = False

        for agent in world.agents:
            agent.state.p_pos = [0, 0]
            agent.cost = 0

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return self.reward([], agent, world)

    def global_reward(self, world):
        all_rewards = sum(self.reward([], agent, world) for agent in world.agents)
        return all_rewards

    def reward(self, action_status, agent, world):
        rewards = 0
        for status in action_status:
            if status == 1:
                rewards += ACTION_COST
            elif status == 2:
                rewards += DEST_REWARD
            elif status == -3:
                rewards += WAIT_REWARD
            elif status == -1 or status == -2:
                rewards += COLLISION_REWARD
        if self.check_for_done(world):
            rewards += FINISH_REWARD
        return rewards

    def check_for_done(self, world):
        finish_box = 0
        for box in world.boxes:
            if box.finished:
                finish_box += 1
        for box in world.boxes:
            if not box.finished:
                return False
        return True

    def observation(self, agent, world):
        obs = np.zeros((world.width, world.height))
        obs_goal = np.zeros((world.width, world.height))
        goal_dict = {}
        for i, goal in enumerate(self.end):
            x, y = goal // world.width, goal % world.width
            goal_dict[(x, y)] = i + 1
        for b, box in enumerate(world.boxes):
            if not box.finished:
                dest = tuple(box.dest.tolist())
                obs[box.state.p_pos[0]][box.state.p_pos[1]] = goal_dict[dest]
                obs_goal[box.dest[0]][box.dest[1]] = goal_dict[dest]
        return np.stack([obs, obs_goal], axis=0)

    def partially_observation(self, boxes, world):
        obs = np.zeros((world.width, world.height))
        for box in boxes:
            obs[box.state.p_pos[0], box.state.p_pos[1]] += 1
        return obs

    def add_box(self, world, np_random):
        poi_start = [x for x in self.start if x not in self.check_start(world)]
        poisson_num = min(stats.poisson.rvs(mu=2, size=1)[0], self.num_box, len(poi_start))

        box_buffer = np.random.choice(poi_start, size=poisson_num, replace=False).tolist()
        dest_buffer = np.random.choice(self.end, size=poisson_num).tolist()

        for b in range(poisson_num):
            box = Box()
            box.index = len(world.boxes) + b + 1
            box.name = box.type + "_" + str(len(world.boxes) + b)
            box.state.p_pos = np.array([box_buffer[b] // world.width, box_buffer[b] % world.width], dtype=int)
            box.state_last.p_pos = box.state.p_pos
            box.dest = np.array([dest_buffer[b] // world.width, dest_buffer[b] % world.width], dtype=int)
            box.color = (self.colors[box.index % len(self.colors)] + box.index // len(self.colors) * 50) % 256
            box.route = 0
            box.path = [box.state.p_pos]
            box.finished = False
            world.boxes.append(box)

    def check_start(self, world):
        num = []
        for box in world.boxes:
            if box.state.p_pos[0] == 0 or box.state.p_pos[1] == 0:
                num.append(box.state.p_pos[0] * world.width + box.state.p_pos[1])
        return num
