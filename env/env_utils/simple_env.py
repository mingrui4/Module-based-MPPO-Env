import gymnasium.spaces
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from env.env_utils.assert_out_of_bounds import AssertOutOfBoundsWrapper


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 2,
    }

    def __init__(
            self,
            scenario,
            world,
            max_cycles,
            scaling=25,
            render_mode=None,
            continuous_actions=False,
            local_ratio=None,
    ):
        super().__init__()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
        pygame.init()
        self.scaling = scaling
        self.viewer = None
        self.width = self.world.width * self.scaling
        self.height = self.world.height * self.scaling
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.space_dim = self.world.dim_p * 2 + 1

        # Set up the drawing windows

        self.renderOn = False
        self._seed()

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.boxes = [box.name for box in self.world.boxes]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        # action space: 5 actions (no actions, left, right, down, up)
        # state space: picture of full screen
        self.action_spaces = dict()
        self.observation_spaces = dict()
        for agent in self.world.agents:
            obs_shape = self.scenario.observation(agent, self.world).shape
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(self.space_dim,)
                )
            else:
                action = np.ones(agent.num) * self.space_dim
                self.action_spaces[agent.name] = spaces.MultiDiscrete(action)
            self.observation_spaces[agent.name] = spaces.Dict(
                {
                    'observation': spaces.Box(
                        low=0, high=127, shape=obs_shape, dtype=np.int8
                    ),
                    'action_mask': spaces.Box(low=0, high=1, shape=(self.space_dim,), dtype=np.int8),
                })

        self.state_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.world.width, self.world.height, self.world.dim_color),
            dtype=np.int16,
        )

        self.steps = 0
        self.current_actions = [None] * self.num_agents

        # wait, left, right, up, down
        self.pos_inc = np.array([
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ])

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        observation = self.scenario.observation(agent, self.world)

        legal_moves = self.legal_moves(agent) if agent == self.agent_selection else []
        agent_obj = self.world.agents[self._index_map[agent]]
        action_mask = [np.zeros(self.space_dim, dtype=np.int8) for _ in range(agent_obj.num)]
        for i, legal_list in enumerate(legal_moves):
            for j in legal_list:
                action_mask[i][j] = 1
        return {"observation": observation, "action_mask": action_mask}

    def state(self):
        states = np.zeros((self.world.width, self.world.height))
        return states

    def legal_moves(self, agent_name):
        agent = self.world.agents[self._index_map[agent_name]]
        legal_moves = [[0] for _ in range(agent.num)]

        if self.world.baseline:
            return [[0, 1, 2, 3, 4] for _ in range(agent.num)]

        for box in self.world.boxes:
            if not box.finished:
                for i in range(1, self.space_dim):
                    x = box.state.p_pos[0] + self.pos_inc[i][0] * agent.width
                    y = box.state.p_pos[1] + self.pos_inc[i][1] * agent.height
                    if x < 0 or x > self.world.width - agent.width:
                        continue
                    if y < 0 or y > self.world.height - agent.height:
                        continue
                    if -1 in self.world.grid[x: x + agent.width, y: y + agent.height]:
                        continue
                    legal_moves[box.state.p_pos[0] * self.world.width + box.state.p_pos[1]].append(i)

        return legal_moves

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents} | {name: {} for name in self.boxes}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            self._set_action(action, agent, self.action_spaces[agent.name])

        action_status = self.world.step()

        # rewards
        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(action_status, agent, self.world))
            if self.local_ratio is not None:
                reward = (
                        global_reward * (1 - self.local_ratio)
                        + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

            # store total cost and route for agents
            self.infos[agent.name]['cost'] = agent.cost

        for box in self.world.boxes:
            if box.name in self.infos:
                self.infos[box.name]['route'] = box.route
                self.infos[box.name]['path'] = box.path
            else:
                self.infos[box.name] = {}
                self.infos[box.name]['route'] = box.route
                self.infos[box.name]['path'] = box.path

        if self.world.add_box:
            self.scenario.add_box(self.world, self.np_random)
        # print(self.infos, self.rewards)

    # set env action for a particular control
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = [np.zeros(self.world.dim_p, dtype=np.int8) for _ in range(agent.num)]

        for i in range(agent.num):
            if action[0] == 1:  # up
                agent.action.u[i][0] = -1
            if action[0] == 2:  # down
                agent.action.u[i][0] = 1
            if action[0] == 3:  # left
                agent.action.u[i][1] = -1
            if action[0] == 4:  # right
                agent.action.u[i][1] = 1
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def check_for_done(self):
        return self.scenario.check_for_done(self.world)

    def step(self, action):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection

        legal_moves = self.legal_moves(cur_agent)
        positions = [box.state.p_pos for box in self.world.boxes]

        for i in range(len(action)):
            assert action[i] in legal_moves[i], 'agent illegal move. i:{}, Action: {}, Legal Moves: {}, Box Positions: {}'.\
                format(i, action, legal_moves, positions)

        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
            done = self.check_for_done()
            if done:
                print('done!')
                self.terminations = {name: True for name in self.agents}
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
            return

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        screen = self.world.getScreenRGB()
        for i in range(self.world.height):
            for j in range(self.world.width):
                # Rect(left, top, width, height)
                pygame.draw.rect(self.screen, screen[i, j, :], tuple(e * self.scaling for e in (j, i, 1, 1)))

        # vertical
        for i in range(1, self.world.width):
            pygame.draw.line(self.screen, (0, 0, 0), (i * self.scaling, 0), (i * self.scaling, self.height), 1)
        # parallel
        for j in range(1, self.world.height):
            pygame.draw.line(self.screen, (0, 0, 0), (0, j * self.scaling), (self.width, j * self.scaling), 1)
        # boundary
        for boundary in self.world.boundary:
            p1, p2 = boundary
            pygame.draw.line(self.screen, (200, 0, 0), (p1[0] * self.scaling, p1[1] * self.scaling),
                             (p2[0] * self.scaling, p2[1] * self.scaling), 2)

        # agent
        margin = 0.1
        for box in self.world.boxes:
            # box goal
            x1, y1 = box.dest[0], box.dest[1]
            goal = tuple(
                e * self.scaling for e in (y1 + margin, x1 + margin, box.height - 2 * margin, box.width - 2 * margin))
            pygame.draw.rect(self.screen, box.color, goal, width=2)

            x, y = box.state.p_pos + (margin, margin)
            rect = tuple(e * self.scaling for e in (y, x, box.height - 2 * margin, box.width - 2 * margin))
            pygame.draw.ellipse(self.screen, box.color, rect)

        pygame.image.save(self.screen, 'screen.png')

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False
