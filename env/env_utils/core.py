import numpy as np
from baseline.single_agent_planner import compute_heuristics, a_star


class EntityState:
    def __init__(self) -> None:
        self.p_pos = None


class AgentState(EntityState):
    def __init__(self) -> None:
        super().__init__()


class Action:
    def __init__(self) -> None:
        self.u = None


class Entity:
    def __init__(self) -> None:
        self.name = ""
        # properties
        self.width = None
        self.height = None
        # color
        self.color = None
        # speed
        self.speed = None
        # state
        self.state = EntityState()


class Control(Entity):
    def __init__(self) -> None:
        super().__init__()
        self.type = 'control'
        # size
        self.width = 1
        self.height = 1
        # color
        self.color = np.array([100, 100, 100])
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # cost: the cost of each controller
        self.cost = None
        # num: the number of controller
        self.num = 0


class Box(Entity):
    def __init__(self) -> None:
        super().__init__()
        self.type = 'box'
        # size
        self.width = 1
        self.height = 1
        # speed
        self.speed = 1
        # color
        self.color = np.array([200, 100, 100])
        # state
        self.state = AgentState()
        self.state_last = AgentState()
        # check if finished
        self.finished = False
        # script behavior to execute
        self.action_callback = None
        # route: travelled distance
        self.route = None
        self.path = None
        # destination of each boxes
        self.dest = np.array([0, 0])
        # index of each box
        self.index = 0


class World:  # multi-agent world
    def __init__(self) -> None:
        # list of agents
        self.agents = []
        self.boxes = []
        # p_position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # world size
        self.width = None
        self.height = None
        self.grid = None
        self.boundary = []
        self.baseline = False

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.boxes

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):

        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # box和control的个数
        control_num = [agent.num for agent in self.agents]
        pos = np.zeros(len(self.boxes))

        # todo, 优化agent与box的计算
        # 在目前的环境里agent只有一个
        for a, agent in enumerate(self.agents):
            control = agent.action.u
            for k in range(control_num[a]):
                if control[k][0] != 0 or control[k][1] != 0:
                    agent.cost += 1

        for b, box in enumerate(self.boxes):
            if not box.finished:
                pos[b] = self.width * box.state.p_pos[0] + box.state.p_pos[1]
            else:
                pos[b] = -1

        step_list = self.move_box(control, pos)

        return step_list

    def getScreenRGB(self):
        screen = np.ones((self.height, self.width, self.dim_color)) * 255
        return screen.astype(np.int16)

    def move_box(self, control, origin):
        step_list = np.zeros(len(self.boxes))
        recent = np.zeros(len(self.boxes))
        for b, box in enumerate(self.boxes):
            if not box.finished:
                position = self.width * box.state.p_pos[0] + box.state.p_pos[1]
                x, y = box.state.p_pos + control[position] * (box.width, box.height)
                recent[b] = self.width * x + y
                if control[position][0] == 0 and control[position][1] == 0:  # no move
                    step_list[b] = -3
            else:
                recent[b] = -1

        # 优先解决循环碰撞
        common_values = np.intersect1d(origin, recent)
        loop_buffer = dict()
        for value in common_values:
            if value != -1:
                index1 = np.where(origin == value)[0]  # 找到value在origin中的索引
                index2 = np.where(recent == value)[0]  # 找到value在recent中的索引
                loop_buffer[value] = sorted(index1.tolist() + index2.tolist())

        result_dict = {}
        for key, value in loop_buffer.items():
            value_set = tuple(value)  # 将列表转换为元组，以便作为字典的key
            if value_set not in result_dict:
                result_dict[value_set] = [key]
            else:
                result_dict[value_set].append(key)

        # 打印出相同value的key
        for value, keys in result_dict.items():
            if len(keys) > 1:
                step_list[list(value)] = -2

        duplicate_values = np.array([1])
        # edge collision
        while duplicate_values.size > 0 and np.any(duplicate_values != -1):
            unique_values, counts = np.unique(recent, return_counts=True)
            duplicate_values = unique_values[counts > 1]
            for value in duplicate_values:
                if value != -1:
                    indices = np.where(recent == value)[0]
                    index = np.random.randint(0, len(indices))
                    new_indices = np.delete(indices, index)
                    step_list[new_indices] = -1
                    for i in new_indices:
                        recent[i] = origin[i]

        for i in range(len(step_list)):
            box = self.boxes[i]
            if step_list[i] == 0 and not box.finished:
                x, y = recent[i] // self.width, recent[i] % self.width
                box.state_last.p_pos = box.state.p_pos
                box.state.p_pos = np.array([x, y], dtype=int)
                if np.array_equal(box.state.p_pos, box.dest):
                    box.finished = True
                if np.sum(np.abs(box.dest - box.state.p_pos)) < np.sum(np.abs(box.dest - box.state_last.p_pos)):
                    step_list[i] = 1
                else:
                    step_list[i] = 2
                box.route += 1
                box.path.append(box.state.p_pos)


        return step_list

    def staticCollision(self, box_a, x, y):
        """
        staticCollision returns True if box_a go to an agent have another box
        """
        #
        for box in self.boxes:
            if box != box_a and not box.finished:
                if x == box.state.p_pos[0] and y == box.state.p_pos[1]:
                    return True
        return False

    def moveCollision(self, box_a, x, y):
        """moveCollision(id,(x,y)) returns true if agent collided with
       any other agent in the state after moving to coordinates (x,y)
       agent_id: id of the desired agent to check for
       newPos: coord the agent is trying to move to (and checking for collisions)
       """

        # def eq(f1,f2):return abs(f1-f2)<0.001
        def collide(a1, a2, b1, b2):
            """
            a1,a2 are coords for agent 1, b1,b2 coords for agent 2, returns true if these collide diagonally
            """
            if a1 is None or b1 is None:
                return False
            return np.isclose((a1[0] + a2[0]) / 2., (b1[0] + b2[0]) / 2.) and np.isclose((a1[1] + a2[1]) / 2.,
                                                                                         (b1[1] + b2[1]) / 2.)

        # up until now we haven't moved the agent, so getPos returns the "old" location
        last_pos = box_a.state_last.p_pos
        new_pos = [x, y]
        for box in self.boxes:
            if box == box_a or box.finished:
                continue
            a_past = box.state_last.p_pos
            a_pres = box.state.p_pos
            if collide(a_past, a_pres, last_pos, new_pos):
                return True
        return False
