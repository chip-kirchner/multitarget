import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3

class Position(object):
    """
    Object to store and compare positions of objects on the grid.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, position) -> int:
        return np.abs(self.x - position.x) + np.abs(self.y - position.y)

    def isDiagonal(self, position, s=1) -> bool:
        return np.abs(self.x - position.x) == s and np.abs(self.y - position.y) == s
    
    def isAdjacent(self, position, ignoreDiagonal=False) -> bool:
        if self.distance(position) <= 1:
            return True
        elif (not ignoreDiagonal) and self.isDiagonal(position):
            return True
        
        return False

    def viewable(self, position, sight, ignoreDiagonal=False):
        """
        Return true if position is viewable from this point given sight and diagonal.
        """


        return False

class Target:
    def __init__(self, position: Position):
        self.position = position
        self.active = True

    def capture(self):
        self.active = False


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.field_size = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.id = None

    def setup(self, position: Position, field_size, id=0):
        self.history = []
        self.position = position
        self.field_size = field_size
        self.reward = 0
        self.id = id

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class MulticaptureEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        field_size,
        num_targets,
        target_capacity,
        sight,
        max_episode_steps,
        target_reward=10,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0,
        step_penalty=0.0
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty
        self.step_penalty = step_penalty
        
        self.num_targets = num_targets
        self._targets = []
        self.target_capacity = target_capacity
        self.target_reward = target_reward

        self.sight = sight
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]

            num_targets = self.num_targets

            min_obs = [-1, -1] * num_targets + [-1, -1] * len(self.players)
            max_obs = [field_x-1, field_y-1] * num_targets + [
                field_x-1,
                field_y-1,
            ] * len(self.players)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer:
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32)

            # target layer:
            targets_min = np.zeros(grid_shape, dtype=np.float32)
            targets_max = np.ones(grid_shape, dtype=np.float32)

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, targets_min, access_min])
            max_obs = np.stack([agents_max, targets_max, access_max])

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @property
    def n_agent(self):
        return self.n_agents

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over
    
    @property
    def state_size(self):
        return int(np.prod(self.get_state().size))
    
    def get_state(self):

        targets = self._targets

        toReturn = []

        for player in self.players:
            for target in targets:
                x, y = target.position

                if not target.active:
                    toReturn.append(0)
                else:
                    toReturn.append((np.abs(x - player.position[0]) + np.abs(y - player.position[1])) / np.sum(self.field_size))

        return np.array(toReturn)

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_target(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_target_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_targets(self, num_targets):
        target_count = 0
        attempts = 0

        while target_count < num_targets and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = 1
            target_count += 1
            self._targets.append(Target((row, col)))

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False
        return True

    def spawn_players(self):
        i = 1
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.field_size,
                        i
                    )
                    i += 1
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_target(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.num_targets):
                obs[2 * i] = -1
                obs[2 * i + 1] = -1

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[2 * i] = y
                obs[2 * i + 1] = x

            for i in range(len(self.players)):
                obs[self.num_targets * 2 + 2 * i] = -1
                obs[self.num_targets * 2 + 2 * i + 1] = -1

            for i, p in enumerate(seen_players):
                obs[self.num_targets * 2 + 2 * i] = p.position[0]
                obs[self.num_targets * 2 + 2 * i + 1] = p.position[1]

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x + self.sight, player_y + self.sight] = 1
            
            targets_layer = np.zeros(grid_shape, dtype=np.float32)
            targets_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[:self.sight, :] = 0.0
            access_layer[-self.sight:, :] = 0.0
            access_layer[:, :self.sight] = 0.0
            access_layer[:, -self.sight:] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0
            
            return np.stack([agents_layer, targets_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * self.sight + 1, agent_y, agent_y + 2 * self.sight + 1
        
        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [get_agent_grid_bounds(*player.position) for player in self.players]
            nobs = tuple([layers[:, start_x:end_x, start_y:end_y] for start_x, end_x, start_y, end_y in agents_bounds])
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}
        
        # check the space of obs
        for i, obs in  enumerate(nobs):
            assert self.observation_space[i].contains(obs), \
                f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"
        
        return nobs, nreward, any(ndone), ninfo

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players()

        self._targets = []
        self.spawn_targets(self.num_targets)
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs, _, _, _ = self._make_gym_obs()

        return nobs

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = -self.step_penalty

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                print("here")
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            if self.adjacent_target(*player.position) > 0:
                frow, fcol = self.adjacent_target_location(*player.position)

                adj_players = self.adjacent_players(frow, fcol)
                adj_players = [
                    p for p in adj_players if p in loading_players or p is player
                ]

                loading_players = loading_players - set(adj_players)
            
                if len(adj_players) < self.target_capacity:
                    # failed to load
                    for a in adj_players:
                        a.reward -= self.penalty
                    continue

                # else target was captured and each player scores points
                for a in adj_players:
                    a.reward += self.target_reward
                    
                # and the food is removed
                self.field[frow, fcol] = 0
                for target in self._targets:
                    if (frow, fcol) == target.position:
                        target.capture()

                ## Add a new food somewhere else on the map
                """
                player_levels = sorted([player.level for player in self.players])
                self.spawn_food(
                    1, max_level=sum(player_levels[:3])
                )
                """

            else:
                player.reward -= self.penalty

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        return self._make_gym_obs()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
