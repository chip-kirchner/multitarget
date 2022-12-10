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

    def __eq__(self, other):

        isPosition = isinstance(other, self.__class__)

        if not isPosition:
            return False
        else:
            return self.x == other.x and self.y == other.y

    def __hash__(self):
        t = (self.x, self.y)
        return hash(t)

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

    def viewable(self, position, sight) -> bool:
        """
        Return true if position is viewable from this point given sight.
        """
        return np.abs(self.x - position.x) <= sight and np.abs(self.y - position.y) <= sight

    def relative(self, other) -> tuple:
        x = other.x - self.x
        y = other.y - self.y
        return (x, y)

class Target:
    def __init__(self, position: Position, max_capacity):
        self.position = position
        self.active = True
        self.max_capacity = max_capacity
        self.capacity = max_capacity

    def capture(self):
        if self.capacity >= self.max_capacity:
            self.active = False
        else:
            self.capacity += 1


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
        self.targets = []

        self.field_size = field_size

        self.penalty = penalty
        self.step_penalty = step_penalty
        
        self.num_targets = num_targets
        self._targets = []
        self.target_capacity = players
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

        self.init_layers()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """

        # grid observation space
        grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

        # agents layer: agent levels
        agents_min = np.zeros(grid_shape, dtype=np.float32)
        agents_max = np.ones(grid_shape, dtype=np.float32)

        # foods layer: foods level
        foods_min = np.zeros(grid_shape, dtype=np.float32)
        foods_max = np.ones(grid_shape, dtype=np.float32)

        # total layer
        min_obs = np.stack([agents_min, foods_min]).flatten()
        max_obs = np.stack([agents_max, foods_max]).flatten()

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @property
    def n_agent(self):
        return self.n_agents

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
                if not target.active:
                    toReturn.append(0)
                else:
                    toReturn.append(player.position.distance(target.position))

        return np.array(toReturn)

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def adjacent_target(self, p: Position):
        number = 0
        positions = []

        for target in self._targets:
            if target.active and p.isAdjacent(target.position):
                number += 1
                positions.append(target)
        
        return number, positions

    def adjacent_players(self, row, col):
        p = Position(row, col)
        return [
            player
            for player in self.players
            if p.isAdjacent(player.position)
        ]

    def spawn_targets(self, num_targets):
        target_count = 0
        attempts = 0

        while target_count < num_targets and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)
            p = Position(row, col)

            # check if it has neighbors:
            skip = False
            for target in self._targets:
                if p.distance(target.position) < 3:
                    skip = True
            
            if skip:
                continue

            target_count += 1
            self._targets.append(Target(p, self.target_capacity))

    def _is_empty_location(self, row, col):
        p = Position(row, col)
        for target in self._targets:
            if target.position == p:
                return False
        for a in self.players:
            if a.position == p:
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
                        Position(row, col),
                        self.field_size,
                        i
                    )
                    i += 1
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        p = player.position
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position.x > 0
                and self.target_layer[p.x + self.sight - 1, p.y + self.sight] == 0
                and self.player_layer[p.x + self.sight - 1, p.y + self.sight] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position.x < self.rows - 1
                and self.target_layer[p.x + self.sight + 1, p.y + self.sight] == 0
                and self.player_layer[p.x + self.sight + 1, p.y + self.sight] == 0
            )
        elif action == Action.WEST:
            return (
                player.position.y > 0
                and self.target_layer[p.x + self.sight, p.y + self.sight - 1] == 0
                and self.player_layer[p.x + self.sight, p.y + self.sight - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position.y < self.cols - 1
                and self.target_layer[p.x + self.sight, p.y + self.sight + 1] == 0
                and self.player_layer[p.x + self.sight, p.y + self.sight + 1] == 0
            )
        elif action == Action.LOAD:
            n, _ = self.adjacent_target(player.position)
            return n > 0

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

        def _player_obs(player, target_layer, player_layer, bounds_layer):
            p = player.position

            left = p.x 
            right = p.x + self.sight*2

            below = p.y
            above = p.y + self.sight*2

            tg = target_layer[left:right+1, below:above+1]
            pl = player_layer[left:right+1, below:above+1]
            bd = bounds_layer[left:right+1, below:above+1]

            return np.stack((tg, pl),axis=0)
        
        # Add players to grid
        for player in self.players:
            p = player.position
            self.player_layer[p.x + self.sight, p.y + self.sight] = 1
            
        for target in self._targets:
            if target.active:
                p = target.position
                self.target_layer[p.x + self.sight, p.y + self.sight] = 1
        
        nobs = tuple([_player_obs(player, self.target_layer, self.player_layer, self.bounds_layer).flatten() for player in self.players])
        rewards = tuple([player.reward for player in self.players])
        
        return nobs, rewards, self._game_over, {}

    def init_layers(self):
        plus_size = (self.field_size[0]+self.sight*2, self.field_size[1]+self.sight*2)
        self.target_layer = np.zeros(plus_size, np.int32)
        self.player_layer = np.zeros(plus_size, np.int32)
        self.bounds_layer = np.ones(plus_size, np.int32)
        self.bounds_layer[self.sight:self.sight+self.field_size[0], self.sight:self.sight+self.field_size[1]] = np.zeros(self.field_size, np.int32)
    
    def update_player_layer(self):
        plus_size = (self.field_size[0]+self.sight*2, self.field_size[1]+self.sight*2)
        self.player_layer = np.zeros(plus_size, np.int32)

        for player in self.players:
            p = player.position
            self.player_layer[p.x + self.sight, p.y + self.sight] = 1

    def update_target_layer(self):
        for target in self._targets:
            p = target.position
            if target.active:
                self.target_layer[p.x + self.sight, p.y + self.sight] = 1
            else:
                self.target_layer[p.x + self.sight, p.y + self.sight] = 0

    def update_position(self, player, new_position):
        """
        update players position and player_layer with new (x, y) pair
        """
        p = player.position
        x, y = new_position
        self.player_layer[p.x + self.sight, p.y + self.sight] = 0
        p.x, p.y = x, y
        self.player_layer[p.x + self.sight, p.y + self.sight] = 1

    
    def reset(self):
        self.init_layers()
        
        self.spawn_players()
        self.update_player_layer()
        

        self._targets = []
        self.spawn_targets(self.num_targets)
        self.update_target_layer()
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
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[(player.position.x, player.position.y)].append(player)
            elif action == Action.NORTH:
                collisions[(player.position.x - 1, player.position.y)].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position.x + 1, player.position.y)].append(player)
            elif action == Action.WEST:
                collisions[(player.position.x, player.position.y - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position.x, player.position.y + 1)].append(player)
            elif action == Action.LOAD:
                collisions[(player.position.x, player.position.y)].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            
            self.update_position(v[0], k)

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            n, tars = self.adjacent_target(player.position)
            if  n > 0:
                t = tars[0]
                frow, fcol = t.position.x, t.position.y

                adj_players = self.adjacent_players(frow, fcol)
                adj_players = [
                    p for p in adj_players if p in loading_players or p is player
                ]

                loading_players = loading_players - set(adj_players)
            
                if len(adj_players) < t.capacity:
                    # failed to load
                    for a in adj_players:
                        a.reward -= self.penalty
                    continue

                # else target was captured and each player scores points
                for a in adj_players:
                    a.reward += self.target_reward
                    
                # and the food is removed
                self.target_layer[frow + self.sight, fcol + self.sight] = 0
                for target in self._targets:
                    if Position(frow, fcol) == target.position:
                        target.capture()

            else:
                player.reward -= self.penalty

        self._game_over = (
            (not any([target.active for target in self._targets])) or self._max_episode_steps <= self.current_step
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
