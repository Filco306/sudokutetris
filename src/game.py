from typing import Tuple, List, Union
import numpy as np
from random import choices, choice
import gym
from gym import spaces
from time import sleep

PIECES = [
    ((0, 0),),
    ((0, 0), (0, 1)),  # Two-piece horizontal
    ((0, 0), (0, 1), (0, 2)),  # 3-piece horizontal
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),  # 5-piece horizontal
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),  # 3-piece vertical
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),  # 5-piece vertical
    ((1, 0), (0, 1)),  # 2-diag
    ((2, 0), (1, 1), (0, 2)),  # 3-diag
    ((0, 0), (0, 1), (1, 1), (2, 1)),  # lying reverse L
    ((0, 0), (1, 0), (2, 0), (0, 1)),  # lying L
    ((0, 0), (1, 0), (0, 1), (1, 1)),  # square
    ((0, 0), (1, 0), (0, 1)),  # tri-corner 1
    ((0, 0), (1, 0), (1, 1)),  # tri-corner 2
    ((0, 0), (0, 1), (1, 1)),  # tri-corner 3
    ((1, 0), (0, 1), (1, 1)),  # tri-corner 4
]


class SudokuTetrisGame(gym.Env):
    def __init__(self, board_size: Tuple[int, int] = (9, 9)):
        super().__init__()
        self.is_finished = False
        self.n_points = 0
        assert board_size == (9, 9), "Not implemented for any other size yet. "
        assert (
            np.sqrt(board_size[0]) % 1 == 0
        ), "Please choose a board size with integer root. "
        self.square_size = int(np.sqrt(board_size[0]))
        self.board_size_x = board_size[0]
        self.board_size_y = board_size[1]
        self.board_size = board_size
        self.observation_space = spaces.Box(
            low=False, high=True, shape=board_size, dtype=np.bool
        )
        self.reward_range = (0, np.inf)
        # TODO: Think of a better action space than this?
        self.action_space = spaces.Discrete(int(len(PIECES) * np.prod(board_size)))
        self.n_actions = int(len(PIECES) * np.prod(board_size))
        self.n_pieces = len(PIECES)
        self.reset()

    def reset(self):
        self.board = self.generate_empty_board((self.board_size_x, self.board_size_y))
        self.current_pieces = self.generate_pieces()
        return self._next_observation()

    def get_coord(self, i: int) -> Tuple[int, int]:
        # Get coord from an index at board.
        # NOTE: Assumes that n_pieces has an int sqrt
        y = int(i % self.board_size_y)
        x = int(((i) - ((i) % self.board_size_x)) / self.board_size_x)
        return x, y

    def interpret_action(self, action: Union[np.ndarray, int]) -> Tuple[int, int, int]:
        if isinstance(action, np.ndarray) is True:
            assert (
                action.sum() == 1
            ), "More than one action chosen. Please choose one action. "
        else:
            assert (
                action < self.n_actions and action >= 0
            ), "Action must be between 0 and {}, got {}".format(self.n_actions, action)
        # For each action, we will have 9x9 = 81 different choices
        n_per_piece = np.prod(self.board_size)
        piece = int((action - (action % n_per_piece)) / n_per_piece)
        x, y = self.get_coord(int(action % n_per_piece))
        return piece, x, y

    def render_piece(self, piece: Tuple[Tuple[int, int]]) -> str:
        max_coord_x = max([x[0] for x in piece]) + 1
        max_coord_y = max([x[1] for x in piece]) + 1
        s = ""
        for x in range(max_coord_x):
            for y in range(max_coord_y):
                if (x, y) in piece:
                    s += "#"
                else:
                    s += " "
            s += "\n"
        return s

    def render(self, mode="human", return_s=False):
        # Generate board
        to_print = ""
        sep_row = "-" * (self.board_size_y * 2 + 1) + "\n"
        to_print += sep_row

        for x_row in self.board:
            s = "|"
            for y in x_row:
                if y == True:  # noqa: E712
                    s += "#"
                else:
                    s += " "
                s += "|"
            s += "\n"
            to_print += s
            to_print += sep_row

        # Generate pieces
        for i, piece in enumerate(self.current_pieces):
            to_print += "\n \n PIECE {} \n".format(i)
            to_print += self.render_piece(piece)

        if return_s is True:
            return to_print
        else:
            print(to_print)

    def sample_action(self):
        available_actions = self.get_available_actions()
        return choice(available_actions)

    def get_available_actions(self, as_array: bool = False) -> List[bool]:
        """
            Returns which ones are available
        """

        available_actions = []
        for piece in self.current_pieces:
            x = PIECES.index(piece)

            available_actions += [
                i
                for i in range(
                    x * np.prod(self.board_size), (x + 1) * np.prod(self.board_size)
                )
            ]
        placeable_actions = []
        # For each action, check if it is placeable
        for action in available_actions:
            piece_, x, y = self.interpret_action(action)
            assert PIECES[piece_] in self.current_pieces, "{} not in {}".format(
                PIECES[piece_], self.current_pieces
            )
            for i, p in enumerate(self.current_pieces):
                if p == piece_:
                    break
            if self.is_placeable(fst_coord=(x, y), piece=PIECES[piece_]):
                placeable_actions.append(action)

        # For these indices, return all
        if as_array is True:
            vec = np.zeros(self.n_actions)
            vec[placeable_actions] = 1
            return vec
        return placeable_actions

    def step(self, action: Union[np.ndarray, int]):
        # Get which piece to place at coord x, y
        piece, x, y = self.interpret_action(action)
        assert PIECES[piece] in self.current_pieces
        # Which piece?

        fst_coord = (x, y)
        is_placeable = self.is_placeable(fst_coord, PIECES[piece])
        assert (
            is_placeable
        ), "Not a valid move. Trying to put {} \n{} on coord {}. \n \nBoard is {}".format(
            PIECES[piece],
            self.render_piece(PIECES[piece]),
            fst_coord,
            self.render(return_s=True),
        )
        reward = 0
        i = self.current_pieces.index(PIECES[piece])
        to_place = self.current_pieces.pop(i)
        self.place_piece(fst_coord=fst_coord, piece=to_place)
        reward += len(to_place) + self.check_and_clear()
        if len(self.current_pieces) == 0:
            self.generate_pieces()
        done = self.check_if_done()
        return (
            self.board,
            reward,
            done,
            {"availablilities": self.get_available_actions(as_array=True)},
        )

    def _next_observation(self):
        return self.board

    def generate_empty_board(self, board_size) -> List[List[Tuple[int, int]]]:
        return np.array(
            [[False for x in range(board_size[1])] for y in range(board_size[0])]
        )

    def draw_piece(self, piece: tuple):
        pass

    def is_placeable(
        self, fst_coord: Tuple[int, int], piece: Tuple[Tuple[int, int]]
    ) -> bool:
        coords = [(fst_coord[0] + x[0], fst_coord[1] + x[1]) for x in piece]
        is_within_bounds = all(
            [
                coord[0] < self.board_size_x and coord[1] < self.board_size_y
                for coord in coords
            ]
        )
        if is_within_bounds is False:
            return False
        else:
            return (
                all(
                    [
                        self.board[coord[0], coord[1]] == False  # noqa: E712
                        for coord in coords
                    ]
                )
                is True
            )

    def generate_pieces(self):
        self.current_pieces = choices(PIECES, k=3)
        return self.current_pieces

    def place_piece(self, fst_coord: Tuple[int, int], piece: Tuple[Tuple[int, int]]):
        assert self.is_placeable(fst_coord, piece), "Not a valid move. "
        for x, y in piece:
            self.board[fst_coord[0] + x, fst_coord[1] + y] = True

    def check_box(self, i) -> bool:
        x = int(i % 3)
        y = int(i - (i % 3))
        return np.sum(self.board[x : (x + 3), y : (y + 3)]) == self.board_size_x

    def clear(self, to_clear: Tuple[str, int]):
        which_ = to_clear[0]
        item_ = to_clear[1]
        if which_ == "row":
            self.board[item_, :] = False
        elif which_ == "col":
            self.board[:, item_] = False
        elif which_ == "box":
            x = int(item_ % 3)
            y = int(item_ - (item_ % 3))
            self.board[x : (x + 3), y : (y + 3)] = False
        else:
            raise Exception("Expection either row, col or box, got {}".format(which_))

    def check_and_clear(self):
        # Check board if we have any rows, columns
        # or squares to reward points for, and clear them in that case.
        clear_rows = []
        clear_cols = []
        clear_box = []
        for i in range(self.board.shape[0]):
            if self.board[i, :].sum() == self.board_size_x:
                print("REWARD!! CLEARING ROW {}".format(i))
                clear_rows.append(("row", i))
            if self.board[:, i].sum() == self.board_size_x:
                print("REWARD!! CLEARING COL {}".format(i))
                clear_cols.append(("col", i))
            if self.check_box(i) is True:
                print("REWARD!! CLEARING BOX {}".format(i))
                clear_box.append(("box", i))
        reward = 0
        # Clear these boxes
        i = 0
        for to_clear in clear_rows + clear_cols + clear_box:
            reward += self.board_size_x + i
            i += 1
            self.clear(to_clear)
        return reward

    def check_if_done(self):
        return len(self.get_available_actions()) == 0


def test_run(n_runs=200):
    env = SudokuTetrisGame()
    obs = env.reset()
    tot_reward = 0
    for i in range(n_runs):
        action = env.sample_action()
        print(
            "Sampled action {} which is {}".format(action, env.interpret_action(action))
        )
        obs, rewards, done, info = env.step(action)
        tot_reward += rewards
        env.render()
        if done is True:
            print("Done! ")
            print("Total reward: {}".format(tot_reward))
            print("Final board: ")
            env.render()
            tot_reward = 0
            obs = env.reset()  # noqa: F841
        sleep(1)


if __name__ == "__main__":
    test_run()
