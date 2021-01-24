from typing import Tuple, List
import numpy as np
from random import choices


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


class SudokuTetrisGame:
    def __init__(self, board_size: Tuple[int, int] = (9, 9)):
        assert board_size == (9, 9), "Not implemented for any other size yet. "
        assert (
            np.sqrt(board_size[0]) % 1 == 0
        ), "Please choose a board size with integer root. "
        self.square_size = int(np.sqrt(board_size[0]))
        self.board_size_x = board_size[0]
        self.board_size_y = board_size[1]
        self.board = self.generate_empty_board(board_size)
        self.current_pieces = self.generate_pieces()

    def generate_empty_board(self, board_size) -> List[List[Tuple[int, int]]]:
        return np.array(
            [[False for x in range(board_size[1])] for y in range(board_size[0])]
        )

    def draw_piece(self, piece: tuple):
        pass

    def is_placable(
        self, fst_coord: Tuple[int, int], piece: Tuple[Tuple[int, int]]
    ) -> bool:
        coords = [(fst_coord[0] + x[0], fst_coord[1] + x[1]) for x in piece]

        return (
            all(
                [
                    coord[0] < self.board_size_x and coord[1] < self.board_size_y
                    for coord in coords
                ]
            )
            is True
            and all([self.board[coord[0], coord[1]] is False for coord in coords])
            is True
        )

    def generate_pieces(self):
        self.current_pieces = choices(PIECES, k=3)
        return self.current_pieces

    def place_piece(self, fst_coord: Tuple[int, int], piece: Tuple[Tuple[int, int]]):
        assert self.is_placable(fst_coord, piece), "Not a valid move. "
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
                clear_rows.append(("row", i))
            if self.board[:, i].sum() == self.board_size_x:
                clear_cols.append(("col", i))
            if self.check_box(i) is True:
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
        pass


class OneGame(SudokuTetrisGame):
    def __init__(self):
        super().__init__()
        self.is_finished = False
        self.n_points = 0

    def place_action(self, fst_coord: Tuple[int, int], i: int):
        assert i < 3 and i >= 0, "Not any of the choices"
        assert self.is_placable(fst_coord, self.current_pieces[i]), "Not a valid move. "
        reward = 0
        to_place = self.current_pieces.pop(i)
        print("placing {}".format(to_place))
        self.place_piece(fst_coord=fst_coord, piece=to_place)
        reward += len(to_place) + self.check_and_clear()
        if len(self.current_pieces) == 0:
            self.generate_pieces()

        return reward, self.check_done()


if __name__ == "__main__":

    game = OneGame()
