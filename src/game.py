from typing import Tuple, List

PIECES = [
          ((0,0),),
          ((0,0), (0,1)), # Two-piece horizontal
          ((0,0), (0,1), (0,2)), # 3-piece horizontal
          ((0,0), (0,1), (0,2), (0,3)),
          ((0,0), (0,1), (0,2), (0,3), (0,4)), # 5-piece horizontal
          ((0,0), (1,0)),
          ((0,0), (1,0), (2,0)),
          ((0,0), (1,0), (2,0), (3,0)), # 3-piece vertical
          ((0,0), (1,0), (2,0), (3,0), (4,0)),
          ((0,0), (1,0), (2,0), (3,0), (4,0), (5,0)), # 5-piece vertical
          ((1,0), (0,1)), # 2-diag
          ((2,0), (1,1), (0,2)), # 3-diag
          ((0,0), (0,1), (1,1), (2,1)), # lying reverse L
          ((0,0), (1,0), (2,0), (0,1)), # lying L
          (()),
          ]

class SudokuTetrisGame:
    def __init__(self, board_size : Tuple[int,int] = (9,9)):
        self.board_size_x = board_size[0]
        self.board_size_y = board_size[1]
        self.board = self.generate_empty_board(board_size)


    def generate_empty_board(self, board_size) -> List[List[Tuple[int,int]]]:
        return [[False for x in range(board_size[1])] for y in range(board_size[0])]

    def draw_piece(self, piece : tuple):
        pass

    def is_placable(self, fst_coord : Tuple[int,int], piece : Tuple[Tuple[int,int]]) -> bool:
        coords = [(fst_coord[0] + x[0], fst_coord[1] + x[1]) for x in piece]
        return all([coords[0] < board_size_x and coords[]]) all([self.board[coords[0]][coords[1]] for x in coords])




if __name__ == "__main__":

    game = SudokuTetrisGame()
    game.board[0][1] = True
    game.board[1][0] = True
