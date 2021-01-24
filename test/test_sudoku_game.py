from src.game import SudokuTetrisGame, PIECES
import numpy as np


def test_get_coords():
    game = SudokuTetrisGame()
    i = 0
    for x in range(game.board_size_x):
        for y in range(game.board_size_y):
            assert game.get_coord(i) == (x, y)
            i += 1


def test_replacementness():
    game = SudokuTetrisGame()
    for i in range(len(PIECES)):
        for j in range(int(np.prod(game.board_size))):
            piece_index, x, y = game.interpret_action(81 * i + j)
            x_, y_ = game.get_coord(j)
            assert piece_index == i
            assert y == y_
            assert x == x_


def test_get_available_actions():
    game = SudokuTetrisGame()
    game.board = np.full((9, 9), True)
    game.board[8, :] = False
    game.current_pieces = [PIECES[4]]
    assert len(game.get_available_actions()) == 5
    game.board[1, :] = False
    assert len(game.get_available_actions()) == 10
    game.board = np.full((9, 9), True)
    game.board[3, 3] = False
    game.board[3, 4] = False
    game.board[4, 4] = False
    game.board[5, 4] = False
    game.current_pieces = [PIECES[12]]
    assert len(game.get_available_actions()) == 1
    game.current_pieces = [PIECES[11], PIECES[12]]
    assert len(game.get_available_actions()) == 1
    game.current_pieces = [PIECES[0], PIECES[11], PIECES[12]]
    assert len(game.get_available_actions()) == 5
