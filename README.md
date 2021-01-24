# sudokutetris

An OpenAI gym environment for models to learn playing SudokuTetris, or also known as [BlockuDoku](https://play.google.com/store/apps/details?id=com.easybrain.block.puzzle.games&hl=en&gl=US) (I have no affiliation to Easybrain, the creators of BlockuDoku).

## How to run

Install the requirements

```
pip install -r requirments.txt
```

Then just do

```
python3 index.py
```

which will run it with a random agent for 200 episodes.

## Action space

The action space is discrete vector of shape `(N_PIECES*N_BOARD_TILES,)` - `(19*81,) = 1539`. Which ones are available at a given time are included in the `info` dictionary, the last in the tuple sequence returned by the `step`-function.

If you like it, don't hesitate to star this repository.

Enjoy!
