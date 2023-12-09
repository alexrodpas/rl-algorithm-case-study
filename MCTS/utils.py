from PIL import Image
import numpy as np
from IPython.display import display

def render_rgb(rgb_array, ipynb=False):
    # Render the RGB array returned by the env as an image
    if ipynb:
        display(Image.fromarray(rgb_array).resize((150,150)))
    else:
        Image.fromarray(rgb_array).resize((450,450)).show()

def action_arrow(action):
    # Mapping the GridWorld actions to arrows
    # We have 4 actions, corresponding to "right", "up", "left", "down"
    if action is None:
        return ""
    else:
        return np.array(['→','↓','←','↑'])[action]
    
import random
def opponent_random(env, player=1):
    """
    Opponent chooses a random free position (valid action) on the board.

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player
            b = field to place stone by index
    """
    valid_moves = env.get_valid_moves()
    return (player, random.choice(valid_moves))

