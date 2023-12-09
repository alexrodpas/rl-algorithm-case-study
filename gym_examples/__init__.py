from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
    id="tictactoe-v0",
    entry_point="gym_examples.envs:TictactoeEnv",
)