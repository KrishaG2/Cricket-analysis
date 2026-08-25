from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).parent / "data" / "players.csv"

BATTING_FEATURES = [
    "batting_average",
    "strike_rate",
    "boundary_pct",
    "dot_ball_pct",
    "powerplay_sr",
    "middle_sr",
    "death_sr",
    "six_rate",
]


def load_players():
    if not DATA_PATH.exists():
        raise FileNotFoundError

    return pd.read_csv(DATA_PATH)


def batting_players(players):
    return players[players["batting_average"].notna()].copy()


def bowling_players(players):
    return players[players["bowling_economy"].notna()].copy()
