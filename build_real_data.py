from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen
import io
import json
import zipfile

import pandas as pd


DATA_URL = "https://cricsheet.org/downloads/ipl_json.zip"
START_YEAR = 2022
MIN_BATTING_BALLS = 250
MIN_BOWLING_BALLS = 180

OUTPUT_PATH = Path(__file__).parent / "data" / "players.csv"


def new_batting_stats():
    return {
        "runs": 0,
        "balls": 0,
        "dismissals": 0,
        "fours": 0,
        "sixes": 0,
        "dot_balls": 0,
        "powerplay_runs": 0,
        "powerplay_balls": 0,
        "middle_runs": 0,
        "middle_balls": 0,
        "death_runs": 0,
        "death_balls": 0,
        "matches": set(),
    }


def new_bowling_stats():
    return {
        "runs_conceded": 0,
        "balls": 0,
        "wickets": 0,
        "dot_balls": 0,
        "boundaries_conceded": 0,
        "powerplay_runs": 0,
        "powerplay_balls": 0,
        "middle_runs": 0,
        "middle_balls": 0,
        "death_runs": 0,
        "death_balls": 0,
        "matches": set(),
    }


def rate(numerator, denominator, multiplier=1.0):
    if denominator == 0:
        return 0.0

    return numerator / denominator * multiplier


def phase(over_number):
    if over_number <= 5:
        return "powerplay"

    if over_number <= 14:
        return "middle"

    return "death"


def legal_delivery(delivery):
    extras = delivery.get("extras", {})
    return "wides" not in extras and "noballs" not in extras


def runs_charged_to_bowler(delivery):
    total = delivery["runs"]["total"]
    extras = delivery.get("extras", {})

    return total - extras.get("byes", 0) - extras.get("legbyes", 0)


def is_boundary(delivery):
    batter_runs = delivery["runs"]["batter"]
    non_boundary = delivery["runs"].get(
        "non_boundary",
        False,
    )

    return batter_runs in (4, 6) and not non_boundary


def batter_is_out(wicket):
    return wicket.get("kind", "") not in {
        "retired hurt",
        "absent hurt",
    }


def bowler_gets_wicket(wicket):
    return wicket.get("kind", "") not in {
        "run out",
        "retired hurt",
        "retired out",
        "obstructing the field",
        "timed out",
    }


def process_match(match, match_id, batting, bowling):
    for innings in match.get("innings", []):
        for over in innings.get("overs", []):
            current_phase = phase(int(over["over"]))

            for delivery in over.get("deliveries", []):
                batter = delivery["batter"]
                bowler = delivery["bowler"]
                batter_runs = delivery["runs"]["batter"]

                batting[batter]["runs"] += batter_runs
                batting[batter]["matches"].add(match_id)

                bowling[bowler]["matches"].add(match_id)
                bowling[bowler]["runs_conceded"] += (
                    runs_charged_to_bowler(delivery)
                )

                if legal_delivery(delivery):
                    batting[batter]["balls"] += 1
                    bowling[bowler]["balls"] += 1

                    if batter_runs == 0:
                        batting[batter]["dot_balls"] += 1

                    if delivery["runs"]["total"] == 0:
                        bowling[bowler]["dot_balls"] += 1

                    batting[batter][
                        f"{current_phase}_runs"
                    ] += batter_runs

                    batting[batter][
                        f"{current_phase}_balls"
                    ] += 1

                    bowling[bowler][
                        f"{current_phase}_runs"
                    ] += runs_charged_to_bowler(delivery)

                    bowling[bowler][
                        f"{current_phase}_balls"
                    ] += 1

                if is_boundary(delivery):
                    if batter_runs == 4:
                        batting[batter]["fours"] += 1
                    else:
                        batting[batter]["sixes"] += 1

                    bowling[bowler][
                        "boundaries_conceded"
                    ] += 1

                for wicket in delivery.get("wickets", []):
                    player_out = wicket.get("player_out")

                    if player_out and batter_is_out(wicket):
                        batting[player_out][
                            "dismissals"
                        ] += 1

                    if bowler_gets_wicket(wicket):
                        bowling[bowler]["wickets"] += 1


def batting_row(player, stats):
    balls = stats["balls"]

    return {
        "player": player,
        "batting_matches": len(stats["matches"]),
        "batting_balls": balls,
        "batting_average": rate(
            stats["runs"],
            stats["dismissals"],
        ),
        "strike_rate": rate(
            stats["runs"],
            balls,
            100,
        ),
        "boundary_pct": rate(
            stats["fours"] + stats["sixes"],
            balls,
            100,
        ),
        "dot_ball_pct": rate(
            stats["dot_balls"],
            balls,
            100,
        ),
        "powerplay_sr": rate(
            stats["powerplay_runs"],
            stats["powerplay_balls"],
            100,
        ),
        "middle_sr": rate(
            stats["middle_runs"],
            stats["middle_balls"],
            100,
        ),
        "death_sr": rate(
            stats["death_runs"],
            stats["death_balls"],
            100,
        ),
        "six_rate": rate(
            stats["sixes"],
            balls,
            100,
        ),
    }


def bowling_row(player, stats):
    balls = stats["balls"]

    return {
        "player": player,
        "bowling_matches": len(stats["matches"]),
        "bowling_balls": balls,
        "bowling_economy": rate(
            stats["runs_conceded"],
            balls / 6,
        ),
        "bowling_strike_rate": rate(
            balls,
            stats["wickets"],
        ),
        "bowling_dot_ball_pct": rate(
            stats["dot_balls"],
            balls,
            100,
        ),
        "boundary_conceded_pct": rate(
            stats["boundaries_conceded"],
            balls,
            100,
        ),
        "powerplay_economy": rate(
            stats["powerplay_runs"],
            stats["powerplay_balls"] / 6,
        ),
        "middle_economy": rate(
            stats["middle_runs"],
            stats["middle_balls"] / 6,
        ),
        "death_economy": rate(
            stats["death_runs"],
            stats["death_balls"] / 6,
        ),
        "wickets_per_match": rate(
            stats["wickets"],
            len(stats["matches"]),
        ),
    }


def build_dataset():
    print("Downloading IPL data from Cricsheet...")

    with urlopen(DATA_URL, timeout=60) as response:
        zip_bytes = response.read()

    batting = defaultdict(new_batting_stats)
    bowling = defaultdict(new_bowling_stats)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for filename in archive.namelist():
            if not filename.endswith(".json"):
                continue

            with archive.open(filename) as file:
                match = json.load(file)

            dates = match.get("info", {}).get("dates", [])

            if not dates:
                continue

            year = int(str(dates[0])[:4])

            if year < START_YEAR:
                continue

            process_match(match, Path(filename).stem, batting, bowling)

    batting_rows = {
        player: batting_row(player, stats)
        for player, stats in batting.items()
        if stats["balls"] >= MIN_BATTING_BALLS
    }

    bowling_rows = {
        player: bowling_row(player, stats)
        for player, stats in bowling.items()
        if (
            stats["balls"] >= MIN_BOWLING_BALLS
            and stats["wickets"] > 0
        )
    }

    rows = []

    for player in sorted(
        set(batting_rows) | set(bowling_rows)
    ):
        row = {
            "player": player,
            "source": "Cricsheet IPL JSON",
            "from_year": START_YEAR,
        }

        if player in batting_rows:
            row.update(batting_rows[player])

        if player in bowling_rows:
            row.update(bowling_rows[player])

        if player in batting_rows and player in bowling_rows:
            row["role"] = "All-rounder"
        elif player in batting_rows:
            row["role"] = "Batter"
        else:
            row["role"] = "Bowler"

        rows.append(row)

    df = pd.DataFrame(rows)

    first_columns = [
        "player",
        "role",
        "source",
        "from_year",
    ]

    other_columns = [
        column
        for column in df.columns
        if column not in first_columns
    ]

    df = df[first_columns + other_columns]

    for column in df.select_dtypes(include="number"):
        df[column] = df[column].round(2)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} players to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_dataset()
