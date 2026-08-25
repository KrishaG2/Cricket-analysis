import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import BATTING_FEATURES, batting_players, bowling_players, load_players
from maths import pca_from_scratch, role_match_scores, similar_players


st.set_page_config(page_title="CricketIQ", page_icon="🏏", layout="wide")


def nice_name(name):
    return name.replace("_", " ").title()


def percent(value):
    return f"{value * 100:.1f}%"


try:
    players = load_players()
except FileNotFoundError:
    st.error("Player data has not been built yet.")
    st.code("py build_real_data.py")
    st.stop()

batters = batting_players(players).reset_index(drop=True)
bowlers = bowling_players(players).reset_index(drop=True)

st.title("🏏 CricketIQ")
st.caption("Exploring IPL batting styles with linear algebra.")

page = st.sidebar.radio(
    "Explore",
    ["Similar Players", "Build a Player", "Player Map", "Batter vs Bowler"],
)


if page == "Similar Players":
    st.header("Similar Player Finder")

    selected_name = st.selectbox("Choose a batter", batters["player"].tolist())
    selected_index = batters.index[batters["player"] == selected_name][0]
    selected = batters.loc[selected_index]

    col1, col2, col3 = st.columns(3)
    col1.metric("Batting average", f"{selected['batting_average']:.2f}")
    col2.metric("Strike rate", f"{selected['strike_rate']:.2f}")
    col3.metric("Boundary ball %", f"{selected['boundary_pct']:.2f}")

    matrix = batters[BATTING_FEATURES].to_numpy(dtype=float)
    scores = similar_players(matrix, selected_index)

    results = batters[["player", "role"]].copy()
    results["similarity"] = scores
    results = results[results["player"] != selected_name]
    results = results.sort_values("similarity", ascending=False).head(5)
    results["similarity"] = results["similarity"].apply(percent)

    st.subheader("Closest batting profiles")
    st.dataframe(results, hide_index=True, use_container_width=True)

    st.caption("Similarity is calculated from the full batting-stat vector using cosine similarity.")


elif page == "Build a Player":
    st.header("Build a Player")

    st.write(
        "Set the batting profile you are looking for, then compare it with real IPL players."
    )

    target = []

    left, right = st.columns(2)

    for index, feature in enumerate(BATTING_FEATURES):
        column = left if index % 2 == 0 else right

        minimum = float(batters[feature].min())
        maximum = float(batters[feature].max())
        middle = float(batters[feature].median())

        with column:
            target.append(
                st.slider(
                    nice_name(feature),
                    min_value=minimum,
                    max_value=maximum,
                    value=middle,
                    key=f"target_{feature}",
                )
            )

    weights = []

    with st.expander("Adjust feature importance"):
        st.write("Leave these at 1.0 if you want every statistic to count equally.")

        for feature in BATTING_FEATURES:
            weights.append(
                st.slider(
                    nice_name(feature),
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key=f"weight_{feature}",
                )
            )

    matrix = batters[BATTING_FEATURES].to_numpy(dtype=float)

    scores = role_match_scores(
        matrix,
        np.array(target),
        np.array(weights),
    )

    results = batters[["player", "role"]].copy()
    results["match"] = scores
    results = results.sort_values("match", ascending=False).head(5)
    results["match"] = results["match"].apply(percent)

    st.subheader("Best matches")
    st.dataframe(results, hide_index=True, use_container_width=True)

    st.caption(
        "The importance settings are applied as a diagonal matrix before cosine similarity."
    )


elif page == "Player Map":
    st.header("PCA Player Map")

    st.write(
        "Players close together have more similar overall batting profiles. "
        "Hover over a point to see the player and their original statistics."
    )

    minimum_balls = st.slider(
        "Minimum balls faced",
        min_value=250,
        max_value=800,
        value=400,
        step=50,
    )

    map_players = batters[
        batters["batting_balls"] >= minimum_balls
    ].copy().reset_index(drop=True)

    if len(map_players) < 5:
        st.warning("Choose a lower minimum balls faced.")
        st.stop()

    matrix = map_players[BATTING_FEATURES].to_numpy(dtype=float)
    pca = pca_from_scratch(matrix, 2)
    projected = pca["projected"]

    map_data = map_players[
        [
            "player",
            "role",
            "batting_balls",
            "batting_average",
            "strike_rate",
            "boundary_pct",
            "dot_ball_pct",
            "powerplay_sr",
            "middle_sr",
            "death_sr",
            "six_rate",
        ]
    ].copy()

    map_data["PC1"] = projected[:, 0]
    map_data["PC2"] = projected[:, 1]

    selected_player = st.selectbox(
        "Highlight a player",
        ["None"] + map_data["player"].tolist(),
    )

    fig = px.scatter(
        map_data,
        x="PC1",
        y="PC2",
        hover_name="player",
        hover_data={
            "role": True,
            "batting_balls": True,
            "batting_average": ":.2f",
            "strike_rate": ":.2f",
            "boundary_pct": ":.2f",
            "dot_ball_pct": ":.2f",
            "powerplay_sr": ":.2f",
            "middle_sr": ":.2f",
            "death_sr": ":.2f",
            "six_rate": ":.2f",
            "PC1": ":.2f",
            "PC2": ":.2f",
        },
        title="IPL batting-style map",
    )

    if selected_player != "None":
        selected = map_data[map_data["player"] == selected_player]

        fig.add_scatter(
            x=selected["PC1"],
            y=selected["PC2"],
            mode="markers+text",
            text=selected["player"],
            textposition="top center",
            marker={"size": 14},
            name="Selected player",
        )

    st.plotly_chart(fig, use_container_width=True)

    explained = pca["explained_ratio"]
    total_explained = explained[0] + explained[1]

    st.info(
        f"This 2D map keeps {total_explained * 100:.1f}% of the variation "
        "from the original eight batting statistics."
    )

    with st.expander("How PCA built this map"):
        st.write(
            f"PC1 keeps {explained[0] * 100:.1f}% of the variation and "
            f"PC2 keeps {explained[1] * 100:.1f}%."
        )
        st.write(
            "PC1 and PC2 are combinations of the original statistics. "
            "Positive and negative values only show direction; they do not mean good or bad."
        )

        loadings = pd.DataFrame(
            pca["components"],
            index=[nice_name(feature) for feature in BATTING_FEATURES],
            columns=["PC1", "PC2"],
        )

        st.dataframe(loadings.round(3), use_container_width=True)


elif page == "Batter vs Bowler":
    st.header("Batter vs Bowler")

    st.write(
        "Compare a batter's scoring strength with a bowler's economy across different parts of an innings."
    )

    batter_name = st.selectbox("Batter", batters["player"].tolist())
    bowler_name = st.selectbox("Bowler", bowlers["player"].tolist())

    batter = batters[batters["player"] == batter_name].iloc[0]
    bowler = bowlers[bowlers["player"] == bowler_name].iloc[0]

    batter_columns = [
        "strike_rate",
        "powerplay_sr",
        "middle_sr",
        "death_sr",
    ]

    bowler_columns = [
        "bowling_economy",
        "powerplay_economy",
        "middle_economy",
        "death_economy",
    ]

    batter_matrix = batters[batter_columns].to_numpy(dtype=float)
    bowler_matrix = bowlers[bowler_columns].to_numpy(dtype=float)

    batter_min = batter_matrix.min(axis=0)
    batter_max = batter_matrix.max(axis=0)
    batter_range = np.where(batter_max - batter_min == 0, 1, batter_max - batter_min)

    batter_strength = (
        batter[batter_columns].to_numpy(dtype=float) - batter_min
    ) / batter_range

    bowler_min = bowler_matrix.min(axis=0)
    bowler_max = bowler_matrix.max(axis=0)
    bowler_range = np.where(bowler_max - bowler_min == 0, 1, bowler_max - bowler_min)

    scaled_economy = (
        bowler[bowler_columns].to_numpy(dtype=float) - bowler_min
    ) / bowler_range

    bowler_strength = 1 - scaled_economy

    comparison = pd.DataFrame(
        {
            "Phase": ["Overall", "Powerplay", "Middle overs", "Death overs"],
            "Batter strength": batter_strength,
            "Bowler strength": bowler_strength,
        }
    )

    chart_data = comparison.melt(
        id_vars="Phase",
        value_vars=["Batter strength", "Bowler strength"],
        var_name="Player",
        value_name="Scaled strength",
    )

    fig = px.bar(
        chart_data,
        x="Phase",
        y="Scaled strength",
        color="Player",
        barmode="group",
        title=f"{batter_name} vs {bowler_name}",
    )

    st.plotly_chart(fig, use_container_width=True)

    comparison["Difference"] = (
        comparison["Batter strength"] - comparison["Bowler strength"]
    )

    st.dataframe(
        comparison.round(3),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Values are scaled within the batting and bowling datasets. "
        "This is an exploratory comparison, not a match prediction."
    )
