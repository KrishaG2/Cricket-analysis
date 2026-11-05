# Cricket analysis 

A small cricket analytics project I built to apply linear algebra to real IPL data.

I wanted the project to stay focused on the maths, so the app is deliberately simple and the main calculations are written directly with NumPy.

## What it does

### Similar Player Finder

Pick an IPL batter and find the players with the closest overall batting profile.

This uses player vectors, feature scaling, dot products, vector norms and cosine similarity.

### Build a Player

Create a batting profile using the sliders and find the real players who match it most closely.

The optional feature weights are applied with a diagonal transformation matrix before the similarity calculation.

### PCA Player Map

Each batter is described by eight statistics, which means the original player data is eight-dimensional.

The PCA page reduces that to two dimensions so the players can be plotted. Players close together have more similar overall batting profiles.

I implemented the PCA steps with NumPy using standardisation, a covariance matrix, eigenvalues, eigenvectors and projection rather than using scikit-learn's PCA class.

### Batter vs Bowler

Compare a batter and bowler across overall, powerplay, middle-over and death-over performance.

The values are scaled so the two profiles can be viewed side by side. It is an exploratory comparison, not a prediction model.

## Data

The project uses IPL ball-by-ball data from [Cricsheet](https://cricsheet.org/).

`build_real_data.py` downloads the IPL JSON archive and calculates the player statistics used by the app. I currently use matches from 2022 onwards and require a minimum sample before a player is included.

Cricsheet data is used with attribution. Its data information is available [here](https://cricsheet.org/register/).

## Batting features

- batting average
- strike rate
- boundary-ball percentage
- dot-ball percentage
- powerplay strike rate
- middle-overs strike rate
- death-overs strike rate
- six rate

## Project structure

```text
Cricket-analysis/
├── app.py
├── maths.py
├── data_utils.py
├── build_real_data.py
├── maths_notes.md
├── requirements.txt
├── .gitignore
└── data/
    ├── README.md
    └── players.csv
```

`players.csv` is created when the data-building script is run.

## Run locally

Install the packages:

```bash
py -m pip install -r requirements.txt
```

Build the player data:

```bash
py build_real_data.py
```

Start the app:

```bash
py -m streamlit run app.py
```

## Why I made it

The aim of the project is not to build a cricket prediction system. It is a practical way for me to use ideas such as vectors, matrix transformations, cosine similarity, covariance, eigenvectors and PCA on data that I find interesting.
