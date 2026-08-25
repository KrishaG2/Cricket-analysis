# Maths notes

These are the main linear algebra ideas used in CricketIQ.

## Player vectors

A batter is represented by eight statistics:

```text
[
    batting average,
    strike rate,
    boundary percentage,
    dot-ball percentage,
    powerplay strike rate,
    middle-overs strike rate,
    death-overs strike rate,
    six rate
]
```

One player is a vector. Putting all the player vectors together gives a matrix.

## Scaling

The statistics use different ranges, so I scale them before comparing players:

```text
(value - minimum) / (maximum - minimum)
```

This puts each feature roughly between 0 and 1.

## Cosine similarity

For two player vectors `a` and `b`:

```text
cos(theta) = (a · b) / (||a|| ||b||)
```

The Similar Player Finder uses this to compare the direction of two player profiles.

## Feature weights

Build a Player lets each feature have a different importance.

The weights are placed on the diagonal of a matrix and the player vectors are multiplied by it before similarity is calculated.

## PCA

The Player Map starts with eight-dimensional player vectors.

First the data is standardised:

```text
Z = (X - mean) / standard deviation
```

Then the covariance matrix is calculated:

```text
C = (Z^T Z) / (n - 1)
```

The eigenvectors of the covariance matrix give the new PCA directions. The eigenvalues tell us how much variation each direction keeps.

The player matrix is then projected onto the first two eigenvectors:

```text
X_2D = ZW
```

The result is a 2D map. Players close together on the map have more similar overall profiles.

PC1 and PC2 are not cricket statistics themselves. They are combinations of the original statistics, and their positive/negative signs only show direction.
