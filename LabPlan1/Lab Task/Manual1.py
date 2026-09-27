
# --- Q1: Outlier Detection & Boxplots ---
# IQR is just the middle 50% of our data. We use it to set upper and lower boundaries.
# Any data point that falls too far outside these boundaries is flagged as an outlier.
# On a boxplot, the main box holds the normal data, and outliers show up as isolated dots.

# --- Q2: Log-Transformation for Skewed Data ---
# Right-skewed data means a few massive numbers are creating a long tail on the right side.
# Log-transforming crushes these massive numbers down and pulls that long tail inward.
# This makes the data look more like a balanced, symmetric bell curve on a histogram.

# --- Q3: Why Feature Scaling Matters for KNN/SVM ---
# Distance-based models calculate the physical distance between points to make decisions.
# If one feature is in the thousands (like Area) and another is a small decimal, the big one wins.
# Scaling normalizes everything to the same scale so every feature gets an equal, fair vote.

# --- Q4: PCA and Class Separability ---
# PCA acts like a camera, condensing 30 features into just 2 main components we can easily look at.
# If the two groups form separate clusters on a scatter plot, the classes are highly separable.
# This tells us that a machine learning model will have a very easy time drawing a line between them.

# --- Q5: SVM vs. KNN Logic ---
# SVM is boundary-focused; it tries to draw the widest possible highway line between classes.
# KNN is neighborhood-focused; it just looks at the closest points around it to make a guess.
# SVM generally performs better here because boundary lines generalize better than dense distance checks.

# --- Q6: Preprocessing Step with the Biggest Impact ---
# Feature scaling had the absolute biggest impact on this dataset.
# Without scaling, the huge numerical differences cause distance checks to completely ignore vital features.
# Standardizing the scales is the single switch that allows SVM and KNN to actually work correctly.