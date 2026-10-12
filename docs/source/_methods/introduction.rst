Introduction
============

Dimensionality reduction condenses high-dimensional data into a smaller set of
informative features, making them easier to visualize and model. Reducing
the number of features helps to reveal the structure that matters, speeds up
downstream analyses, and lowers the risk of overfitting.

High-dimensional data are affected by the **curse of dimensionality**: as the number
of features grows, the data points become sparse and the distances between them less
informative, so underlying patterns are harder to find and models overfit more easily.
Working in a smaller, well-chosen feature space avoids this problem to a large extent. A common
application is **gene expression analysis**, where each data instance can carry tens of
thousands of features that cannot be inspected directly. Dimensionality reduction is a
routine first step in gene expression studies, and likewise for the analysis of image, text, and sensor data. It is used for data exploration, visualization, and preprocessing steps to prepare modeling.

The methods in this course fall into three broad families, differing in how they use the original features.


Feature transformation
----------------------

Feature transformation methods **compute new features (components)** based on all original features, 
keeping as much of the relevant information as
possible. In linear methods (PCA and ICA), each component is a weighted sum of
the original features, while nonlinear methods (NMDS, t-SNE, UMAP and **autoencoders**)
can capture more complex, nonlinear mappings and combinations of orogonal features. 
Autoencoders are the neural-network member of this family.


Feature aggregation
-------------------

Feature aggregation methods **group related original features and summarize each group**
into a single aggregated feature, for example the mean expression of a set of genes
that share a biological function. The result is far more compact while staying close
to the original measurements.


Feature selection
-----------------

Feature selection methods **keep a subset of the original features** and discard the rest.
The low-dimensional data instances are still represented by original features,
which is valuable if interpretability of visualization  or modeling results matters. A final topic,
the **stability** of feature selection, takes into acocunt how reliable that chosen subset
is across different training data for the task at hand.


At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 28 24 22 26

   * - Method
     - Keeps original features?
     - Creates new features?
     - Interpretability
   * - Feature selection
     - Yes
     - No
     - High
   * - Feature transformation
     - No
     - Yes
     - Lower
   * - Feature aggregation
     - No
     - Yes
     - Moderate (depends on method)

The following pages describe the individual methods in each family, followed by a
page on assessing the stability of feature selection and optimizing its
hyperparameters.
