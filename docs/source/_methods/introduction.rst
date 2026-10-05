Introduction
============

Dimensionality reduction compresses high-dimensional data into a smaller set of
informative features, making it easier to visualize, model, and interpret. Reducing
the number of features helps to reveal the structure that matters, speeds up
downstream analyses, and lowers the risk of overfitting.

High-dimensional data is affected by the **curse of dimensionality**: as the number
of features grows, the data points become sparse and the distances between them less
informative, so genuine patterns are harder to find and models overfit more easily.
Working in a smaller, well-chosen feature space avoids much of this problem. A common
application is **gene-expression analysis**, where each sample can carry tens of
thousands of features that cannot be inspected directly; dimensionality reduction is a
routine first step there, and likewise for imaging, text, and sensor data, for
exploration, visualization, and preprocessing before modeling.

The methods in this course fall into three broad families, distinguished by what they
do with the original features.


Feature transformation
----------------------

Feature transformation **creates new features (*components*)** by combining or
re-expressing the original ones, keeping as much of the relevant information as
possible. The linear methods (PCA and ICA) form each component as a weighted sum of
the originals, while the nonlinear methods (NMDS, t-SNE, UMAP and **autoencoders**)
can capture more complex, nonlinear structure. Autoencoders are the neural-network,
nonlinear member of this same family.


Feature aggregation
-------------------

Feature aggregation **groups related original features and summarizes each group**
into a single aggregated feature, for example the mean expression of a set of genes
that share a biological function. The result is far more compact while staying close
to the original measurements.


Feature selection
-----------------

Feature selection **keeps a subset of the original features** and discards the rest.
The reduced data is still expressed in the original, fully interpretable features,
which is valuable when it matters *which* measurements drive a result. A final topic,
the **stability** of feature selection, then looks at how reliable that chosen subset
is across different training data.


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
