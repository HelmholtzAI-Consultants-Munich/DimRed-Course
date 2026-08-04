Introduction
============

Dimensionality reduction compresses high-dimensional data into a smaller set of
informative features, making it easier to visualize, model, and interpret. Many real
datasets, for example gene-expression profiles with tens of thousands of features per
sample, cannot be inspected directly, and a large fraction of their features are
redundant or noisy. Reducing the number of features helps to reveal the structure
that matters, speeds up downstream analyses, and lowers the risk of overfitting.

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
   :widths: 26 24 22 28

   * - Family
     - Original features kept?
     - New features created?
     - Typical strength
   * - Feature transformation
     - no
     - yes (*components*)
     - Compact summaries, good for visualization
   * - Feature aggregation
     - no (grouped)
     - yes (group summaries)
     - Compact, yet close to the original meaning
   * - Feature selection
     - yes (a subset)
     - no
     - Interpretability: keeps the actual measurements

The following pages describe the individual methods in each family, followed by a
page on assessing the stability of feature selection and optimizing its
hyperparameters.
