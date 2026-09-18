import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_components(data_transformed, j=0, k=1, tissuelegend=True, legendloc="upper right", axislabel="Component", title="",
                    data_with_labels = pd.read_csv("../data/tomato_with_targets.txt", index_col=0), 
                    tissue_dict = {'floral': 'gold', 'leaf': 'chartreuse', 'root': 'gray', 'sdling': 'mediumseagreen', 'stem': 'darkgreen', 'veg': 'purple'}, 
                    species_condition_dict = {'penn.Sun': 's', 'penn.Sh': 'd', 'M82.Sun': '^', 'M82.Sh': 'v'}):
    for i in range(0,data_transformed.shape[0]):
        plt.scatter(data_transformed[i,j], data_transformed[i,k], marker=species_condition_dict['.'.join([data_with_labels.iloc[i]['species'], data_with_labels.iloc[i]['position']])], c=tissue_dict[data_with_labels.iloc[i]['tissue']])
    plt.xlabel("{0} {1}".format(axislabel, j))  
    plt.ylabel("{0} {1}".format(axislabel, k))
    plt.suptitle("{}".format(title))
    if tissuelegend:
        plt.legend(tissue_dict, loc=legendloc)
    else:
        def make_markers(ind):
            return(plt.Line2D([], [], color='black', marker=list(species_condition_dict.values())[ind], linestyle='None'))
        plt.legend([make_markers(l) for l in range(0,len(species_condition_dict.values()))], species_condition_dict.keys(), loc=legendloc)
    plt.show()

def compute_neighbor_confusion(dist1, dist2, k=4):
    np.fill_diagonal(dist1, np.inf)
    np.fill_diagonal(dist2, np.inf)
    confusion_matrix = np.zeros((2, 2), dtype=np.int8)
    
    for i in range(dist1.shape[0]):
        s1 = np.argsort(dist1[i,])
        w1 = s1[:k]
        s2 = np.argsort(dist2[i,])
        w2 = s2[:k]
        b = len(np.setdiff1d(w1, w2))
        c = len(np.setdiff1d(w2, w1))
        d = len(np.intersect1d(w1, w2))
        a = dist1.shape[1] - 1 - b - c - d # instance itself does not count as neighbor or non-neighbor, therefore -1
        currmatrix = np.array([[a, c], [b, d]])
        confusion_matrix += currmatrix
    
    df = pd.DataFrame(data = confusion_matrix,  
                index = ['No neighbor (original space)', 'Neighbor (original space)'],  
                columns = ['No neighbor (display)', 'Neighbor (display)']) 
    return df

def extract_part(x, part, spl='.'):
    s = [elem.split(spl) for elem in x]
    return [elem[part] for elem in s]


def get_data(go, data, feature_names, rd = pd.read_csv("../data/goslim_to_genes.txt", header=0, delimiter='\t')):
    df = pd.DataFrame(data)
    df.columns = extract_part(feature_names, part=0)
    w = rd[rd['GO term'] == go].index[0]
    s = rd.iloc[w, 1].split(',')
    dat1raw = df.loc[:, df.columns.intersection(s)]
    return(dat1raw)

def clara_kmedoids(data, n_clusters, n_samples=5, sample_size=None, random_state=None):
    """Cluster the rows of `data` with k-medoids, using the CLARA strategy.

    k-medoids picks actual data points as cluster centres, so it works from
    pairwise distances alone. Solving it exactly needs the full distance
    matrix, which is not possible here: with ~28,000 genes that matrix would
    be about 6 GB.

    CLARA (Kaufman and Rousseeuw, 1990) avoids that. It draws a small random
    sample, solves k-medoids on the sample only, and then assigns every
    remaining point to its nearest medoid. Repeating this over several samples
    and keeping the cheapest result gives a good clustering at a fraction of
    the cost.

    Returns (labels, medoid_indices).
    """
    from sklearn.metrics import pairwise_distances
    import kmedoids

    rng = np.random.default_rng(random_state)
    n_points = data.shape[0]
    if sample_size is None:
        sample_size = min(n_points, 40 + 2 * n_clusters)

    best_cost, best_medoids = np.inf, None
    for _ in range(n_samples):
        sample = rng.choice(n_points, size=sample_size, replace=False)
        distances = pairwise_distances(data[sample])
        result = kmedoids.fasterpam(distances, n_clusters, random_state=random_state)
        medoids = sample[result.medoids]
        # Total distance from every point to its nearest medoid.
        cost = pairwise_distances(data, data[medoids]).min(axis=1).sum()
        if cost < best_cost:
            best_cost, best_medoids = cost, medoids

    labels = pairwise_distances(data, data[best_medoids]).argmin(axis=1)
    return labels, best_medoids
