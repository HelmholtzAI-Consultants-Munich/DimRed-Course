import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_components(data_transformed, j=0, k=1, tissuelegend=True, legendloc="upper right", 
                    data_with_labels = pd.read_csv("../data/tomato_with_targets.txt", index_col=0), 
                    tissue_dict = {'floral': 'gold', 'leaf': 'chartreuse', 'root': 'gray', 'sdling': 'mediumseagreen', 'stem': 'darkgreen', 'veg': 'purple'}, 
                    species_condition_dict = {'penn.Sun': 's', 'penn.Sh': 'd', 'M82.Sun': '^', 'M82.Sh': 'v'}):
    for i in range(0,data_transformed.shape[0]):
        plt.scatter(data_transformed[i,j], data_transformed[i,k], marker=species_condition_dict['.'.join([data_with_labels.iloc[i]['species'], data_with_labels.iloc[i]['position']])], c=tissue_dict[data_with_labels.iloc[i]['tissue']])
    plt.xlabel("Component {}".format(j)) 
    plt.ylabel("Component {}".format(k))
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
    confusion_matrix = np.zeros((2, 2))
    
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