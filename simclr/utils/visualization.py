import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.manifold import TSNE


def visualize_embeddings(embeddings, labels, save_path, num_categories):

    print('Visualizing embeddings...')

    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)

    # Plot those points as a scatter plot and label them based on the pred labels

    cmap_name = 'prism'
    if num_categories <= 10:
        cmap_name = 'tab10'

    cmap = cm.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(40,40))

    for lab in range(num_categories):
        indices = labels.squeeze()==lab
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)

    print('Showing...')
    plt.show()
    plt.savefig(save_path)
