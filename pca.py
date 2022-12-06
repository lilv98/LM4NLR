import torch
import pickle
from sklearn.decomposition import PCA
import numpy as np


def pca(addr):

    embed_list = pickle.load(open(addr, "rb"))
    embeds = np.array(embed_list)
    print(len(embeds), len(embeds[0]))

    # embed_list = torch.tensor(embed_list)
    # embed_list = torch.squeeze(embed_list, 1)
    # pickle.dump(embed_list.tolist(), open("embeds/embeds768.pkl", "wb"))

    for target in [128, 256, 512]:

        pca = PCA(n_components=target)
        embeds_new = pca.fit_transform(embed_list)
        pickle.dump(embeds_new.tolist(), open("embeds/embeds" + str(target) + ".pkl", "wb"))


if __name__ == '__main__':
    pca("embeds/embeds768.pkl")
