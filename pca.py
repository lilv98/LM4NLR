import torch
import pickle
from sklearn.decomposition import PCA
import numpy as np
import pdb

def pca(addr):

    embed_list = pickle.load(open(addr, "rb"))
    embeds = np.array(embed_list).squeeze(axis=1)
    print(len(embeds), len(embeds[0]))

    # embed_list = torch.tensor(embed_list)
    # embed_list = torch.squeeze(embed_list, 1)
    # pickle.dump(embed_list.tolist(), open("embeds/embeds768.pkl", "wb"))

    for target in [128, 256, 512]:

        pca = PCA(n_components=target)
        embeds_new = pca.fit_transform(embeds)
        pickle.dump(embeds_new.tolist(), open("./private_data/emb" + str(target) + ".pkl", "wb"))


if __name__ == '__main__':
    pca("./private_data/emb768.pkl")
