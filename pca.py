import torch
import pickle
from sklearn.decomposition import PCA
import numpy as np
import pdb

def pca(addr):
    embed_list = pickle.load(open(addr, "rb"))
    for target in [128, 256, 512]:
        print(f'Compressing to {target} dimension.')
        pca = PCA(n_components=target)
        embeds_new = pca.fit_transform(embed_list)
        pickle.dump(embeds_new.tolist(), open("./private_data/emb" + str(target) + ".pkl", "wb"))


if __name__ == '__main__':
    pca("./private_data/emb768.pkl")
