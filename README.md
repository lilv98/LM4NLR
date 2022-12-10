# LM4NLR
Language Modeling for Neural Multi-hop Logical Reasoning over Knowledge Graphs


## Requirements
    python == 3.8.5
    torch == 1.8.1
    numpy == 1.19.2
    pandas == 1.0.1
    tqdm == 4.61.0
    gdown == 3.3.1
    tensorboardx == 2.5.1


## Preprocessing

- Generating More General Queries
    > `python create_mark.py --test_num 1000`

- Extracting Pretrained Entity Name Embeddings
    > `python extract_emb.py ./private_data/id2ent.pkl ./private_data/emb768.pkl`

- Reducing the Dimensionality of Entity Name Embeddings
    > `python pca.py`

- You may also download the preprocessed dataset to `./private_data`
    > `gdown https://drive.google.com/uc?id=1apP8i_bDiNdpgfoyzl8nfILJTe5EWURy`

## Run

- To reproduce the Query Type Generalization and Language Model Enhanced Reasoning experimental results on NELL995, please refer to the commands provided in `examples.sh`.

## Acknowledgement
This repo is based on the official implementation of `GQE`, `Query2Box`, and `BetaE` provided at `https://github.com/snap-stanford/KGReasoning`.
