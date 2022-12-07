# LM4NLR
Language Modeling for Neural Multi-hop Logical Reasoning over Knowledge Graphs


## Requirements
    python == 3.8.5
    torch == 1.8.1
    numpy == 1.19.2
    pandas == 1.0.1
    tqdm == 4.61.0
    gdown == 3.3.1


## Preprocessing

- Generating More General Queries
    > `TODO: Mark/Jingbo`

- Extracting Pretrained Entity Name Embeddings
    > `python extract_emb.py ./private_data/id2ent.pkl ./private_data/emb768.pkl`

- Reducing the Dimensionality of Entity Name Embeddings
    > `python pca.py`

- You may also download the preprocessed dataset to `./private_data`
    > `gdown https://drive.google.com/uc?id=1apP8i_bDiNdpgfoyzl8nfILJTe5EWURy`

## Run

- Reproducing Query Type Generalization Performance on NELL995 
    * GQE
        > `CUDA_VISIBLE_DEVICES=0 nohup python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 800 -g 24 -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --print_on_screen >vec.log 2>&1 &`

    * Query2Box
        > `CUDA_VISIBLE_DEVICES=0 nohup python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 400 -g 60 -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 -betam "(1600,2)" --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --print_on_screen >beta.log 2>&1`
    
    * BetaE
        > `CUDA_VISIBLE_DEVICES=0 nohup python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 400 -g 24 -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 -boxm "(none,0.02)" --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --print_on_screen >box.log 2>&1 &`

- Reproducing Language Model Enhanced Reasoning Performance on NELL995 
    * GQE
        > `CUDA_VISIBLE_DEVICES=0 nohup python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 800 -g 24 -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --word_embed_path ./private_data/emb128.pkl --print_on_screen >vec_lm.log 2>&1 &`
    
    * Query2Box
        > `TODO: PY`

    * BetaE
        > `TODO: PY`


## Acknowledgement
This repo is based on the official implementation of `GQE`, `Query2Box`, and `BetaE` provided at `https://github.com/snap-stanford/KGReasoning`.