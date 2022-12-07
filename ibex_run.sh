#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J LM4NLR
#SBATCH -o LM4NLR.%J.out
#SBATCH -e LM4NLR.%J.err
#SBATCH --time=5:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=[v100]

python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 800 -g 24 -lr 0.0001 --max_steps 4501 --cpu_num 1 --geo vec --valid_steps 150 --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --word_embed_path ./private_data/emb128.pkl --print_on_screen
