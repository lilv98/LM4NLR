CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/NELL-q2b -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" \
  --word_embed_path  embeds/embeds768.pkl \
  --print_on_screen
