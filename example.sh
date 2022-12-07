# old examples
#CUDA_VISIBLE_DEVICES=0 nohup python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 800 -g 24 -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --print_on_screen >vec.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=1 nohup python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 400 -g 60 -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 -betam "(1600,2)" --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --print_on_screen >beta.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=0 nohup python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 400 -g 24 -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 -boxm "(none,0.02)" --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --print_on_screen >box.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test --data_path ./private_data -n 128 -b 512 -d 800 -g 24 -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 --tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" --word_embed_path ./private_data/emb128.pkl --print_on_screen

# new version
CUDA_VISIBLE_DEVICES=0 python main.py \
--cuda --do_train --do_valid --do_test --data_path ./private_data \
-n 128 -b 512 -d 800 -g 24 -lr 0.0001 \
--max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
--tasks "1p.2p.3p.4p.5p.2i.3i.4i.5i.ip.pi.2u.up" \
--print_on_screen

