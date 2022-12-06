from transformers import BertTokenizer, BertModel
import sys
import pickle
from tqdm import tqdm
import torch
import pdb

def list_to_embedding(model, tokenizer, text_list):

    encoded_input = tokenizer(text_list, padding=True, return_tensors='pt').to('cuda:0')
    output = model(**encoded_input)
    embed_list = output.pooler_output

    return embed_list

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").to('cuda:0')
    
    source_addr = sys.argv[1]
    print(source_addr)
    f = open(source_addr, 'rb')
    source_dic = pickle.load(f)
    target_addr = sys.argv[2]
    output = open(target_addr, 'wb')

    source_list = []
    for i in tqdm(range(len(source_dic))):
        v = source_dic[i]
        is_digit = True
        for c in v:
            if c.isalpha():
                is_digit = False
                break

        if is_digit:
            v = "geographical coordinates " + v.replace(",", " ")
        else:
            v = v[8:]
            v = v.replace("_", " ")
        source_list.append(v)

    target_list = []
    for target in tqdm(source_list):
        emb_each = list_to_embedding(model, tokenizer, target).detach().cpu().numpy().tolist()
        target_list.append(emb_each)
    
    # print(target_list.shape, type(target_list))
    pickle.dump(target_list, output)


if __name__ == '__main__':
    main()