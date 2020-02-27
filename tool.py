from pytorch_pretrained_bert import BertTokenizer
import numpy as np

dd = []
word_count = {}
for line in open("./data/t.txt"):
    w_list = []
    for word in line.strip().split():
        if 'N' in word:
            w = 'N'
        else:
            sub_words = BertTokenizer.from_pretrained('bert-base-uncased').tokenize(word)
            w = sub_words[0]
        word_count[w] = word_count.get(w, 0) + 1
        w_list.append(w)
    w_list = ['[CLS]'] + w_list
    dd.append(w_list)