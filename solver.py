from model import *
from utils import *
import os
import subprocess
import random

class Solver():
    def __init__(self, args):
        self.args = args
        self.model_dir = make_save_dir(args.model_dir)
        if not os.path.exists(os.path.join(self.model_dir, 'code')):
            os.makedirs(os.path.join(self.model_dir, 'code'))

        self.data_utils = data_utils(args)
        # V=len(self.data_utils.new_vocab) #字典长度
        V=2000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1) #size为src_vocab
        self.model = make_model(V,V,2)
        self.model = self.model.to(self.device)



    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])
        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(name)
                ttt = 1
                for s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:', tt)
        model_opt = get_std_opt(self.model)

        # data_yielder = self.data_utils.train_data_yielder()
        # optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        # optim = BertAdam(self.model.parameters(), lr=1e-4)

        for epoch in range(self.args.num_step):
            print("Epoch %d :" %(epoch))
            self.model.train()
            run_epoch(self.data_utils.data_load(self.args.batch_size), self.model,
                      SimpleLossCompute(self.model.generator, self.criterion, model_opt), self.device)


        print('saving!!!!')
        model_name = 'model.pth'
        state = {'step': 0, 'state_dict': self.model.state_dict()}
        torch.save(state, os.path.join(self.model_dir, model_name))

    def test(self, threshold=0.8):
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()

        # with open(self.args.train_path, "r") as f:
        #     all = [line.strip() for line in f.readlines()]
        # all_dict = []
        # for i in all:
        #     all_dict.append(ast.literal_eval(i))
        #

        # src = Variable(torch.from_numpy(np.expand_dims(self.data_utils.text2id(all_dict[0].get("code").strip()),axis=0)).long())
        src = Variable(torch.LongTensor([[3, 6, 7, 8, 5, 6, 7, 8, 9, 10,12,0]]))
        print(self.data_utils.id2sent([3, 6, 7, 8, 5, 6, 7, 8, 9, 10,12]))
        src_mask = Variable(torch.ones(1, 1, 12)) #一个 一行11列的矩阵
        res = greedy_decode(self.model, src, src_mask, max_len=15, start_symbol=1)#max_len为输出的语句长度
        print(res)
        print(self.data_utils.id2sent(np.squeeze(res.cpu().numpy())))