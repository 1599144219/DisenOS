import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import time

from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from args import read_options
from data_loader import *
from matcher import *
from tensorboardX import SummaryWriter
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Trainer(object):

    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)

        self.meta = not self.no_meta

        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            self.load_symbol2id()
            use_pretrain = False
            self.load_embed()
        else:
            # 加载预训练的表示
            self.load_embed()
        self.use_pretrain = use_pretrain

        # self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        # self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, dropout=self.dropout,
                                    batch_size=self.batch_size,
                                    process_steps=self.process_steps, aggregate=self.aggregate,
                                    factors=self.k_factors, ent_embeddings=self.ent_embeddings,
                                    rel_embeddings=self.rel_embeddings)
        #self.matcher = nn.DataParallel(self.matcher)
        self.matcher.cuda()
        self.batch_nums = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200000], gamma=0.5)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))
        
        self.dev_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        self.test_tasks = json.load(open(self.dataset + '/test_tasks.json'))

    def load_symbol2id(self):
        '''
        当指定不使用TransE训练的表示作为初始向量时，调用这个函数即可
        只会将实体和关系 与 id进行对应，不会初始化对应的表示
        '''
        self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.symbol2id = (self.ent2id, self.rel2id)

        self.ent_embeddings = None
        self.rel_embeddings = None

    def load_embed(self):
        self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        # self.symbol2id = (self.ent2id, self.rel2id)

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        ent_embeddings = []
        with open(self.dataset + '/entity2vec.' + self.embed_model) as f:
            for line in f:
                # ent_vec = []
                tmp = [float(val) for val in line.strip().split()]
                # for i in range(self.k_factors):
                #     ent_vec += tmp
                ent_embeddings.append(tmp)
        self.ent_embeddings = torch.FloatTensor(np.array(ent_embeddings))

        rel_embeddings = []
        with open(self.dataset + '/relation2vec.' + self.embed_model) as f:
            for line in f:
                tmp = [float(val) for val in line.strip().split()]  
                rel_embeddings.append(tmp)
        self.rel_embeddings = torch.FloatTensor(np.array(rel_embeddings))
    def build_connection(self, max_=100):
        self.connections_in = (np.ones((self.num_ents, max_, 2))).astype(int)
        self.connections_out = (np.ones((self.num_ents, max_, 2))).astype(int)

        self.e1_rele2_in = defaultdict(list)  
        self.e1_rele2_out = defaultdict(list)  
        self.e1_degrees = defaultdict(int)

        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2_out[e1].append((self.rel2id[rel], self.ent2id[e2]))
                self.e1_rele2_in[e2].append((self.rel2id[rel], self.ent2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors_out = self.e1_rele2_out[ent]
            neighbors_in = self.e1_rele2_in[ent]
            if len(neighbors_out) > max_:
                neighbors_out = neighbors_out[:max_]
            if len(neighbors_in) > max_:
                neighbors_in = neighbors_in[:max_]
            degrees[ent] = len(neighbors_out) + len(neighbors_in)
            self.e1_degrees[id_] = len(neighbors_out) + len(neighbors_in)
            for idx, _ in enumerate(neighbors_in):
                self.connections_in[id_, idx, 0] = _[0]
                self.connections_in[id_, idx, 1] = _[1]
            for idx, _ in enumerate(neighbors_out):
                self.connections_out[id_, idx, 0] = _[0]
                self.connections_out[id_, idx, 1] = _[1]
        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.matcher.state_dict(), path)

    def load(self):
        self.matcher.load_state_dict(torch.load(self.save_path))

    # def get_meta(self, left, right):
    #     left_connections = Variable(
    #         torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).cuda()
    #     left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
    #     right_connections = Variable(
    #         torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).cuda()
    #     right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
    #     return (left_connections, left_degrees, right_connections, right_degrees)

    def get_meta(self, left, right):
        left_connections_out = Variable(
            torch.LongTensor(np.stack([self.connections_out[_, :, :] for _ in left], axis=0))).cuda()
        left_connections_in = Variable(
            torch.LongTensor(np.stack([self.connections_in[_, :, :] for _ in left], axis=0))).cuda()
        left_connections = (left_connections_out, left_connections_in)
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()

        right_connections_out = Variable(
            torch.LongTensor(np.stack([self.connections_out[_, :, :] for _ in right], axis=0))).cuda()
        right_connections_in = Variable(
            torch.LongTensor(np.stack([self.connections_in[_, :, :] for _ in right], axis=0))).cuda()
        right_connections = (right_connections_out, right_connections_in)
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()

        return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        logging.info('START TRAINING...')

        best_hits10 = 0.0
        best_batches = 0
        losses = deque([], self.log_every)
        margins = deque([], self.log_every)

        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.ent2id,
                                   self.e1rel_e2):
            start = time.time()
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data
            
            reg = 0
            reg_intra = 0
            entities = list(np.array(support).flatten())
            for name, param in self.matcher.named_parameters():
                if name == "ent_embeddings.weight":
                    ent_emb = param
                elif name == "linear1.weight":
                    w1 = param
                elif name == "linear1.bias":
                    b1 = param
                elif name == "linear2.weight":
                    w2 = param
                elif name == "linear2.bias":
                    b2 = param
                elif name == "linear3.weight":
                    w3 = param
                elif name == "linear3.bias":
                    b3 = param
            ent1 = (torch.mm(ent_emb, w1) + b1).unsqueeze(1)
            ent2 = (torch.mm(ent_emb, w2) + b2).unsqueeze(1)
            ent3 = (torch.mm(ent_emb, w3) + b3).unsqueeze(1)
            embeds = torch.cat([ent1, ent2, ent3], dim=1)

            # inter-factor diversity    reg: 254.5183
            for entity in entities:
                embed = embeds[entity]
                temp = torch.mm(embed, torch.t(embed))
                reg += torch.mean(torch.abs(temp - torch.diag(temp)))
        
            '''
            # intra-factor consistency  reg: 319
            embeds = embeds.view(ent_emb.shape[0], -1)
            embeds_ = torch.split(embeds[entities], ent_emb.shape[1], dim=1)
            total_embeds = torch.split(embeds, ent_emb.shape[1], dim=1)
            for i in range(self.k_factors):
                mean_embed = torch.mean(total_embeds[i], dim=0, keepdim=True)
                reg_intra += torch.mean(F.pairwise_distance(mean_embed, embeds_[i]))
            '''
            self.batch_nums += 1
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).cuda()
            query = Variable(torch.LongTensor(query)).cuda()
            false = Variable(torch.LongTensor(false)).cuda()

            if self.no_meta:
                # for ablation
                query_scores = self.matcher(query, support)
                false_scores = self.matcher(false, support)
            else:
                query_scores = self.matcher(query, support, query_meta, support_meta)
                false_scores = self.matcher(false, support, false_meta, support_meta)
             
            margin_ = query_scores - false_scores
            margins.append(margin_.mean().item())

            # max(0, margin + false_scores - query_scores)
            #loss = F.relu(self.margin - margin_).mean() 
            loss = F.relu(self.margin - margin_).mean() + 2 * reg
            #loss = F.relu(self.margin - margin_).mean() + reg + reg_intra

            #self.writer.add_scalar('MARGIN', np.mean(margins), self.batch_nums)

            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.batch_nums % self.eval_every == 0:
                hits10, hits5, mrr = self.eval(meta=self.meta)
                #self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                #self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                #self.writer.add_scalar('MRR', mrr, self.batch_nums)

                self.save()

                if mrr > best_hits10:
                    self.save(self.save_path + '_bestHits10')
                    best_hits10 = mrr
                    best_batches = self.batch_nums
            if self.batch_nums % self.log_every == 0:
                lr = self.optim.param_groups[0]['lr']
                # logging.info('AVG_BATCH_LOSS: {:.2f} AT STEP {}'.format(np.mean(losses), self.batch_nums))
                logging.info(
                    'Batch: {:d}, AVG_BATCH_LOSS: {:.6f}, lr: {:.6f}, cost time:{:.4f}'.format(self.batch_nums, np.mean(losses), lr, (time.time() - start) * self.log_every))
                #self.writer.add_scalar('Avg_batch_loss', np.mean(losses), self.batch_nums)

            self.scheduler.step()
            if self.batch_nums == self.max_batches:
                self.save()
                break

            if self.batch_nums - best_batches > self.eval_every * 10:
                logging.info('Early stop!')
                self.save()
                break

    def eval(self, mode='dev', meta=False):
        
        self.matcher.eval()

        # symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = self.dev_tasks
        else:
            test_tasks = self.test_tasks
 
        rel2candidates = self.rel2candidates

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []

        for query_ in test_tasks.keys():
            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []

            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[self.ent2id[triple[0]], self.ent2id[triple[2]]] for triple in support_triples]

            if meta:
                support_left = [self.ent2id[triple[0]] for triple in support_triples]
                support_right = [self.ent2id[triple[2]] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)
           
            support = Variable(torch.LongTensor(support_pairs)).cuda()
            for triple in test_tasks[query_][few:]:
                true = triple[2]
                query_pairs = []
                query_pairs.append([self.ent2id[triple[0]], self.ent2id[triple[2]]])
                if meta:
                    query_left = []
                    query_right = []
                    query_left.append(self.ent2id[triple[0]])
                    query_right.append(self.ent2id[triple[2]])

                for ent in candidates:
                    if (ent not in self.e1rel_e2[triple[0] + triple[1]]) and ent != true:
                        query_pairs.append([self.ent2id[triple[0]], self.ent2id[ent]])
                        if meta:
                            query_left.append(self.ent2id[triple[0]])
                            query_right.append(self.ent2id[ent])

                query = Variable(torch.LongTensor(query_pairs)).cuda()

                if meta:
                    query_meta = self.get_meta(query_left, query_right)
                    scores = self.matcher(query, support, query_meta, support_meta)
                    scores.detach()
                    scores = scores.data

                else:
                    scores = self.matcher(query, support)
                    scores.detach()
                    scores = scores.data

                scores = scores.cpu().numpy()
                sort = list(np.argsort(scores))[::-1]
                rank = sort.index(0) + 1  
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)
            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(query_, np.mean(hits10_),
                                                                                              np.mean(hits5_),
                                                                                              np.mean(hits1_),
                                                                                              np.mean(mrr_)))
            logging.info('Number of candidates: {}, number of test examples {}'.format(len(candidates), len(hits10_)))

        
        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MRR: {:.3f}'.format(np.mean(mrr)))

        self.matcher.train()

        return np.mean(hits10), np.mean(hits5), np.mean(mrr)

    def test_(self):
        self.load()
        logging.info('Pre-trained model loaded')
        #self.eval(mode='dev', meta=self.meta)
        self.eval(mode='test', meta=self.meta)


if __name__ == '__main__':
    args = read_options()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)
    if args.test:
        trainer.test_() 
    else:
        trainer.train() 
