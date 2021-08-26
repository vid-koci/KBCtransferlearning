import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bisect import bisect_left

import encoder
import random
import tqdm

class ConvE(nn.Module):

    def __init__(self, data, rel_encoder, ent_encoder, dimension=300, dropout=0.1):
        super(ConvE,self).__init__()
        self.data = data
        self.dimension = dimension
        self.dropout = dropout
        self.rel_encoder = rel_encoder
        self.ent_encoder = ent_encoder
        self.inp_drop = nn.Dropout(self.dropout)
        self.hidden_drop = nn.Dropout(self.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(self.dropout)

        self.conv1 = nn.Conv2d(1,32,(3,3))#Exact architecture follows Gupta et al, CaRe model
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(self.dimension)
        self.register_parameter('b',nn.Parameter(torch.zeros(len(self.data.list_of_ent))))
        if self.dimension==300:
            self.fc = nn.Linear(16128,self.dimension)
        else: #assuming dim=500
            self.fc = nn.Linear(27968,self.dimension)

    def forward(self, pairs):#given a batch of <head,rel> pairs (as text), return their embeddings
        head_enc, head_len = self.ent_encoder.prepare_batch([t[0] for t in pairs])
        if self.dimension==300:
            head_emb = self.ent_encoder.forward(head_enc,head_len).view(-1,1,15,20)
        else: #assuming dim==500
            head_emb = self.ent_encoder.forward(head_enc,head_len).view(-1,1,20,25)
        rel_enc, rel_len = self.rel_encoder.prepare_batch([t[1] for t in pairs])
        if self.dimension==300:
            rel_emb = self.rel_encoder.forward(rel_enc,rel_len).view(-1,1,15,20)
        else:
            rel_emb = self.rel_encoder.forward(rel_enc,rel_len).view(-1,1,20,25)
        stacked_inputs = torch.cat([head_emb,rel_emb],2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

    def trip2text(self,trips):#turns a list of triples with IDs to <head, rel> pairs and <tail, rel^-1> pairs as text
        return [[self.data.id2ent[t[0]],self.data.id2rel[t[1]]] for t in trips]+[[self.data.id2ent[t[2]],"inverse of "+self.data.id2rel[t[1]]] for t in trips]

    def batch_loss(self, batch, only_batch_negative = False):#given a batch (triples as IDs), compute loss (both directions) if only_batch_negative flag is positive, only candidates in the batch will be used as negative examples
        if only_batch_negative:
            negative_tail = list(set([x[0] for x in batch]+[x[2] for x in batch]))
        else:
            negative_tail = self.data.list_of_ent
        all_tail_texts = [self.data.id2ent[ent] for ent in negative_tail]
        tail_enc, tail_len = self.ent_encoder.prepare_batch(all_tail_texts)
        tail_emb = self.ent_encoder.forward(tail_enc,tail_len)
        mapped_embs = self.forward(self.trip2text(batch))
        scores = torch.mm(mapped_embs, tail_emb.transpose(1,0))
        if only_batch_negative:
            scores += self.b.index_select(0,torch.LongTensor(negative_tail).to(scores.device)).expand_as(scores)
        else:
            scores += self.b.expand_as(scores)
        expected_tail_ids = torch.LongTensor(
                [negative_tail.index(triple[2]) for triple in batch]+
                [negative_tail.index(triple[0]) for triple in batch]).to(scores.device)
        return F.cross_entropy(scores,expected_tail_ids)

    def get_rank(self, triples, filtered, evalTail=True):#given a list of triples and filtered heads/tails (all as IDs), eval them but take advantage of memoization. For eval only.
        filtered_clusts = []
        for f_list in filtered:
            f_clusts = set(self.data.ent2cluster[x] for x in f_list)
            if not self.data.canonicalized:
                all_filtered = [x for x in self.data.list_of_ent if self.data.ent2cluster[x] in f_clusts]
            else:
                all_filtered = [x for x in f_clusts]
            filtered_clusts.append(sorted(all_filtered))
        correct_positions = []
        for trip in triples:
            cluster = self.data.ent2cluster[trip[2 if evalTail else 0]]
            if not self.data.canonicalized:
                pos_in_cluster = [self.data.list_of_ent.index(x) for x in self.data.list_of_ent if self.data.ent2cluster[x]==cluster]
                correct_positions.append(pos_in_cluster)
            else:
                correct_positions.append([cluster])
        ranks = []
        if evalTail:
            input_text = [[self.data.id2ent[t[0]],self.data.id2rel[t[1]]] for t in triples]
        else:
            input_text = [[self.data.id2ent[t[2]],"inverse of "+self.data.id2rel[t[1]]] for t in triples]
        mapped_embs = self.forward(input_text)

        #Let us get the scores of correct answers for each input
        correct_scores = []
        for i in range(len(mapped_embs)):
            all_correct_texts = [self.data.id2ent[ent] for ent in correct_positions[i]]
            tail_enc, tail_len = self.ent_encoder.prepare_batch(all_correct_texts)
            correct_emb_mat = self.ent_encoder.forward(tail_enc,tail_len)
            scores = torch.mm(mapped_embs[i].view(1,-1),correct_emb_mat.transpose(1,0))
            scores += self.b.index_select(0,torch.LongTensor(correct_positions[i]).to(scores.device))
            scores = scores.cpu()
            correct_scores.append(torch.max(scores))

        ranks = [1]*len(mapped_embs)
        eval_batch_size=1000
        for i in range(0,len(self.data.id2ent),eval_batch_size):#evaluate all candidates in batches of 10k
            ent_texts = [self.data.id2ent[ent] for ent in range(i,min(i+eval_batch_size,len(self.data.id2ent)))]
            tail_enc, tail_len = self.ent_encoder.prepare_batch(ent_texts)
            tail_emb_mat = self.ent_encoder.forward(tail_enc,tail_len)
            scores = torch.mm(mapped_embs,tail_emb_mat.transpose(1,0))
            scores += self.b.narrow(0,i,len(ent_texts)).expand_as(scores)
            scores = scores.detach().cpu()
            for j in range(len(ranks)):
                l,r = bisect_left(filtered_clusts[j],i),bisect_left(filtered_clusts[j],i+eval_batch_size)
                for target in filtered_clusts[j][l:r]:
                    scores[j][target-i]=-1e9
                ranks[j]+=torch.sum(scores[j]>correct_scores[j]).numpy()
        return ranks

class TuckER(nn.Module):

    def __init__(self, data, rel_encoder, ent_encoder, dimension=300, dropout=0.1):
        super(TuckER,self).__init__()
        self.data = data
        self.dimension = dimension
        self.dropout = dropout
        self.rel_encoder = rel_encoder
        self.ent_encoder = ent_encoder
        self.W = nn.Parameter(torch.zeros([dimension,dimension,dimension],
            dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.W)
        self.input_drop = nn.Dropout(self.dropout)
        self.hidden_drop1 = nn.Dropout(self.dropout)
        self.hidden_drop2 = nn.Dropout(self.dropout)
        self.bn0 = nn.BatchNorm1d(dimension)
        self.bn1 = nn.BatchNorm1d(dimension)

    def forward(self, pairs):#given a batch of <head,rel> pairs (as text), return their embeddings
        head_enc, head_len = self.ent_encoder.prepare_batch([t[0] for t in pairs])
        head_emb = self.ent_encoder.forward(head_enc,head_len).view(-1,self.dimension)
        rel_enc, rel_len = self.rel_encoder.prepare_batch([t[1] for t in pairs])
        rel_emb = self.rel_encoder.forward(rel_enc,rel_len).view(-1,self.dimension)

        x = self.bn0(head_emb)
        x = self.input_drop(x)
        x = x.view(-1, 1, head_emb.size(1))

        W_mat = torch.mm(rel_emb,self.W.view(rel_emb.size(1),-1))
        W_mat = W_mat.view(-1,head_emb.size(1),head_emb.size(1))
        W_mat = self.hidden_drop1(W_mat)

        x = torch.bmm(x,W_mat)
        x = x.view(-1,head_emb.size(1))
        x = self.bn1(x)
        x = self.hidden_drop2(x)
        return x

    def trip2text(self,trips):#turns a list of triples with IDs to <head, rel> pairs and <tail, rel^-1> pairs as text
        return [[self.data.id2ent[t[0]],self.data.id2rel[t[1]]] for t in trips]+[[self.data.id2ent[t[2]],"inverse of "+self.data.id2rel[t[1]]] for t in trips]

    def batch_loss(self, batch, only_batch_negative = False):#given a batch (triples as IDs), compute loss (both directions) if only_batch_negative flag is positive, only candidates in the batch will be used as negative examples
        if only_batch_negative:
            negative_tail = list(set([x[0] for x in batch]+[x[2] for x in batch]))
        else:
            negative_tail = self.data.list_of_ent
        all_tail_texts = [self.data.id2ent[ent] for ent in negative_tail]
        tail_enc, tail_len = self.ent_encoder.prepare_batch(all_tail_texts)
        tail_emb = self.ent_encoder.forward(tail_enc,tail_len)
        mapped_embs = self.forward(self.trip2text(batch))
        scores = torch.mm(mapped_embs, tail_emb.transpose(1,0))
        expected_tail_ids = torch.LongTensor(
                [negative_tail.index(triple[2]) for triple in batch]+
                [negative_tail.index(triple[0]) for triple in batch]).to(scores.device)
        return F.cross_entropy(scores,expected_tail_ids)

    def get_rank(self, triples, filtered, evalTail=True):#given a list of triples and filtered heads/tails (all as IDs), eval them but take advantage of memoization. For eval only.
        filtered_clusts = []
        for f_list in filtered:
            f_clusts = set(self.data.ent2cluster[x] for x in f_list)
            if not self.data.canonicalized:
                all_filtered = [x for x in self.data.list_of_ent if self.data.ent2cluster[x] in f_clusts]
            else:
                all_filtered = [x for x in f_clusts]
            filtered_clusts.append(sorted(all_filtered))
        correct_positions = []
        for trip in triples:
            cluster = self.data.ent2cluster[trip[2 if evalTail else 0]]
            if not self.data.canonicalized:
                pos_in_cluster = [self.data.list_of_ent.index(x) for x in self.data.list_of_ent if self.data.ent2cluster[x]==cluster]
                correct_positions.append(pos_in_cluster)
            else:
                correct_positions.append([cluster])
        ranks = []
        if evalTail:
            input_text = [[self.data.id2ent[t[0]],self.data.id2rel[t[1]]] for t in triples]
        else:
            input_text = [[self.data.id2ent[t[2]],"inverse of "+self.data.id2rel[t[1]]] for t in triples]
        mapped_embs = self.forward(input_text)

        #Let us get the scores of correct answers for each input
        correct_scores = []
        for i in range(len(mapped_embs)):
            all_correct_texts = [self.data.id2ent[ent] for ent in correct_positions[i]]
            tail_enc, tail_len = self.ent_encoder.prepare_batch(all_correct_texts)
            correct_emb_mat = self.ent_encoder.forward(tail_enc,tail_len)
            scores = torch.mm(mapped_embs[i].view(1,-1),correct_emb_mat.transpose(1,0))
            scores = scores.cpu()
            correct_scores.append(torch.max(scores))

        ranks = [1]*len(mapped_embs)
        eval_batch_size=1000
        for i in range(0,len(self.data.id2ent),eval_batch_size):#evaluate all candidates in batches of 1k
            ent_texts = [self.data.id2ent[ent] for ent in range(i,min(i+eval_batch_size,len(self.data.id2ent)))]
            tail_enc, tail_len = self.ent_encoder.prepare_batch(ent_texts)
            tail_emb_mat = self.ent_encoder.forward(tail_enc,tail_len)
            scores = torch.mm(mapped_embs,tail_emb_mat.transpose(1,0)).detach().cpu()
            for j in range(len(ranks)):
                l,r = bisect_left(filtered_clusts[j],i),bisect_left(filtered_clusts[j],i+eval_batch_size)
                for target in filtered_clusts[j][l:r]:
                    scores[j][target-i]=-1e9
                ranks[j]+=torch.sum(scores[j]>correct_scores[j]).numpy()
        return ranks

class FiveStarE(nn.Module):

    def __init__(self, data, rel_encoder, ent_encoder, dimension=200, regularization = 0.1):
        super(FiveStarE,self).__init__()
        self.data = data
        self.dimension = dimension
        self.rel_encoder = rel_encoder
        self.ent_encoder = ent_encoder
        self.regularization = regularization

    def forward(self, pairs):#given a batch of <head,rel> pairs (as text), return their embeddings
        head_enc, head_len = self.ent_encoder.prepare_batch([t[0] for t in pairs])
        head_emb = self.ent_encoder.forward(head_enc,head_len).view(-1,self.dimension*2)
        rel_enc, rel_len = self.rel_encoder.prepare_batch([t[1] for t in pairs])
        rel_emb = self.rel_encoder.forward(rel_enc,rel_len).view(-1,self.dimension*8)

        head_re, head_im = head_emb[:,:self.dimension], head_emb[:,self.dimension:]
        rel_re_a,rel_im_a,rel_re_b,rel_im_b,rel_re_c,rel_im_c,rel_re_d,rel_im_d = rel_emb[:,:self.dimension],rel_emb[:,self.dimension:2*self.dimension],rel_emb[:,2*self.dimension:3*self.dimension],rel_emb[:,3*self.dimension:4*self.dimension],rel_emb[:,4*self.dimension:5*self.dimension],rel_emb[:,5*self.dimension:6*self.dimension],rel_emb[:,6*self.dimension:7*self.dimension],rel_emb[:,7*self.dimension:]
        score_re_a = head_re * rel_re_a - head_im * rel_im_a
        score_im_a = head_re * rel_im_a + head_im * rel_re_a

        #ah+b
        score_re_top = score_re_a + rel_re_b
        score_im_top = score_im_a + rel_im_b

        #ch
        score_re_c = head_re * rel_re_c - head_im * rel_im_c
        score_im_c = head_re * rel_im_c + head_im * rel_re_c

        #ch+d
        score_re_dn = score_re_c + rel_re_d
        score_im_dn = score_im_c + rel_im_d

        #(ah+b)Conj(ch+d)
        dn_re = torch.sqrt(score_re_dn * score_re_dn+score_im_dn*score_im_dn)
        up_re = torch.div(score_re_top * score_re_dn + score_im_top * score_im_dn, dn_re)
        up_im = torch.div(score_re_top * score_im_dn - score_im_top * score_re_dn, dn_re)
        #For regularization, head embeddings are multiplied by 2 because the same embeddings also appear as tails. 
        reg_weight = self.regularization*torch.sum(2*(head_re**2+head_im**2)**1.5 + (rel_re_a**2+rel_im_a**2+rel_re_b**2+rel_im_b**2+rel_re_c**2+rel_im_c**2+rel_re_d**2+rel_im_d**2)**1.5)/len(pairs)
        return (up_re,up_im,reg_weight)

    def trip2text(self,trips):#turns a list of triples with IDs to <head, rel> pairs and <tail, rel^-1> pairs as text
        return [[self.data.id2ent[t[0]],self.data.id2rel[t[1]]] for t in trips]+[[self.data.id2ent[t[2]],"inverse of "+self.data.id2rel[t[1]]] for t in trips]

    def batch_loss(self, batch, only_batch_negative = False):#given a batch (triples as IDs), compute loss (both directions) if only_batch_negative flag is positive, only candidates in the batch will be used as negative examples
        if only_batch_negative:
            negative_tail = list(set([x[0] for x in batch]+[x[2] for x in batch]))
        else:
            negative_tail = self.data.list_of_ent
        all_tail_texts = [self.data.id2ent[ent] for ent in negative_tail]
        tail_enc, tail_len = self.ent_encoder.prepare_batch(all_tail_texts)
        tail_emb = self.ent_encoder.forward(tail_enc,tail_len)
        tail_re, tail_im = tail_emb[:,:self.dimension], tail_emb[:,self.dimension:]
        up_re, up_im, reg_weight = self.forward(self.trip2text(batch))
        scores = up_re @ tail_re.transpose(0,1) + up_im @ tail_im.transpose(0,1)
        expected_tail_ids = torch.LongTensor(
                [negative_tail.index(triple[2]) for triple in batch]+
                [negative_tail.index(triple[0]) for triple in batch]).to(scores.device)
        return F.cross_entropy(scores,expected_tail_ids)+reg_weight

    def get_rank(self, triples, filtered, evalTail=True):#given a list of triples and filtered heads/tails (all as IDs), eval them but take advantage of memoization. For eval only.
        filtered_clusts = []
        for f_list in filtered:
            f_clusts = set(self.data.ent2cluster[x] for x in f_list)
            if not self.data.canonicalized:
                all_filtered = [x for x in self.data.list_of_ent if self.data.ent2cluster[x] in f_clusts]
            else:
                all_filtered = [x for x in f_clusts]
            filtered_clusts.append(sorted(all_filtered))
        correct_positions = []
        for trip in triples:
            cluster = self.data.ent2cluster[trip[2 if evalTail else 0]]
            if not self.data.canonicalized:
                pos_in_cluster = [self.data.list_of_ent.index(x) for x in self.data.list_of_ent if self.data.ent2cluster[x]==cluster]
                correct_positions.append(pos_in_cluster)
            else:
                correct_positions.append([cluster])
        ranks = []
        if evalTail:
            input_text = [[self.data.id2ent[t[0]],self.data.id2rel[t[1]]] for t in triples]
        else:
            input_text = [[self.data.id2ent[t[2]],"inverse of "+self.data.id2rel[t[1]]] for t in triples]
        up_re,up_im,_ = self.forward(input_text)

        #Let us get the scores of correct answers for each input
        correct_scores = []
        for i in range(len(up_re)):
            all_correct_texts = [self.data.id2ent[ent] for ent in correct_positions[i]]
            tail_enc, tail_len = self.ent_encoder.prepare_batch(all_correct_texts)
            correct_emb_mat = self.ent_encoder.forward(tail_enc,tail_len)

            correct_re, correct_im = correct_emb_mat[:,:self.dimension], correct_emb_mat[:,self.dimension:]
            scores = up_re[i].view(1,-1) @ correct_re.transpose(0,1) + up_im[i].view(1,-1) @ correct_im.transpose(0,1)
            scores = scores.detach().cpu()
            correct_scores.append(torch.max(scores))

        ranks = [1]*len(up_re)
        eval_batch_size=1000
        for i in range(0,len(self.data.id2ent),eval_batch_size):#evaluate all candidates in batches of 10k
            ent_texts = [self.data.id2ent[ent] for ent in range(i,min(i+eval_batch_size,len(self.data.id2ent)))]
            tail_enc, tail_len = self.ent_encoder.prepare_batch(ent_texts)
            tail_emb_mat = self.ent_encoder.forward(tail_enc,tail_len)

            tail_re, tail_im = tail_emb_mat[:,:self.dimension], tail_emb_mat[:,self.dimension:]
            scores = (up_re @ tail_re.transpose(0,1) + up_im @ tail_im.transpose(0,1)).detach().cpu()
            for j in range(len(ranks)):
                l,r = bisect_left(filtered_clusts[j],i),bisect_left(filtered_clusts[j],i+eval_batch_size)
                for target in filtered_clusts[j][l:r]:
                    scores[j][target-i]=-1e9
                ranks[j]+=torch.sum(scores[j]>correct_scores[j]).numpy()
        return ranks

class BoxE(nn.Module):

    def __init__(self, data, rel_encoder, ent_encoder, dimension=300, neg_examples = 100, margin=9.0):
        super(BoxE,self).__init__()
        self.data = data
        self.dimension = dimension
        #it is assumed that entity encoder gives outputs of size 2*dimension, e,b concatenated
        #it is assumed that relation encoder gives outputs of size 4*dimension c1,s1,c2,s2 concatenated
        self.rel_encoder = rel_encoder 
        self.ent_encoder = ent_encoder
        self.neg_examples = neg_examples
        self.margin = margin

    def forward(self, head, rel, tail, evalTail = True):#given a batch of embedded <head,rel,tail> return their scores
        if evalTail:
            e = torch.narrow(tail,1,0,self.dimension)+torch.narrow(head,1,self.dimension,self.dimension)
        else:
            e = torch.narrow(head,1,0,self.dimension)+torch.narrow(tail,1,self.dimension,self.dimension)
        if evalTail:
            c = torch.narrow(rel,1,2*self.dimension,self.dimension)
            s = torch.narrow(rel,1,3*self.dimension,self.dimension)
        else:
            c = torch.narrow(rel,1,0,self.dimension)
            s = torch.narrow(rel,1,self.dimension,self.dimension)
        l_placeholder = c-s/2
        u_placeholder = c+s/2
        l = torch.min(l_placeholder,u_placeholder)
        u = torch.max(l_placeholder,u_placeholder)
        w = u-l+1
        kappa = 0.5*(w-1)*(w-1/w)
        in_box = torch.logical_and(l<=e,e<=u)
        return torch.norm(in_box * torch.abs(e-c)/w + torch.logical_not(in_box)*(torch.abs(e-c)*w-kappa),dim=1)

    def trip2text(self,trips):#turns a list of triples with IDs to <head, rel, tail> pairs as text
        return [[self.data.id2ent[t[0]],self.data.id2rel[t[1]],self.data.id2ent[t[2]]] for t in trips]

    def batch_loss(self, batch, only_batch_negative = True):#given a batch (triples as IDs), compute loss (both directions)
        all_triples = self.trip2text(batch)
        head_enc, head_len = self.ent_encoder.prepare_batch([t[0] for t in all_triples])
        head_emb = self.ent_encoder.forward(head_enc,head_len).view(-1,self.dimension*2)
        rel_enc, rel_len = self.rel_encoder.prepare_batch([t[1] for t in all_triples])
        rel_emb = self.rel_encoder.forward(rel_enc,rel_len).view(-1,self.dimension*4)
        tail_enc, tail_len = self.ent_encoder.prepare_batch([t[2] for t in all_triples])
        tail_emb = self.ent_encoder.forward(tail_enc,tail_len).view(-1,self.dimension*2)
        batch_size = len(batch)

        if only_batch_negative:
            neg_emb = torch.cat([head_emb,tail_emb])
        else:
            all_neg_texts = [self.data.id2ent[ent] for ent in self.data.list_of_ent]
            neg_enc, neg_len = self.ent_encoder.prepare_batch(all_neg_texts)
            neg_emb = self.ent_encoder.forward(neg_enc,neg_len)
        #eval tails
        positive_scores = self.forward(head_emb,rel_emb,tail_emb,evalTail=True)
        negative_heads = head_emb.repeat(self.neg_examples,1)
        negative_rels = rel_emb.repeat(self.neg_examples,1)
        negative_tails = neg_emb.index_select(0,torch.LongTensor(np.random.randint(len(neg_emb),size=self.neg_examples*batch_size)))
        negative_scores = self.forward(negative_heads,negative_rels,negative_tails,evalTail=True)
        loss = -torch.mean(F.logsigmoid(self.margin-positive_scores))-torch.mean(F.logsigmoid(negative_scores-self.margin))

        #eval_heads
        positive_scores = self.forward(head_emb,rel_emb,tail_emb,evalTail=False)
        negative_heads = neg_emb.index_select(0,torch.LongTensor(np.random.randint(len(neg_emb),size=self.neg_examples*batch_size)))
        negative_tails = tail_emb.repeat(self.neg_examples,1)
        negative_scores = self.forward(negative_heads,negative_rels,negative_tails,evalTail=False)
        loss += -torch.mean(F.logsigmoid(self.margin-positive_scores))-torch.mean(F.logsigmoid(negative_scores-self.margin))
        return loss/2 #divided by 2 to get the average of heads and tails

    def get_rank(self, triples, filtered, evalTail=True):#given a list of triples and filtered heads/tails (all as IDs), eval them. For eval only.
        filtered_clusts = []
        for f_list in filtered:
            f_clusts = set(self.data.ent2cluster[x] for x in f_list)
            if not self.data.canonicalized:
                all_filtered = [x for x in self.data.list_of_ent if self.data.ent2cluster[x] in f_clusts]
            else:
                all_filtered = [x for x in f_clusts]
            filtered_clusts.append(sorted(all_filtered))
        correct_positions = []
        for trip in triples:
            cluster = self.data.ent2cluster[trip[2 if evalTail else 0]]
            if not self.data.canonicalized:
                pos_in_cluster = [self.data.list_of_ent.index(x) for x in self.data.list_of_ent if self.data.ent2cluster[x]==cluster]
                correct_positions.append(pos_in_cluster)
            else:
                correct_positions.append([cluster])
        ranks = []
        input_text=self.trip2text(triples)
        if evalTail:
            input_text = [[self.data.id2ent[t[0]],self.data.id2rel[t[1]]] for t in triples]
        else:
            input_text = [[self.data.id2ent[t[2]],self.data.id2rel[t[1]]] for t in triples] 

        head_enc, head_len = self.ent_encoder.prepare_batch([t[0] for t in input_text])
        head_emb = self.ent_encoder.forward(head_enc,head_len).view(-1,self.dimension*2)
        rel_enc, rel_len = self.rel_encoder.prepare_batch([t[1] for t in input_text])
        rel_emb = self.rel_encoder.forward(rel_enc,rel_len).view(-1,self.dimension*4)

        #Let us get the scores of correct answers for each input
        correct_scores = []
        for i in range(len(input_text)):
            all_correct_texts = [self.data.id2ent[ent] for ent in correct_positions[i]]
            tail_enc, tail_len = self.ent_encoder.prepare_batch(all_correct_texts)
            tail_emb = self.ent_encoder.forward(tail_enc,tail_len)
            if evalTail:
                scores = self.forward(torch.narrow(head_emb,0,i,1).expand(len(all_correct_texts),self.dimension*2),torch.narrow(rel_emb,0,i,1).expand(len(all_correct_texts),self.dimension*4),tail_emb,evalTail=True)
            else:
                scores = self.forward(tail_emb,torch.narrow(rel_emb,0,i,1).expand(len(all_correct_texts),self.dimension*4),torch.narrow(head_emb,0,i,1).expand(len(all_correct_texts),self.dimension*2),evalTail=False)
            scores = scores.cpu()
            correct_scores.append(torch.min(scores))

        ranks = [1]*len(input_text)
        eval_batch_size=10
        for i in tqdm.trange(0,len(self.data.id2ent),eval_batch_size,desc="computing rank"):#evaluate all candidates in batches of 10
            ent_texts = [self.data.id2ent[ent] for ent in range(i,min(i+eval_batch_size,len(self.data.id2ent)))]
            tail_enc, tail_len = self.ent_encoder.prepare_batch(ent_texts)
            tail_emb = self.ent_encoder.forward(tail_enc,tail_len)
            batch_size = len(ent_texts)

            all_heads = torch.cat([torch.narrow(head_emb,0,j,1).expand(batch_size,self.dimension*2) for j in range(len(ranks))])
            all_rels = torch.cat([torch.narrow(rel_emb,0,j,1).expand(batch_size,self.dimension*4) for j in range(len(ranks))])
            all_tails = tail_emb.repeat(len(ranks),1)
            if evalTail:
                scores= self.forward(all_heads,all_rels,all_tails, evalTail=True)
            else:
                scores= self.forward(all_tails,all_rels,all_heads, evalTail=False)
            scores = torch.reshape(scores,(len(ranks),batch_size)).detach().cpu()
            for j in range(len(ranks)):
                l,r = bisect_left(filtered_clusts[j],i),bisect_left(filtered_clusts[j],i+eval_batch_size)
                for target in filtered_clusts[j][l:r]:
                    scores[j][target-i]=1e9
                ranks[j]+=torch.sum(scores[j]<correct_scores[j]).numpy()
        return ranks
