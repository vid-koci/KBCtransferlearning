import torch
import numpy as np
import tqdm

import argparse
import os
import logging
import json

import random
import data_reader
import encoder
import embedding
import scoring

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Name of this project")

    parser.add_argument('-data',default='Data/ReVerb45K', help='folder with the data')
    parser.add_argument('-embedding',default='TuckER', help='vector embeddings', choices=['ConvE','TuckER','BoxE','FiveStarE'])
    parser.add_argument('-encoder',default='GRU', help='encoder', choices=['GRU','NoEncoder'])
    parser.add_argument('-dim', default=300, type=int, help='dimension of the embeddings')
    parser.add_argument('-dropout', default=0.3, type=float, help='dropout for ConvE and TuckER')
    parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('-batch', default=100, type=int, help='size of the batches')
    parser.add_argument('-neg_examples', default=100, type=int, help='number of negative examples (BoxE)')
    parser.add_argument('-margin', default=9.0, type=float, help='margin for BoxE training loss')
    parser.add_argument('-regularization', default=0.1, type=float, help='weight of N3 regularization for FiveStarE')
    parser.add_argument('-n_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('-seed', default=42, type=int, help='random seed for reproducibility')
    parser.add_argument('-hits', default=[10,30,50], help='choice of n in Hits@n')
    parser.add_argument('-output_dir', default="results", help='folder with the best model and results')
    parser.add_argument('-pretrained_dir', default=None, help='folder with the pretrained model and its vocabulary')
    parser.add_argument('-no_train', action='store_true', help='if True, it will load the model from -output_dir instead of training it')
    parser.add_argument('-only_batch_negative', action='store_true', help='if True, it will only use entities within the batch as negative examples')
    parser.add_argument('-dump_vocab', action='store_true', help='if True, it will store vocabulary of encoders. Necessary for transfer learning.')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = data_reader.Dataset(args.data, add_inverse=True, canonicalized = (args.data.find("ReVerb")==-1))
    ent_dim = {"TuckER":args.dim, "ConvE": args.dim, "BoxE": 2*args.dim, "FiveStarE":2*args.dim}[args.embedding] 
    rel_dim = {"TuckER":args.dim, "ConvE": args.dim, "BoxE": 4*args.dim, "FiveStarE":8*args.dim}[args.embedding] 

    if not args.pretrained_dir is None: #if loading from pretrained, it gets a little bit messy
        pretrained_dict = torch.load(os.path.join(args.pretrained_dir,"best_model"))
        if args.encoder=='GRU':
            ent_enc = encoder.GRUEncoder(data.word2id_ent, data.spacy_nlp, hidden_size = ent_dim,
                    pretrained_vocab = json.load(open(os.path.join(args.pretrained_dir,"word2id_ent.json"),'r')),
                    pretrained_emb = pretrained_dict["ent_encoder.embed.weight"].cpu().numpy())
            rel_enc = encoder.GRUEncoder(data.word2id_rel, data.spacy_nlp, hidden_size = rel_dim,
                    pretrained_vocab = json.load(open(os.path.join(args.pretrained_dir,"word2id_rel.json"),'r')),
                    pretrained_emb = pretrained_dict["rel_encoder.embed.weight"].cpu().numpy())
        elif args.encoder=='NoEncoder':
            pretrained_vocab = json.load(open(os.path.join(args.pretrained_dir,"word2id_ent.json"),'r'))
            pretrained_enc = encoder.GRUEncoder(pretrained_vocab, data.spacy_nlp, hidden_size = ent_dim,
                    pretrained_vocab = pretrained_vocab,
                    pretrained_emb = pretrained_dict["ent_encoder.embed.weight"].cpu().numpy())
            encoder_state_dict = {k.replace("ent_encoder.",""):v for k,v in pretrained_dict.items() if k.find("ent_encoder.")!=-1} #load only weights from the entity encoder and rename them appropriately
            pretrained_enc.load_state_dict(encoder_state_dict)
            ent_enc = encoder.NoEncoder(data.ent2id,hidden_size = ent_dim, pretrained_encoder = pretrained_enc)
            #same for rel encoder
            pretrained_vocab = json.load(open(os.path.join(args.pretrained_dir,"word2id_rel.json"),'r'))
            pretrained_enc = encoder.GRUEncoder(pretrained_vocab, data.spacy_nlp,
                    hidden_size = rel_dim, pretrained_vocab = pretrained_vocab,
                    pretrained_emb = pretrained_dict["rel_encoder.embed.weight"].cpu().numpy())
            encoder_state_dict = {k.replace("rel_encoder.",""):v for k,v in pretrained_dict.items() if k.find("rel_encoder.")!=-1}
            pretrained_enc.load_state_dict(encoder_state_dict)
            rel_enc = encoder.NoEncoder(data.rel2id,hidden_size = rel_dim, pretrained_encoder = pretrained_enc, add_inverse=True)
    else:
        if args.encoder=='GRU':
            ent_enc = encoder.GRUEncoder(data.word2id_ent, data.spacy_nlp, hidden_size = ent_dim)
            rel_enc = encoder.GRUEncoder(data.word2id_rel, data.spacy_nlp, hidden_size = rel_dim)
        else:
            ent_enc = encoder.NoEncoder(data.ent2id,hidden_size = ent_dim)
            rel_enc = encoder.NoEncoder(data.rel2id,hidden_size = rel_dim, add_inverse=(args.embedding!="BoxE"))
    if args.embedding=='ConvE':
        embedding = embedding.ConvE(data,rel_enc,ent_enc,dimension=args.dim, dropout=args.dropout)
    elif args.embedding=='TuckER':
        embedding = embedding.TuckER(data,rel_enc,ent_enc,dimension=args.dim, dropout=args.dropout)
    elif args.embedding=='BoxE':
        embedding = embedding.BoxE(data,rel_enc,ent_enc,dimension=args.dim, neg_examples = args.neg_examples, margin = args.margin)
    elif args.embedding=='FiveStarE':
        embedding = embedding.FiveStarE(data,rel_enc,ent_enc,dimension=args.dim, regularization = args.regularization) 
    if not args.pretrained_dir is None:
        new_dict = {k:v for k,v in pretrained_dict.items() if not k in ['ent_encoder.embed.weight','rel_encoder.embed.weight','b']}#Load everything but the embeddings and biases in ConvE
        embedding.load_state_dict(new_dict, strict=False)

    if torch.cuda.is_available():
        embedding.cuda()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    if type(args.hits)==str:
        args.hits = json.loads(args.hits)
    if not args.no_train:
        optimizer = torch.optim.Adam(embedding.parameters(),lr=args.lr)
        try:
            best_MRR = float(list(open(os.path.join(args.output_dir,"best_MRR.txt"),'r'))[0])
        except:
            best_MRR = 0
        for epoch in range(args.n_epochs):
            embedding.train()
            random.shuffle(data.train_data)
            t = tqdm.trange(0,len(data.train_data),args.batch, desc=f"epoch {epoch+1} loss: 0.0")
            running_loss = 0.0
            for batch_start in t:
                batch = data.train_data[batch_start:batch_start+args.batch]
                loss=embedding.batch_loss(batch,only_batch_negative = args.only_batch_negative)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(embedding.parameters(),1.0)
                running_loss+=loss.cpu().detach().numpy()/args.batch
                optimizer.step()
                if (batch_start//args.batch + 1)%10==0:
                    t.set_description(f"epoch {epoch+1} loss:{running_loss/5:.4f}")
                    running_loss=0
            
            if (epoch+1)%20!=0:
                continue
            embedding.eval()
            MRR = scoring.evaluate(embedding,data,eval_set='valid',logger=logger, hits=args.hits)
            embedding.train()
            logger.info(f"epoch {epoch + 1}\t MRR {MRR:.5f}")
            try:#update best MRR in case of parallel tests
                updated_MRR = float(list(open(os.path.join(args.output_dir,"best_MRR.txt"),'r'))[0])
            except:
                updated_MRR = 0
            best_MRR = max(best_MRR,updated_MRR)
            if MRR>best_MRR:
                best_MRR = MRR
                torch.save(embedding.state_dict(),os.path.join(args.output_dir,"best_model"))
                if args.dump_vocab:
                    json.dump(data.word2id_ent, open(os.path.join(args.output_dir,"word2id_ent.json"),'w'))
                    json.dump(data.word2id_rel, open(os.path.join(args.output_dir,"word2id_rel.json"),'w'))
                json.dump(args.__dict__,open(os.path.join(args.output_dir,"best_args.json"),'w'))
                with open(os.path.join(args.output_dir,"best_MRR.txt"),'w') as MRR_report:
                    MRR_report.write(f"{best_MRR:.5f}")
        embedding.load_state_dict(torch.load(os.path.join(args.output_dir,"best_model")))
    embedding.eval()
    logger.info("validation set")
    _=scoring.evaluate(embedding,data,eval_set='valid',logger=logger, hits=args.hits)
    logger.info("test set")
    _=scoring.evaluate(embedding,data,eval_set='test',logger=logger, hits=args.hits)
