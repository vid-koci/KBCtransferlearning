import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
import numpy as np
import spacy
import json

Glove_path = "glove/glove.6B.300d.txt"


class GRUEncoder(nn.Module):

    def __init__(self,
                 word2id,
                 tokenizer,
                 hidden_size=100,
                 pretrained_vocab=None,
                 pretrained_emb=None):
        super(GRUEncoder, self).__init__()
        self.word2id = word2id
        self.tokenizer = tokenizer
        self.init_embeddings(pretrained_vocab, pretrained_emb)
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_embeddings=self.embed_matrix.shape[0],
                                  embedding_dim=self.embed_matrix.shape[1],
                                  padding_idx=0)
        self.encoder = nn.GRU(self.embed_matrix.shape[1],
                              self.hidden_size,
                              batch_first=True)
        self.embed.weight.data.copy_(torch.from_numpy(self.embed_matrix))
        self.tokenization_memo = {}

    def init_embeddings(self, pretrained_vocab, pretrained_emb):
        self.word_embed = {}
        if pretrained_vocab is None or pretrained_emb is None:  #initialize from Glove if not pretrained
            with open(Glove_path, encoding="utf8") as glove:
                for line in glove:
                    word, vec = line.split(' ', 1)
                    if word in self.word2id:
                        self.word_embed[self.word2id[word]] = np.fromstring(
                            vec, sep=' ')
        else:
            for word, w_id in pretrained_vocab.items():
                if word in self.word2id:
                    self.word_embed[self.word2id[word]] = pretrained_emb[w_id]
        #initialize unknown word according to normal distribution
        uninitialized = [
            words for words in self.word2id.values()
            if not words in self.word_embed
        ]
        for word in uninitialized:
            self.word_embed[word] = np.random.normal(size=300)

        self.embed_matrix = np.zeros((len(self.word_embed), 300))
        for word in self.word_embed:
            self.embed_matrix[word] = self.word_embed[word]

    def forward(self, batch, doc_len):
        size, sort = torch.sort(doc_len, dim=0, descending=True)
        _, unsort = torch.sort(sort, dim=0)
        batch = torch.index_select(batch, dim=0, index=sort)
        embedded = self.embed(batch)
        packed = pack(embedded, size.data.tolist(), batch_first=True)
        encoded, h = self.encoder(packed)
        return torch.index_select(h, dim=1, index=unsort)[0]

    def prepare_batch(
        self, texts
    ):  #gets a list of untokenized texts, returns a padded batch that can be used as an input to forward, puts it on the same device as the GRU
        batch_size = len(texts)
        tokenized_texts = []
        for text in texts:
            if text in self.tokenization_memo:
                tokenized_texts.append(self.tokenization_memo[text])
            else:
                tokenized = [
                    self.word2id[token.text.lower()] for token in
                    self.tokenizer(text, disable=['parser', 'tagger', 'ner'])
                    if token.text.lower() in self.word2id
                ]
                self.tokenization_memo[text] = tokenized
                tokenized_texts.append(tokenized)
        padded_length = max(len(text) for text in tokenized_texts)
        phrase_batch = np.zeros((batch_size, padded_length),
                                dtype=int)  #<PAD> has id 0, that's why zeros
        for i, tokens in enumerate(tokenized_texts):
            phrase_batch[i, 0:len(tokens)] = np.array(tokens)
        device = self.encoder.weight_ih_l0.device
        phrase_batch = torch.from_numpy(phrase_batch).to(device)
        phrase_len = torch.LongTensor([len(text)
                                       for text in tokenized_texts]).to(device)
        return phrase_batch, phrase_len


class NoEncoder(nn.Module):

    def __init__(self,
                 name2id,
                 hidden_size=100,
                 pretrained_encoder=None,
                 add_inverse=False,
                 pretrained_json=None):
        super(NoEncoder, self).__init__()
        self.name2id = {}
        if add_inverse:
            for k, v in name2id.items():
                self.name2id[k] = v
                self.name2id["inverse of " + k] = v + len(name2id)
        else:
            self.name2id = name2id
        self.hidden_size = hidden_size
        if not pretrained_encoder is None:
            pretrained_encoder.eval()
            self.init_embeddings(pretrained_encoder)
            self.embed = nn.Embedding(
                num_embeddings=self.embed_matrix.shape[0],
                embedding_dim=self.embed_matrix.shape[1])
            self.embed.weight.data.copy_(torch.from_numpy(self.embed_matrix))
        elif pretrained_json is not None:
            self.init_embeddings_from_json(
                json.load(open(pretrained_json, 'r')))
            self.embed = nn.Embedding(
                num_embeddings=self.embed_matrix.shape[0],
                embedding_dim=self.embed_matrix.shape[1])
            self.embed.weight.data.copy_(torch.from_numpy(self.embed_matrix))
        else:
            self.embed = nn.Embedding(num_embeddings=len(self.name2id),
                                      embedding_dim=self.hidden_size)

    def init_embeddings_from_json(self, pretrained_json):
        self.embed_matrix = np.zeros((len(self.name2id), self.hidden_size))
        for ent_name, ent_id in self.name2id.items():
            if ent_name in pretrained_json:
                self.embed_matrix[ent_id] = np.array(pretrained_json[ent_name])
            else:
                self.embed_matrix[ent_id] = np.random.normal(
                    size=self.hidden_size)

    def init_embeddings(self, pretrained_model):
        self.embed_matrix = np.zeros((len(self.name2id), self.hidden_size))
        all_keys = [k for k in self.name2id.keys()]
        for i in range(0, len(all_keys), 1000):
            encoded_batch, lens = pretrained_model.prepare_batch(
                all_keys[i:i + 1000])
            nonzero_indices = torch.nonzero(lens, as_tuple=False).squeeze(
            )  #since unknown words are removed, we can get length 0 inputs. Remove to avoid crash, replace with random initializations later
            new_encoded_batch = torch.index_select(encoded_batch, 0,
                                                   nonzero_indices)
            new_lens = torch.index_select(lens, 0, nonzero_indices)
            if len(new_lens) != 0:
                pretrained_embeddings = pretrained_model.forward(
                    new_encoded_batch, new_lens)
                pretrained_embeddings = pretrained_embeddings.detach().cpu(
                ).numpy()
            nonzero_indices = nonzero_indices.cpu().numpy()
            cnt = 0
            for j in range(len(lens)):
                if j in nonzero_indices:
                    self.embed_matrix[self.name2id[all_keys[
                        i + j]]] = pretrained_embeddings[cnt]
                    cnt += 1
                else:
                    self.embed_matrix[self.name2id[all_keys[
                        i + j]]] = np.random.normal(size=self.hidden_size)

    def forward(self, batch, doc_len):
        return self.embed(batch)

    def prepare_batch(
        self, texts
    ):  #gets a list of untokenized texts, returns a padded batch that can be used as an input to forward, puts it on the same device as the GRU
        device = self.embed.weight.device
        return torch.from_numpy(np.array([self.name2id[t] for t in texts])).to(
            device
        ), None  #Extra None is returned so that the same code works for "regular" encoder and this


#left for future testing
#if __name__=="__main__":
#    import data_reader
#    data = data_reader.Dataset("Data/ReVerb20K")
#    a = ALBERTEncoder()
#    test_list_of_texts = [data.id2ent[i] for i in range(1,11)]
#    print(test_list_of_texts)
#    b,attent=a.prepare_batch(test_list_of_texts)
#    print(b,attent)
#    output = a.forward(b,attent)
#    print(output)
#    #to be tested
