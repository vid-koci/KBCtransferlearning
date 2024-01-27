import os
import spacy


class Dataset():

    def __init__(self,
                 path,
                 canonicalized=False,
                 add_inverse=False,
                 diagnostic_test=False):
        self.ent2cluster = None
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.ent2id, self.id2ent, self.word2id_ent = self.read_term_ids(
            os.path.join(path, "ent2id.txt"))
        self.rel2id, self.id2rel, self.word2id_rel = self.read_term_ids(
            os.path.join(path, "rel2id.txt"), add_inverse=add_inverse)
        self.list_of_ent = [key for key in self.id2ent.keys()
                            ]  #needed for neg examples
        self.train_data = self.read_trips(os.path.join(path, "train_trip.txt"))
        self.valid_data = self.read_trips(os.path.join(path, "valid_trip.txt"))
        if diagnostic_test:
            self.test_data = self.read_trips_diagnostic(
                os.path.join(path, "test_trip.txt"))
        else:
            self.test_data = self.read_trips(
                os.path.join(path, "test_trip.txt"))
        self.canonicalized = canonicalized
        if not canonicalized:
            self.ent2cluster = self.read_clusters(
                os.path.join(path, "gold_npclust.txt"))
        else:
            self.ent2cluster = {i: i
                                for i in self.list_of_ent
                                }  #each entity its own cluster
        #rename all entities and relations to 0 - [n-1]
        rename_ent_ids = {}
        for i in range(len(self.list_of_ent)):
            rename_ent_ids[self.list_of_ent[i]] = i
        self.id2ent = {}
        for ent in self.ent2id.keys():
            self.id2ent[rename_ent_ids[self.ent2id[ent]]] = ent
            self.ent2id[ent] = rename_ent_ids[self.ent2id[ent]]
        rename_rel_ids = {}
        for i, rel_id in enumerate(self.id2rel.keys()):
            rename_rel_ids[rel_id] = i
        self.id2rel = {}
        for rel in self.rel2id.keys():
            self.id2rel[rename_rel_ids[self.rel2id[rel]]] = rel
            self.rel2id[rel] = rename_rel_ids[self.rel2id[rel]]
        self.list_of_ent = [i for i in range(len(self.list_of_ent))]
        self.train_data = [(rename_ent_ids[t[0]], rename_rel_ids[t[1]],
                            rename_ent_ids[t[2]]) for t in self.train_data]
        self.valid_data = [(rename_ent_ids[t[0]], rename_rel_ids[t[1]],
                            rename_ent_ids[t[2]]) for t in self.valid_data]
        if diagnostic_test:
            temp_data = []
            for t in self.test_data:
                if "alternative_tails" in t:
                    triple_ = t["triple"]
                    temp_data.append({
                        "triple": (rename_ent_ids[triple_[0]], rename_rel_ids[triple_[1]], rename_ent_ids[triple_[2]]),
                        "alternative_tails":
                        tuple(rename_ent_ids[x]
                              for x in t["alternative_tails"])
                    })
                elif "alternative_heads" in t:
                    triple_ = t["triple"]
                    temp_data.append({
                        "triple": (rename_ent_ids[triple_[0]], rename_rel_ids[triple_[1]], rename_ent_ids[triple_[2]]),
                        "alternative_heads":
                        tuple(rename_ent_ids[x]
                              for x in t["alternative_heads"])
                    })
            self.test_data = temp_data
        else:
            self.test_data = [(rename_ent_ids[t[0]], rename_rel_ids[t[1]],
                               rename_ent_ids[t[2]]) for t in self.test_data]
        tmp_clust = {}
        for clust_id, clust in self.ent2cluster.items():
            tmp_clust[rename_ent_ids[clust_id]] = rename_ent_ids[clust]
        self.ent2cluster = tmp_clust

    def read_term_ids(self, path, add_inverse=False):
        term2id = {}
        id2term = {}
        word2id = {"<PAD>": 0}
        if add_inverse:
            word2id["inverse"] = 1
            word2id["of"] = 2
        for line in list(open(path, 'r', encoding="utf-8"))[1:]:
            term, term_id = line.strip().split("\t")
            term_id = int(term_id)
            while term in term2id:  #fb15k237 will have certain entities appear under same name, we add a space so that it's treated as a different word by vocab, but isn't detected by the tokenizer
                term += " "
            term2id[term] = term_id
            id2term[term_id] = term
            tokens = self.spacy_nlp(
                term, disable=['parser', 'tagger', 'ner', 'lemmatizer'])
            for token in tokens:
                if not token.text.lower() in word2id:
                    new_id = len(word2id)
                    word2id[token.text.lower()] = new_id
        return term2id, id2term, word2id

    def read_trips(self, path):
        return [
            tuple(int(x) for x in line.strip().split())
            for line in list(open(path, 'r'))[1:]
        ]

    def read_trips_diagnostic(self, path):
        triples = []
        for line in list(open(path, 'r'))[1:]:
            if line.strip().split()[3] == "tail":  #else it's head
                triples.append({
                    "triple":
                    tuple(int(x) for x in line.strip().split()[:3]),
                    "alternative_tails":
                    tuple(int(x) for x in line.strip().split()[4:])
                })
            else:
                triples.append({
                    "triple":
                    tuple(int(x) for x in line.strip().split()[:3]),
                    "alternative_heads":
                    tuple(int(x) for x in line.strip().split()[4:])
                })
        return triples

    def read_clusters(self, path):
        #Reads (gold) clusters, returns a map from an entity to a cluster id.
        #Cluster id is id of the smallest id in the cluster (we only need them for eval anyway, why complicate?)
        id2clust = {}
        for line in list(open(path, 'r')):
            ids = [int(x) for x in line.strip().split("\t")]
            id2clust[ids[0]] = min(ids[2:])
        return id2clust


if __name__ == '__main__':
    path_ = "./Data/ReVerb20k"
    data = Dataset("./Data/ReVerb20k",
                               add_inverse=True,
                               canonicalized=(path_.find("ReVerb") == -1),
                               diagnostic_test=True)

    triples = [d['triple'] for d in data.test_data]
    relations = [t[1] for t in triples]

    for i, rel in enumerate(relations):
        print(i, data.id2rel[rel])


#Left here for any future testing
#if __name__ == "__main__":
#    #let's test stuff
#    data = Dataset("Data/ReVerb20K")
#    print("entity")
#    for i in range(11):
#        print(i,data.id2ent[i],data.ent2id[data.id2ent[i]])
#    print("relation")
#    for i in range(11):
#        print(i,data.id2rel[i],data.rel2id[data.id2rel[i]])
#    print("train",data.train_data[:10])
#    print("cluster")
#    for i in [187,2515,5456,2,0]:
#        print(i,": ",data.ent2cluster[i])
#    print("word ids")
#    i=1
#    for word,w_id in data.word2id_ent.items():
#        print(word, ": ", w_id)
#        i+=1
#        if i>10:
#            break
#
