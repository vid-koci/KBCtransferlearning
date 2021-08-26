train_triples = [l.strip().split("\t") for l in list(open("train.txt"))]
valid_triples = [l.strip().split("\t") for l in list(open("valid.txt"))]
test_triples = [l.strip().split("\t") for l in list(open("test.txt"))]
ent2id = {}
rel2id = {}
ent_cnt = 0
rel_cnt = 0
for l in [train_triples,valid_triples,test_triples]:
    for trip in l:
        if not trip[0] in ent2id:
            ent2id[trip[0]]=ent_cnt
            ent_cnt+=1
        if not trip[1] in rel2id:
            rel2id[trip[1]]=rel_cnt
            rel_cnt+=1
        if not trip[2] in ent2id:
            ent2id[trip[2]]=ent_cnt
            ent_cnt+=1
with open("train_trip.txt",'w') as f:
    f.write(str(len(train_triples))+"\n")
    for trip in train_triples:
        f.write(f"{ent2id[trip[0]]}\t{rel2id[trip[1]]}\t{ent2id[trip[2]]}\n")
with open("valid_trip.txt",'w') as f:
    f.write(str(len(valid_triples))+"\n")
    for trip in valid_triples:
        f.write(f"{ent2id[trip[0]]}\t{rel2id[trip[1]]}\t{ent2id[trip[2]]}\n")
with open("test_trip.txt",'w') as f:
    f.write(str(len(test_triples))+"\n")
    for trip in test_triples:
        f.write(f"{ent2id[trip[0]]}\t{rel2id[trip[1]]}\t{ent2id[trip[2]]}\n")
with open("ent2id.txt",'w') as f:
    f.write(str(len(ent2id))+"\n")
    for name,ent_id in ent2id.items():
        f.write(f"{name}\t{ent_id}\n")
with open("rel2id.txt",'w') as f:
    f.write(str(len(rel2id))+"\n")
    for name,rel_id in rel2id.items():
        f.write(f"{name}\t{rel_id}\n")
