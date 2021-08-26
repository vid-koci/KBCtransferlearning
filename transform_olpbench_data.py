from tqdm import tqdm
#read entity ids
outfile = open("Data/OlpBench/ent2id.txt",'w')
with open("Data/olpbench/mapped_to_ids/entity_id_map.txt") as f:
    lines = list(f)[1:]
    outfile.write(str(len(lines))+"\n")
    for l in tqdm(lines, desc="copying entity list"):
        name, ent_id, *sth = l.split("\t")
        outfile.write(name+"\t"+ent_id+"\n")
outfile.close()

outfile = open("Data/OlpBench/rel2id.txt",'w')
with open("Data/olpbench/mapped_to_ids/relation_id_map.txt") as f:
    lines = list(f)[1:]
    outfile.write(str(len(lines))+"\n")
    for l in tqdm(lines,desc="copying relation list"):
        name, rel_id, *sth = l.split("\t")
        outfile.write(name+"\t"+rel_id+"\n")
outfile.close()

outfile = open("Data/OlpBench/train_trip.txt",'w')
with open("Data/olpbench/mapped_to_ids/train_data_thorough.txt") as f:
    lines = list(f)
    outfile.write(str(len(lines))+"\n")
    for l in tqdm(lines,desc="copying train data"):
        e1, r, e2, *sth = l.split("\t")
        outfile.write(e1+"\t"+r+"\t"+e2+"\n")
outfile.close()

outfile = open("Data/OlpBench/valid_trip.txt",'w')
with open("Data/olpbench/mapped_to_ids/validation_data_linked.txt") as f:
    lines = list(f)
    outfile.write(str(len(lines))+"\n")
    for l in tqdm(lines,desc="copying validation data"):
        e1, r, e2, *sth = l.split("\t")
        outfile.write(e1+"\t"+r+"\t"+e2+"\n")
outfile.close()

outfile = open("Data/OlpBench/test_trip.txt",'w')
with open("Data/olpbench/mapped_to_ids/test_data.txt") as f:
    lines = list(f)
    outfile.write(str(len(lines))+"\n")
    for l in tqdm(lines,desc="copying test data"):
        e1, r, e2, *sth = l.split("\t")
        outfile.write(e1+"\t"+r+"\t"+e2+"\n")
outfile.close()
