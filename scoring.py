import numpy as np
import tqdm
def evaluate(model, data, hits=[10,30,50], eval_set = 'valid', logger=None):
    h_rank = []
    h_hits = np.zeros(len(hits))
    t_rank = []
    t_hits = np.zeros(len(hits))
    HatN=[0]*len(hits)
    filtered_facts_tail = {}
    filtered_facts_head = {}
    for dataset in [data.train_data, data.valid_data, data.test_data]:
        for triple in dataset:
            if (data.ent2cluster[triple[0]],triple[1]) in filtered_facts_tail:
                filtered_facts_tail[(data.ent2cluster[triple[0]],triple[1])].append(data.ent2cluster[triple[2]])
            else:
                filtered_facts_tail[(data.ent2cluster[triple[0]],triple[1])]=[data.ent2cluster[triple[2]]]
            if (triple[1],data.ent2cluster[triple[2]]) in filtered_facts_head:
                filtered_facts_head[(triple[1],data.ent2cluster[triple[2]])].append(data.ent2cluster[triple[0]])
            else:
                filtered_facts_head[(triple[1],data.ent2cluster[triple[2]])]=[data.ent2cluster[triple[0]]]
    evaluation_data = data.test_data if eval_set=='test' else data.valid_data
    filtered_lists_head = []
    filtered_lists_tail = []
    for triple in evaluation_data:
        filtered_lists_head.append(filtered_facts_head[(triple[1],data.ent2cluster[triple[2]])])
        filtered_lists_tail.append(filtered_facts_tail[(data.ent2cluster[triple[0]],triple[1])])
    h_rank = model.get_rank(evaluation_data, filtered_lists_head, evalTail=False)
    t_rank = model.get_rank(evaluation_data, filtered_lists_tail, evalTail=True)
    for i,n in enumerate(hits):
        for rank in h_rank+t_rank:
            if rank<=n:
                HatN[i]+=1
    n_ranks = len(h_rank)+len(t_rank)
    h_rrank = sum(1.0/x for x in h_rank)/len(h_rank)
    h_rank = sum(h_rank)/len(h_rank)
    t_rrank = sum(1.0/x for x in t_rank)/len(t_rank)
    t_rank = sum(t_rank)/len(t_rank)
    all_rank = (h_rank+t_rank)/2
    all_rrank = (h_rrank+t_rrank)/2
    if logger is None:
        print(f"MR:\tHead {h_rank:.2f}\tTail {t_rank:.2f}\tOverall {all_rank:.2f}")
        print(f"MRR:\tHead {h_rrank:.3f}\tTail {t_rrank:.3f}\tOverall {all_rrank:.3f}")
        print("H@N\t","\t".join([f"{hits[i]}: {HatN[i]/n_ranks:.3f}" for i in range(len(hits))]))
    else:
        logger.info(f"MR:\tHead {h_rank:.2f}\tTail {t_rank:.2f}\tOverall {all_rank:.2f}")
        logger.info(f"MRR:\tHead {h_rrank:.3f}\tTail {t_rrank:.3f}\tOverall {all_rrank:.3f}")
        logger.info("H@N\t"+("\t".join([f"{hits[i]}: {HatN[i]/n_ranks:.3f}" for i in range(len(hits))])))
    return all_rrank #assuming that we pick model based on MRR

