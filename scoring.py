import numpy as np
import tqdm

scoring_empty_return_value = 0.0


def evaluate(model, data, hits=[10, 30, 50], eval_set='valid', logger=None, out_file=None):
    h_rank = []
    h_hits = np.zeros(len(hits))
    t_rank = []
    t_hits = np.zeros(len(hits))
    HatN = [0] * len(hits)
    filtered_facts_tail = {}
    filtered_facts_head = {}
    for dataset in [data.train_data, data.valid_data, data.test_data]:
        for triple in dataset:
            if type(triple) is dict:  # for diagnostic dataset
                triple = triple["triple"]
            if (data.ent2cluster[triple[0]], triple[1]) in filtered_facts_tail:
                filtered_facts_tail[(data.ent2cluster[triple[0]],
                                     triple[1])].append(
                                         data.ent2cluster[triple[2]])
            else:
                filtered_facts_tail[(data.ent2cluster[triple[0]],
                                     triple[1])] = [
                                         data.ent2cluster[triple[2]]
                                     ]
            if (triple[1], data.ent2cluster[triple[2]]) in filtered_facts_head:
                filtered_facts_head[(triple[1],
                                     data.ent2cluster[triple[2]])].append(
                                         data.ent2cluster[triple[0]])
            else:
                filtered_facts_head[(triple[1],
                                     data.ent2cluster[triple[2]])] = [
                                         data.ent2cluster[triple[0]]
                                     ]
    evaluation_data = data.test_data if eval_set == 'test' else data.valid_data
    if len(evaluation_data) == 0:
        global scoring_empty_return_value  #dirty hack that ensures that when no validation set is given, the final evaluated model is returned.
        scoring_empty_return_value += 1e-8
        return scoring_empty_return_value
    model.prepare_evaluation()
    filtered_lists_head = []
    filtered_lists_tail = []
    for triple in evaluation_data:
        #head_filtered = []
        #tail_filtered = []
        #for entity in data.list_of_ent:
        #    tail_replaced = (data.ent2cluster[triple[0]],triple[1],data.ent2cluster[entity])
        #    if tail_replaced in filtered_facts:
        #        tail_filtered.append(entity)
        #    head_replaced = (data.ent2cluster[entity],triple[1],data.ent2cluster[triple[2]])
        #    if head_replaced in filtered_facts:
        #        head_filtered.append(entity)
        filtered_lists_head.append(
            filtered_facts_head[(triple[1], data.ent2cluster[triple[2]])])
        filtered_lists_tail.append(
            filtered_facts_tail[(data.ent2cluster[triple[0]], triple[1])])
    h_rank = model.get_rank(evaluation_data,
                            filtered_lists_head,
                            evalTail=False)
    t_rank = model.get_rank(evaluation_data,
                            filtered_lists_tail,
                            evalTail=True)
    for i, n in enumerate(hits):
        for rank in h_rank + t_rank:
            if rank <= n:
                HatN[i] += 1
    n_ranks = len(h_rank) + len(t_rank)
    h_rrank = sum(1.0 / x for x in h_rank) / len(h_rank)
    h_rank = sum(h_rank) / len(h_rank)
    t_rrank = sum(1.0 / x for x in t_rank) / len(t_rank)
    t_rank = sum(t_rank) / len(t_rank)
    all_rank = (h_rank + t_rank) / 2
    all_rrank = (h_rrank + t_rrank) / 2
    if logger is None:
        print(
            f"MR:\tHead {h_rank:.2f}\tTail {t_rank:.2f}\tOverall {all_rank:.2f}"
        )
        print(
            f"MRR:\tHead {h_rrank:.3f}\tTail {t_rrank:.3f}\tOverall {all_rrank:.3f}"
        )
        print(
            "H@N\t", "\t".join([
                f"{hits[i]}: {HatN[i]/n_ranks:.3f}" for i in range(len(hits))
            ]))
    else:
        logger.info(
            f"MR:\tHead {h_rank:.2f}\tTail {t_rank:.2f}\tOverall {all_rank:.2f}"
        )
        logger.info(
            f"MRR:\tHead {h_rrank:.3f}\tTail {t_rrank:.3f}\tOverall {all_rrank:.3f}"
        )
        logger.info("H@N\t" + ("\t".join(
            [f"{hits[i]}: {HatN[i]/n_ranks:.3f}" for i in range(len(hits))])))

    if out_file:
        with open(out_file, 'w') as f:
            mr_text = f"MR:\tHead {h_rank:.2f}\tTail {t_rank:.2f}\tOverall {all_rank:.2f}"
            f.write(f"\n {mr_text} \n")

            mrr_text = f"MRR:\tHead {h_rrank:.3f}\tTail {t_rrank:.3f}\tOverall {all_rrank:.3f}"
            f.write(f"\n {mrr_text} \n")

            hr_text = "H@N\t", "\t".join([
                f"{hits[i]}: {HatN[i]/n_ranks:.3f}" for i in range(len(hits))
            ])
            f.write(f"\n {hr_text} \n")
    return all_rrank  #assuming that we pick model based on MRR


def evaluate_diagnostic(model,
                        data,
                        hits=[10, 30, 50],
                        eval_set='test',
                        logger=None,
                        out_file="diagnostic_predictions.txt"):
    #design choice: just print ranks and have an external script produce actual scores - easier for future use
    filtered_facts_tail = {}
    filtered_facts_head = {}
    for triple in data.test_data:
        if (data.ent2cluster[triple["triple"][0]],
                triple["triple"][1]) not in filtered_facts_tail:
            filtered_facts_tail[(data.ent2cluster[triple["triple"][0]],
                                 triple["triple"][1])] = []
        filtered_facts_tail[(data.ent2cluster[triple["triple"][0]],
                             triple["triple"][1])].append(
                                 data.ent2cluster[triple["triple"][2]])
        if (triple["triple"][1], data.ent2cluster[triple["triple"][2]]
            ) not in filtered_facts_head:
            filtered_facts_head[(triple["triple"][1],
                                 data.ent2cluster[triple["triple"][2]])] = []
        filtered_facts_head[(triple["triple"][1],
                             data.ent2cluster[triple["triple"][2]])].append(
                                 data.ent2cluster[triple["triple"][0]])
        for ent in data.list_of_ent:
            if "alternative_tails" in triple:
                if ent not in triple["alternative_tails"]:
                    filtered_facts_tail[(data.ent2cluster[triple["triple"][0]],
                                         triple["triple"][1])].append(
                                             data.ent2cluster[ent])
            elif "alternative_heads" in triple:
                if ent not in triple["alternative_heads"]:
                    filtered_facts_head[(
                        triple["triple"][1],
                        data.ent2cluster[triple["triple"][2]])].append(
                            data.ent2cluster[ent])
    evaluation_data = [x["triple"] for x in data.test_data
                       ]  #if eval_set=='test' else data.valid_data
    model.prepare_evaluation()
    filtered_lists_head = []
    filtered_lists_tail = []
    for triple in evaluation_data:
        filtered_lists_head.append(
            filtered_facts_head[(triple[1], data.ent2cluster[triple[2]])])
        filtered_lists_tail.append(
            filtered_facts_tail[(data.ent2cluster[triple[0]], triple[1])])
    h_rank = model.get_rank(evaluation_data,
                            filtered_lists_head,
                            evalTail=False)
    t_rank = model.get_rank(evaluation_data,
                            filtered_lists_tail,
                            evalTail=True)
    combined_ranks = []
    for i in range(len(h_rank)):
        if "alternative_heads" in data.test_data[i]:
            combined_ranks.append(h_rank[i])
        elif "alternative_tails" in data.test_data[i]:
            combined_ranks.append(t_rank[i])
    with open(out_file, 'w') as f:
        f.write("\n".join(str(x) for x in combined_ranks) + "\n")
    return sum(1.0 / x for x in combined_ranks) / len(combined_ranks)
