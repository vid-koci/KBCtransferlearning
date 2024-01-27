import argparse
import os
import json
import scipy.stats

def wilcoxon_test(predictions_file_x, predictions_file_y):
    predictions1 = [float(x.strip()) for x in list(open(predictions_file_x))]
    predictions2 = [float(y.strip()) for y in list(open(predictions_file_y))]
    print("p-value: ", scipy.stats.wilcoxon(predictions1,predictions2))


def compute_general_metrics(data_folder, predictions_file):
    predictions = list(open(predictions_file))
    metadata = list(open(os.path.join(data_folder,"test_metadata.jsonlines")))
    subgroups = {}
    id2rank = {}
    for rank, group in zip(predictions,metadata):
        rank = int(rank.strip())
        group = json.loads(group.strip())
        id2rank[group["id"]]=rank
        if "category" in group and group["category"] not in subgroups:
            subgroups[group["category"]]=[]
        if "category" in group:
            subgroups[group["category"]].append(rank)
    for group, ranks in subgroups.items():
        print(group, "\tMR: ",sum(ranks)/len(ranks), "\tMRR: ", sum(1.0/x for x in ranks)/len(ranks), "\tH@1: ",sum(1 for x in ranks if x==1)/len(ranks))
    entity_synonym_rank_diffs = []
    relation_synonym_rank_diffs = []
    inverse_relation_rank_diffs = []
    deductive_subset_rank = []
    for group in metadata:
        group = json.loads(group.strip())
        if "entity_synonym_of" in group:
            entity_synonym_rank_diffs.append(id2rank[group["id"]]-id2rank[group["entity_synonym_of"]])
        if "relation_synonym_of" in group:
            relation_synonym_rank_diffs.append(id2rank[group["id"]]-id2rank[group["relation_synonym_of"]])
        if "inverse_relation_of" in group:
            inverse_relation_rank_diffs.append(id2rank[group["id"]]-id2rank[group["inverse_relation_of"]])
        if "deductive_subset" in group:
            deductive_subset_rank.append(id2rank[group["id"]])
    if len(entity_synonym_rank_diffs)>0:
        print("Entity synonym impact: mean ",sum(entity_synonym_rank_diffs)/len(entity_synonym_rank_diffs), " stdev ",scipy.stats.tstd(entity_synonym_rank_diffs))
    if len(relation_synonym_rank_diffs)>0:
        print("Relation synonym impact: mean ",sum(relation_synonym_rank_diffs)/len(relation_synonym_rank_diffs), " stdev ",scipy.stats.tstd(relation_synonym_rank_diffs))
    if len(inverse_relation_rank_diffs)>0:
        print("Inverse Relation impact: mean ",sum(inverse_relation_rank_diffs)/len(inverse_relation_rank_diffs), " stdev ",scipy.stats.tstd(inverse_relation_rank_diffs))
    if len(deductive_subset_rank)>0:
        print("Metrics on the deductive reasoning subset:\tMR: ",sum(deductive_subset_rank)/len(deductive_subset_rank), "\tMRR: ", sum(1.0/x for x in deductive_subset_rank)/len(deductive_subset_rank), "\tH@1: ",sum(1 for x in deductive_subset_rank if x==1)/len(deductive_subset_rank))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analysis of OKBC diagnostic outputs")
    parser.add_argument('-gender_stereotypical_data',default=None, help='folder with the gender stereotypical data')
    parser.add_argument('-gender_stereotypical_predictions',default=None, help='model predictions on gender stereotypical data')
    parser.add_argument('-gender_anti_stereotypical_data',default=None, help='folder with the gender anti-stereotypical data')
    parser.add_argument('-gender_anti_stereotypical_predictions',default=None, help='model predictions on gender anti-stereotypical data')
    parser.add_argument('-occupations_data',default=None, help='folder with the data about occupations')
    parser.add_argument('-occupations_predictions',default=None, help='model predictions on data about occupations')
    parser.add_argument('-general_data',default=None, help='folder with the general knowledge data')
    parser.add_argument('-general_predictions',default=None, help='model predictions on general knowledge data')
    parser.add_argument('-deductive_reasoning_data',default=None, help='folder with the general knowledge data')
    parser.add_argument('-deductive_reasoning_predictions',default=None, help='model predictions on general knowledge data')
    args = parser.parse_args()
    if args.gender_stereotypical_data is not None and args.gender_stereotypical_predictions is not None:
        compute_general_metrics(args.gender_stereotypical_data,args.gender_stereotypical_predictions)
    if args.gender_anti_stereotypical_data is not None and args.gender_anti_stereotypical_predictions is not None:
        compute_general_metrics(args.gender_anti_stereotypical_data,args.gender_anti_stereotypical_predictions)
    if args.gender_stereotypical_predictions is not None and args.gender_anti_stereotypical_predictions is not None:
        print("Statistical significance of the gender impact (Wilcoxon test)")
        wilcoxon_test(args.gender_stereotypical_predictions,args.gender_anti_stereotypical_predictions)
    if args.occupations_data is not None and args.occupations_predictions is not None:
        compute_general_metrics(args.occupations_data,args.occupations_predictions)
    if args.general_data is not None and args.general_predictions is not None:
        compute_general_metrics(args.general_data,args.general_predictions)
    if args.deductive_reasoning_data is not None and args.deductive_reasoning_predictions is not None:
        compute_general_metrics(args.deductive_reasoning_data,args.deductive_reasoning_predictions)

