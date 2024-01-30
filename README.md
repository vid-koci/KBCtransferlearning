# Knowledge Base Completion Meets Transfer Learning

This code accompanies the paper [Knowledge Base Completion Meets Transfer Learning](https://arxiv.org/abs/2108.13073) published at EMNLP 2021 and [ Pre-training and Diagnosing Knowledge Base Completion Models](https://arxiv.org/abs/2401.15439) published in AI Journal.


## Setup
Following packages are needed to run the code
 * Python >=3.6
 * pytorch>=1.6
 * spacy>=2.0 and en\_core\_web\_sm model
 * tqdm

Run setup.sh to download and transform GloVe embeddings and OlpBench dataset. Please note that this downloads 3.5GB of files which unzip into around 10GB of content.

## Running the code

For full help, run `python main.py -h`, a couple of examples are given below:

For pre-training TuckER on OlpBench, run
```
python main.py -data Data/OlpBench -dim 300 -lr 1e-4 -batch 4096 -n_epochs 100 -embedding TuckER -dropout 0.3 -encoder GRU -hits [1,3,5,10,30,50] -output_dir TuckEROlpBench300 -dump_vocab -only_batch_negative
```
For pretraining, it is important to add `-dump_vocab` to store encoder vocabulary. Otherwise it is not possible to load the stored model for fine-tuning.
For any large-scale pre-training it is important to add `-only_batch_negative` argument to avoid encoding all entities at every training step.

To fine-tune the model obtained with the above command on ReVerb20K using NoEncoder, use the command below.
```
python main.py -data Data/ReVerb20K -dim 300 -lr 3e-4 -batch 512 -n_epochs 500 -embedding TuckER -dropout 0.3 -encoder NoEncoder -hits [1,3,5,10,30,50] -output_dir TuckERReVerb20K -pretrained_dir TuckEROlpBench300
```

To train the with same setup but from a randomly-initialized model, just remove the `-pretrained_dir` argument.
```
python main.py -data Data/ReVerb20K -dim 300 -lr 3e-4 -batch 512 -n_epochs 500 -embedding TuckER -dropout 0.3 -encoder NoEncoder -hits [1,3,5,10,30,50] -output_dir TuckERReVerb20K
```

## Running diagnostics on the DOGE dataset

Given you have trained a pre-trained model, use the `run_full_diagnostics.sh` script to run all the diagnostics on the model.
Update the parameters at the top of the script to fit your usecase. This will run the following tests:
 * Detect how much the model associates gender names with stereotypical occupations on `diagnostics/gender_stereotypical` dataset
 * Detect how much the model associates gender names with anti-stereotypical occupations on `diagnostics/gender_anti-stereotypical` dataset
 * Detect how well the model performs on the knowledge of occupations, used in the above test, using the `diagnostics/occupations_knowledge` dataset
 * Detect how well the model performs on the general knowledge test, using the `diagnostics/general_knowledge` dataset
 * Perform the deductive reasoning tests, with and without fine-tuning (for comparison), using `diagnostics/deductive_reasoning_test` dataset
 * Repeat the tests on gender stereotypical and gender anti-stereotypical tests, this time with fine-tuning on facts about individuals, to detect how the stereotypes affect its deductive reasoning ability.
 * Run the diagnostics and tests of statistical significance with `diagnostics/output_analysis.py`

## Reference
If you use the code from this repo, please cite the following work.
```
@inproceedings{kocijan2021KBCtransfer,
    title = "Knowledge Base Completion Meets Transfer Learning",
    author = "Kocijan, Vid  and
      Lukasiewicz, Thomas",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
}
```
