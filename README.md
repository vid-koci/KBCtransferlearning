# OKBCwithEncoders

This code accompanies the paper Knowledge Base Completion Meets Transfer Learning.


##Setup
Following packages are needed to run the code
 * Python >=3.6
 * pytorch>=1.6
 * spacy>=2.0 and en\_core\_web\_sm model
 * tqdm

Run setup.sh to download and transform GloVe embeddings and OlpBench dataset. Please note that this downloads 3.5GB of files which unzip into around 10GB of content.

##Running the code

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
