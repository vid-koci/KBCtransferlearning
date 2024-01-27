dim=500
batch=100
n_epochs=120
lr=3e-4
embedding=FiveStarE
dropout=0.4
regularization=0.3
encoder=NoEncoder
output_dir=bv2
pretrained_dir=FiveStarEOlpBench500
occupations_output=diagnostics/occupations_output.txt
general_output=diagnostics/general_output.txt
stereotypical_output=diagnostics/diagnostic_stereotypical_output.txt
antistereotypical_output=diagnostics/diagnostic_anti-stereotypical_output.txt
deductive_output=diagnostics/deductive_output.txt

#run pretrained gender stereotypical occupations
rm -r $output_dir
echo "pre-trained gender stereotypical occupations diagnostics"
python main.py -data diagnostics/gender_stereotypical -dim $dim -lr $lr -batch $batch -n_epochs $n_epochs -embedding $embedding -dropout $dropout -encoder $encoder -hits [1,10,50] -output_dir $output_dir -pretrained_dir $pretrained_dir -diagnostic -diagnostic_predictions_file $stereotypical_output -no_train -regularization $regularization

#run pretrained gender anti-stereotypical occupations
rm -r $output_dir
echo "pre-trained gender anti-stereotypical occupations diagnostics"
python main.py -data diagnostics/gender_anti-stereotypical -dim $dim -lr $lr -batch $batch -n_epochs $n_epochs -embedding $embedding -dropout $dropout -encoder $encoder -hits [1,10,50] -output_dir $output_dir -pretrained_dir $pretrained_dir -diagnostic -diagnostic_predictions_file $antistereotypical_output -no_train -regularization $regularization

#run occupations
rm -r $output_dir
echo "general occupations knowledge diagnostics"
python main.py -data diagnostics/occupations_knowledge -dim $dim -lr $lr -batch $batch -n_epochs $n_epochs -embedding $embedding -dropout $dropout -encoder $encoder -hits [1,10,50] -output_dir $output_dir -pretrained_dir $pretrained_dir -diagnostic -diagnostic_predictions_file $occupations_output -no_train -regularization $regularization

#run general knowledge
rm -r $output_dir
echo "general knowledge diagnostics"
python main.py -data diagnostics/general_knowledge -dim $dim -lr $lr -batch $batch -n_epochs $n_epochs -embedding $embedding -dropout $dropout -encoder $encoder -hits [1,10,50] -output_dir $output_dir -pretrained_dir $pretrained_dir -diagnostic -diagnostic_predictions_file $general_output -no_train -regularization $regularization

#run deductive dataset without training
rm -r $output_dir
echo "deductive reasoning without finetuning"
python main.py -data diagnostics/deductive_reasoning_test -dim $dim -lr $lr -batch $batch -n_epochs $n_epochs -embedding $embedding -dropout $dropout -encoder $encoder -hits [1,10,50] -output_dir $output_dir -pretrained_dir $pretrained_dir -diagnostic -diagnostic_predictions_file $deductive_output -no_train -regularization $regularization

#analysis of obtained results
python diagnostics/output_analysis.py -gender_anti_stereotypical_data diagnostics/gender_anti-stereotypical -gender_anti_stereotypical_predictions $antistereotypical_output -gender_stereotypical_data diagnostics/gender_stereotypical -gender_stereotypical_predictions $stereotypical_output -occupations_data diagnostics/occupations_knowledge -occupations_predictions $occupations_output -general_data diagnostics/general_knowledge -general_predictions $general_output -deductive_reasoning_data diagnostics/deductive_reasoning_test -deductive_reasoning_predictions $deductive_output

#run finetuning of diagnostic dataset
rm -r $output_dir
echo "fine-tuned diagnostic reasoning dataset"
python main.py -data diagnostics/deductive_reasoning_test -dim $dim -lr $lr -batch $batch -n_epochs $n_epochs -embedding $embedding -dropout $dropout -encoder $encoder -hits [1,10,50] -output_dir $output_dir -pretrained_dir $pretrained_dir -diagnostic -diagnostic_predictions_file $deductive_output -regularization $regularization
python diagnostics/output_analysis.py -deductive_reasoning_data diagnostics/deductive_reasoning_test -deductive_reasoning_predictions $deductive_output

#run finetuning of gender stereotypical occupations
rm -r $output_dir
echo "fine-tuned gender stereotypical occupations diagnostics"
python main.py -data diagnostics/gender_stereotypical -dim $dim -lr $lr -batch $batch -n_epochs $n_epochs -embedding $embedding -dropout $dropout -encoder $encoder -hits [1,10,50] -output_dir $output_dir -pretrained_dir $pretrained_dir -diagnostic -diagnostic_predictions_file $stereotypical_output -regularization $regularization

#run finetuning of gender anti-stereotypical occupations
rm -r $output_dir
echo "fine-tuned gender anti-stereotypical occupations diagnostics"
python main.py -data diagnostics/gender_anti-stereotypical -dim $dim -lr $lr -batch $batch -n_epochs $n_epochs -embedding $embedding -dropout $dropout -encoder $encoder -hits [1,10,50] -output_dir $output_dir -pretrained_dir $pretrained_dir -diagnostic -diagnostic_predictions_file $antistereotypical_output -regularization $regularization

echo "fine-tuned model results on gender bias:"
python diagnostics/output_analysis.py -gender_anti_stereotypical_data diagnostics/gender_anti-stereotypical -gender_anti_stereotypical_predictions $antistereotypical_output -gender_stereotypical_data diagnostics/gender_stereotypical -gender_stereotypical_predictions $stereotypical_output
 
