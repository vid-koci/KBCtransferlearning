#collect GloVe
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove
rm glove.6B.zip

#collect OlpBench benchmark
wget http://data.dws.informatik.uni-mannheim.de/olpbench/olpbench.tar.gz
tar xzvf olpbench.tar.gz
mv olpbench Data/
mkdir Data/OlpBench
python transform_olpbench_data.py
rm -r Data/olpbench
rm olpbench.tar.gz
