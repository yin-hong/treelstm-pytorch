#!/usr/bin/env bash
set -e


echo "Downloading SICK Dataset"
mkdir -p data/sick/
cd data/sick/
wget -c -q http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip
unzip -C -q -o sick_train.zip
rm -r sick_train.zip
wget -c -q http://alt.qcri.org/semeval2014/task1/data/uploads/sick_trial.zip
unzip -C -q -o sick_trial.zip
rm -r sick_trial.zip
wget -c -q http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip
unzip -C -q -o sick_test_annotated.zip
rm -r sick_test_annotated.zip
rm -rf readme.txt
cd ../../

echo "Downloading Stanford Parser"
cd lib/
wget -c -q http://nlp.stanford.edu/software/stanford-postagger-2015-01-29.zip
unzip -C -q stanford-postagger-2015-01-29.zip
rm -r stanford-postagger-2015-01-29.zip
mv stanford-postagger-2015-01-30/ stanford-tagger

wget -c -q http://nlp.stanford.edu/software/stanford-parser-full-2015-01-29.zip
unzip -C -q stanford-parser-full-2015-01-29.zip
rm -r stanford-parser-full-2015-01-29.zip
mv stanford-parser-full-2015-01-30/ stanford-parser

