SYNOPSIS
        python tagger3.py TASK (task can be 'pos' or 'ner')

To run tagger3.py you should also do the following:               
1) Run from a directory that has a sub directory named data
2) Within data you should place ner and pos directories that includes all the needed files (train,dev,test)
3) Make sure you have the files words_mapping_ner.txt and words_mapping_pos.txt in the current working directory - these are
   the mappings from words in the training set to their most similar word in the external vocabulary.
3) After the training phase - the script will predict the words that in the blind test file and place the resulted file
   in data/[pos/ner]/test1.[pos/ner] 
4) The script will also write the Dev loss & accuracy for each iterations in a separate files (in the current working directory).
