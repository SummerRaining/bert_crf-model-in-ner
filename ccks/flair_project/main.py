# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:16:05 2020

@author: tunan
"""
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from custom_embedding import WordEmbeddings
from flair.embeddings.token import TransformerWordEmbeddings,StackedEmbeddings
import torch
# define columns
torch.cuda.empty_cache()
columns = {0: 'text', 1: 'ner'}
data_folder = 'dataset'
tag_type = 'ner'
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt', 
                              dev_file='dev.txt')
#print(corpus.train[0].to_tagged_string('ner'))
tag_dictionary = corpus.make_tag_dictionary(tag_type =tag_type)

# 4. initialize embeddings
embedding_types = [

    WordEmbeddings('custom'),
    TransformerWordEmbeddings('bert-base-chinese'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

#覆盖另外一个
from flair.models import SequenceTagger
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_type = tag_type,
                                        tag_dictionary=tag_dictionary,
                                        use_crf=True)
trainer: ModelTrainer = ModelTrainer(tagger, corpus)
# 7. start training
trainer.train('resources/ccks/bert_lstm_crf',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

