#!/usr/bin/env python
# coding: utf-8

#LIBRARY OF DATA REPRESENTATION BUILDING TOOLS
import model
import warnings
import random
warnings.filterwarnings("ignore")
import pandas as pd
import torch
import tensorflow as tf
import math
import time
from transformers import BertTokenizer, AutoModel, BertModel, AutoTokenizer

from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


import numpy
print ('Pytorch Version:', torch.__version__)
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize

from numpy import array
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import numpy as np
import re
from tqdm import tqdm_notebook
from time import mktime
import csv 
import spacy 
#max_seq_length=1024
MAX_LEN = 512

from bert.tokenization import FullTokenizer
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

import os
 
'''
The first block of models and tokenizers is for GENERIC BERT model.
The second block uses the models produced from finetuning the BERT model
'''
# Load BERT models and tokenizers.
#tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
#bert_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

pretrained_bertpath = "./models/BertFinetune/clustFT-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_bertpath, max_length=MAX_LEN)
bert_model = BertForSequenceClassification.from_pretrained(pretrained_bertpath)

entmodel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
#entmodel.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ('DEVICE IS', device)
entmodel.to(device)

#CREATE BASE LANGUAGE MODEL AND 
nlp = spacy.load('en_core_web_sm') 
gaussian_sigma = 1
                           
#LOAD SAVED TFIDF MODELS FROM DISK
transformer = TfidfTransformer()

token_vec = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open("models/token_feature.pkl", "rb")))
entity_vec = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open("models/entity_feature.pkl", "rb")))
lemma_vec = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open("models/lemma_feature.pkl", "rb")))

def c_score (reps, clfmodel):
    
    model_keys = model.model_reps + model.time_reps

    model_weights = clfmodel.load_SVMweights(model_keys)
    c_score = sparse_dotprod(reps, model_weights)

    return(c_score)

def sparse_dotprod(fv0, fv1):
    dotprod = 0
    for f_id_0, f_value_0 in fv0.items():
        if f_id_0 in fv1:
            f_value_1 = fv1[f_id_0]
            dotprod += f_value_0 * f_value_1
    
    return dotprod

def sim_reps_dc(d0, c1):
    numdays_stddev = 3.0
    reps = clustdoc_similarities(d0.reprs, c1.reprs )

    reps["NEWEST_TS"] = timestamp_feature(
        d0.timestamp.timestamp(), c1.newest_timestamp.timestamp(), numdays_stddev)
    reps["OLDEST_TS"] = timestamp_feature(
        d0.timestamp.timestamp(), c1.oldest_timestamp.timestamp(), numdays_stddev)
    reps["RELEVANCE_TS"] = timestamp_feature(
        d0.timestamp.timestamp(), c1.get_relevance_stamp(), numdays_stddev)
    return reps

def gaussian_sim(x1,x2,sigma):
    dist = np.linalg.norm(x1-x2)
    return np.exp(-dist**2/(2.*(sigma**2.)))

def normalized_gaussian(mean, stddev, x):
  return (math.exp(-((x - mean) * (x - mean)) / (2 * stddev * stddev)))

def timestamp_feature(tsi, tst, gstddev):
  return normalized_gaussian(0, gstddev, (tsi-tst)/(60*60*24.0))

def clustdoc_similarities (document_rep, cluster_rep):
    similarities = {}
    for n in model.model_reps:
        similarities[n] = cosine_sim(cluster_rep[n], document_rep[n])[0][0]
    return (similarities)

def cosine_sim(A, B):
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
    return (cos_sim)

def build_features(document):
    rep_ = get_base(document)
    rep_df = add_tokenized(rep_)
    rep_df = add_lemmas(rep_df)
    rep_df = add_entities(rep_df)
    docfeatures = get_tfidf(rep_df)
    return (docfeatures)

def create_vec(similarities):
    sim_vec = []
    for s in similarities.values():
        if isinstance(s, (list, tuple, np.ndarray)):
            sim_vec.append(s[0][0])
        else:
            sim_vec.append(s) 
    return(sim_vec)

'''
Get Entities from the text sentence using Spacy library. If mode is set to 1, it will return a long string. Otherwise it will return a list of entities.
'''
def get_entity_spacy(sentence, mode):
    doc = nlp(''.join(str(sentence)))
    ents = [ent.text for ent in doc.ents] 
    if mode ==1: 
        return " ".join(ents)
    else:
        return list(set(ents))

def get_entity_nltk(sentence):
    word = nltk.word_tokenize(sentence)
    pos_tag = nltk.pos_tag(word)
    NE = ["".join(w for w, t in ele) for ele in chunk if instance(ele, nltk.Tree)]
    return (NE)

def get_lemma(sentence):
    prep = preprocess_text(sentence)
    doc = nlp(sentence)
    lemmas = " ".join([token.lemma_ for token in doc])
    return (lemmas)


def get_base(df):
    base = {}
    base['pid'] = (df.id)
    base['cluster'] = (df.cluster)
    base['title'] = (df.title)
    base['body'] = (df.body)
    base['timestamp'] = (df.timestamp)
    base['titlebody'] = (str(df.title)+str(df.body))
 
    #rep_df = pd.DataFrame(list(zip(pid, cluster, title, body, timestamp, title_body)), columns=['id','cluster','title','body', 'timestamp', 'title_body'])
    return (base)

def add_tokenized(rep_df):
    rep_df['title_tok'] = preprocess_text(rep_df['title'])
    rep_df['body_tok'] = preprocess_text(rep_df['body'])
    rep_df['titlebody_tok'] = preprocess_text(rep_df['titlebody'])
    return (rep_df)

def add_lemmas(rep_df):
    rep_df['body_lemmas'] = get_lemma(rep_df['body'])
    rep_df['title_lemmas'] = get_lemma(rep_df['title'])
    rep_df['titlebody_lemmas'] = get_lemma(rep_df['titlebody'])
    return (rep_df)


def add_entities(rep_df):
    #get Entities and Lemma
    rep_df['body_entities'] = get_entity_spacy(rep_df['body'], 1)
    rep_df['title_entities'] = get_entity_spacy(rep_df['title'], 1)
    rep_df['titlebody_entities'] = get_entity_spacy(rep_df['titlebody'], 1)
    
    return (rep_df)

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def preprocess_text(sen):
    sen = str(sen)
    
    stop_words = set(stopwords.words('english'))
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    the_tags = re.compile('<.*?>')
    tagfree = re.sub(the_tags, '', sentence)
    sentence = re.sub(r'\s+', ' ', tagfree)
    word_tokens = word_tokenize(sentence)     
    no_stop_words = [w for w in word_tokens if not w in stop_words]

    return " ".join(no_stop_words)


def get_tfidf(rep_df):
    '''
    Parameters:  a dataframe of tite, body, title&body lemmas and tokens and entites and 
    Returns: a dictionary of label:vectors representing each of their tf_idf values 
    respectively
    
    '''
    tf_idf = {}
    tf_idf['id'] = rep_df['pid']
    tf_idf['cluster'] = rep_df['cluster']
    
    tf_idf['title'] = token_vec.fit_transform([rep_df['title_tok']]).todense()
    tf_idf['body'] = token_vec.fit_transform([rep_df['body_tok']]).todense()    
    tf_idf['titlebody'] = token_vec.fit_transform([rep_df['titlebody_tok']]).todense()
    tf_idf['title_lemmas'] = lemma_vec.fit_transform([rep_df['title_lemmas']]).todense()
    tf_idf['body_lemmas'] = lemma_vec.fit_transform([rep_df['body_lemmas']]).todense()
    tf_idf['titlebody_lemmas'] = lemma_vec.fit_transform([rep_df['titlebody_lemmas']]).todense()
    tf_idf['title_entities'] = entity_vec.fit_transform([rep_df['title_entities']]).todense()
    tf_idf['body_entities'] = entity_vec.fit_transform([rep_df['body_entities']]).todense()
    tf_idf['titlebody_entities'] =  entity_vec.fit_transform([rep_df['titlebody_entities']]).todense()

    #GET BOTH SETS OF BERT EMBEDDINGS: FOR THE BODY, AND THOSE WITH POSITIONAL VECTORS THAT DENOTE ENTITIES
    tf_idf['bert'] = dense_body_embeddings([rep_df['body']],device)
    
    tf_idf['bert_ent'] = bert_ents([rep_df['body']])
    #tf_idf['bert_ent'] = get_maskembedding(entmodel, entmodel_tokenizer, rep_df['body'], device, get_entity(rep_df['body']) )

    #list_of_dicts = [dict(zip(tf_idf,t)) for t in zip(*tf_idf.values())]

    return (tf_idf)

def get_accuracy (accurate, allc, testsize):
    cs = []
    sing_errtot = 0
    for c in allc:
        cs.append(c.clustid)
        
    dupesdict = {i:cs.count(i) for i in cs}
    
    for key,val in dupesdict.items():
        err = val - 1
        sing_errtot = sing_errtot + err 
    print (dupesdict)
    return ((accurate-sing_errtot)/testsize) 

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def dense_body_embeddings_old(body):
    encoded_input = tokenizer(body, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sen_embarray = sentence_embeddings.numpy()
    return (sen_embarray)


def dense_body_embeddings(body, device):
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(body[0], truncation=True, max_length=MAX_LEN)).unsqueeze(0)  
        input_ids.to(device)
        outputs = bert_model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return(last_hidden_states.numpy())

def get_word_indeces(tokenizer, text, word):

    word_tokens = tokenizer.tokenize(word)
    masks_str = ' '.join(['[MASK]']*len(word_tokens))
    text_masked = text.replace(word, masks_str)
    input_ids = tokenizer.encode(text_masked, truncation=True)
    mask_token_indeces = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_indeces

def get_BERTids(body):
    input_ids = []

    for sent in body:
        encoded_sent = tokenizer.encode(
                        sent,                      
                        add_special_tokens = True, 
                        truncation = True,
                      )
        input_ids.append(encoded_sent)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post")
    return input_ids

def get_BERTattention(input_ids):
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return (attention_masks)

def get_BERTentity_mask_old(body):
    entity_masks = []
    for sent in body:
        entity_mask = []
        tokenized = tokenizer.tokenize(sent)
        for token in tokenized:
            #Create the Entity Embedding
            if (get_entity_spacy(token, 2)):
                entity_mask.append(1)
            else:
                entity_mask.append(0)
        #print (list(zip(tokenized, entity_mask)))
        entity_masks.append(entity_mask)
    entity_masks = pad_sequences(entity_masks, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post")
    return (entity_masks)


def get_BERTentity_masks(body):
    #TAKE AN EXCERPT FROM THE BODY - FIRST PARAGRAPHS CONTAIN 5Ws ON NEWS STORIES GENERALLY
    body = body[0]
    entity_masks = [0] * MAX_LEN
    #GET THE LIST OF ENTITIES CONTAINED IN THE EXCERPT
    ents = get_entity_spacy(body, 2)
    word_indeces = []
  
    #RUN PASSES TO GET THE BERT INDICES WHERE THE ENTITIES HAVE OCCURRED

    for e in ents:
        i = get_word_indeces(tokenizer, body, e)
        word_indeces.extend(i)
    word_indeces.sort()
   
    #CREATE A ONE HOT VECTOR THAT MATCHES THE INDEX AND PADDED TO THE MAX LENGTH 
    for i in word_indeces:
        entity_masks[i] =1
    
    entity_masks = pad_sequences([entity_masks], maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    inverse_masks=(~entity_masks.astype(bool)).astype(int)
 
    return (entity_masks)

def get_BERTsegments(body):
    tokenized_text = tokenizer.tokenize(body[0], truncation = True)
    segment_ids = [[1] * MAX_LEN]
    return segment_ids

def bert_ents(body):

    tok_ids = get_BERTids(body)
    att_masks = get_BERTattention(tok_ids)
    ent_masks = get_BERTentity_masks(body)
    seg_masks = get_BERTsegments(body)

    tok_ids = torch.tensor(tok_ids).to(device)
    att_masks = torch.tensor(att_masks).to(device)
    ent_masks = torch.tensor(ent_masks).to(device)
    seg_masks = torch.tensor(seg_masks).to(device)

    #train_data = TensorDataset(tok_ids, att_masks, ent_masks, seg_masks)
    #train_sampler = RandomSampler(train_data)
    #train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)
    with torch.no_grad():
        if torch.cuda.is_available():
            output = entmodel(tok_ids,  att_masks,ent_masks, seg_masks)

    logits = output[1]
    logits = logits.cpu().numpy()
    torch.cuda.empty_cache()

    return (logits)


def get_maskembedding(b_model, b_tokenizer, text, device, ents='' ):
    '''
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`
    '''
    if not ents == '':
        word_indeces = []
        for e in ents:
            i = get_word_indeces(b_tokenizer, text, e)
            word_indeces.extend(i)
        word_indeces.sort()
    
    encoded_dict = b_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        truncation =True,
                        max_length=MAX_LEN,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        return_tensors = 'pt'     # Return pytorch tensors.
                    )

    input_ids = encoded_dict['input_ids']
    
    with torch.no_grad():
        input_ids.to(device)
        outputs = b_model(input_ids)
        hidden_states = outputs[2]

    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.detach().numpy()

    return sentence_embedding





#=========== this needs to be repeated because pickle currently does not have serialization and does not know the loaded class
class Cluster:
    def __init__(self, document):

        self.ids = set()
        self.num_docs = 0
        self.reprs = {}
        self.sum_timestamp = 0
        self.sumsq_timestamp = 0
        self.newest_timestamp = datetime.datetime.strptime(
            "1000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        self.oldest_timestamp = datetime.datetime.strptime(
            "3000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        self.add_document(document)  
        
    def get_relevance_stamp(self):
        z_score = 1
        mean = self.sum_timestamp / self.num_docs
        try:
          std_dev = math.sqrt((self.sumsq_timestamp / self.num_docs) - (mean*mean))
        except:
          std_dev = 0.0
        return mean + ((z_score * std_dev) * 3600.0) # its in secods since epoch

    def add_document(self, document):
        self.ids.add(document.id)
        self.newest_timestamp = max(self.newest_timestamp, document.timestamp)
        self.oldest_timestamp = min(self.oldest_timestamp, document.timestamp)
        ts_hours =  (document.timestamp.timestamp() / 3600.0)
        self.sum_timestamp += ts_hours
        self.sumsq_timestamp += ts_hours * ts_hours
        self.__add_reps(document.reprs)
   
    def __add_reps(self, reprs0):

        if self.reprs: 
            for n in model.model_reps:
                self.reprs[n] = np.nanmean(np.array([self.reprs[n], reprs0[n]]), axis=0)
        else:
            self.reprs = reprs0        
        self.num_docs += 1   