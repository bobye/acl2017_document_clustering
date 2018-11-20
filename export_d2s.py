from gensim import  models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
from gensim.models import word2vec,LdaModel
from gensim.corpora import MmCorpus

import cPickle,os,re,codecs,sys,glob

from categoricalClustDist import categorical_clust_dist,token_to_mat
#from spelling_corrector import correct

def extract_features(in_file='story_clusters.txt',features_file='feature_words.txt', filtered_category_list=[]):
    f = codecs.open(in_file,'r',encoding='utf-8',errors='ignore')
    clusters = f.read().strip('%%%').split('\n%%%')
    docs = []
    labels = []
    word_length = 0
    document_count = 0
    for i,cluster in enumerate(clusters):
        if (not filtered_category_list) or (cluster.strip().split('\n')[0] in filtered_category_list):
            for ln in cluster.strip().split('\n')[1:]:
                docs.append(ln)
                labels.append(i)
                document_count = document_count + 1
                word_length = word_length + len(ln.split(' '));

    print "raw categories: %d" % len(clusters)
    print "document count: %d" % document_count
    print "average words: %d" % (word_length / document_count)

    vectorizer = CountVectorizer(lowercase=True, stop_words='english', token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b")
    vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names()
    fw = open(features_file,'w')
    fw.write(u'\n'.join(features).encode('utf-8'))
    fw.close()

def build_word2vec_index(feature_file='feature_words.txt', word_vecs_file='GoogleNews-vectors-negative300.bin', out_dic='word_vecs.pkl'):
    words = open(feature_file).read().strip().split('\n')
    dic = dict()
    model = word2vec.Word2Vec.load_word2vec_format(word_vecs_file, binary=True, norm_only=False)
    for w in words:
        try:
            v = model[w]
            dic[w] = v
        except:
            continue # whatever
            try:
                v = model[correct(w)]
                dic[w] = v
            except:
                print w + ' not existed in word2vec model'
                continue
    fw = open(out_dic,'wb')
    cPickle.dump(dic, fw)
    fw.close()

def convert_d2_format(in_file='story_clusters.txt', embedding_dic='word_vecs.pkl',embedding_dim_size=300, weighting_type='tf', d2_vocab='story_cluster.d2s.vocab0', d2s_file='story_cluster.d2s',filtered_category_list=[]):
    word2vec_dic = cPickle.load(open(embedding_dic,'rb'))
    vocab = word2vec_dic.keys()
    #clusters = open(in_file).read().strip('%%%').split('\n%%%')
    f = codecs.open(in_file,'r',encoding='utf-8',errors='ignore')
    clusters = f.read().strip('%%%').split('\n%%%')
    docs = []
    labels = []
    for i,cluster in enumerate(clusters):
        if (not filtered_category_list) or (cluster.strip().split('\n')[0] in filtered_category_list):
            for ln in cluster.strip().split('\n')[1:]:
                docs.append(ln)                
                labels.append(i)
    if weighting_type == 'tf':
        vectorizer = CountVectorizer(lowercase=True, stop_words='english', vocabulary=vocab)
    if weighting_type == 'tfidf':
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', vocabulary=vocab)
    X = vectorizer.fit_transform(docs)
    print X.shape
    fw = open(d2s_file,'w')
    for i in range(X.shape[0]):
        fw.write(str(embedding_dim_size)+'\n')
        nonzero_ids = X[i].nonzero()
        if (len(nonzero_ids[0])>0):
            fw.write(str(len(nonzero_ids[0]))+'\n')
            fw.write(' '.join(str(X[i][0,j]) for j in nonzero_ids[1])+'\n')
            fw.write(' '.join(str(j+1) for j in nonzero_ids[1])+'\n')
        else:
            print >> sys.stderr, "empty document found!"
            fw.write('1\n')
            fw.write('1\n')
            fw.write('0\n')
    fw.close()
    fw = open(d2_vocab,'w')
    fw.write(str(embedding_dim_size)+' '+str(len(vocab))+'\n')
    fw.write('\n'.join(' '.join(str(v) for v in word2vec_dic[w]) for w in vocab))
    fw.close()


def d2clustering_metrics(in_file='story_clusters.txt',in_label_file=None,filtered_category_list=[]):
    f = codecs.open(in_file, 'r', encoding='utf-8', errors='ignore')
    clusters = f.read().strip('%%%').split('\n%%%')
    labels = []
    for i,cluster in enumerate(clusters):
        if (not filtered_category_list) or (cluster.strip().split('\n')[0] in filtered_category_list):
            for ln in cluster.strip().split('\n')[1:]:
                labels.append(i)
    lines = open(in_label_file).read().strip().split()
    d2_labels = [int(i) for i in lines]
    print 'number of clusters: %d' % (max(d2_labels)+1)
    print np.array(np.sum(token_to_mat(labels), axis=0), dtype='int32')
    print np.array(np.sum(token_to_mat(d2_labels), axis=0), dtype='int32')
    print 'Homogeneity: %0.3f' % metrics.homogeneity_score(labels, d2_labels)
    print 'Completeness: %0.3f' % metrics.completeness_score(labels, d2_labels)
    print 'V-measure: %0.3f' % metrics.v_measure_score(labels, d2_labels)
    print 'Normalized Mutual Information: %0.3f' % metrics.normalized_mutual_info_score(labels, d2_labels)
    print 'Adjusted Mutual Information: %0.3f' % metrics.adjusted_mutual_info_score(labels, d2_labels)
    print 'Adjusted Rand Index: %0.3f' % metrics.adjusted_rand_score(labels, d2_labels)
    #print 'Categorical Cluster Distance: %0.3f' % categorical_clust_dist(token_to_mat(labels), token_to_mat(d2_labels), method='instance_count')['dist']


if __name__ == '__main__':
    dataset = 'story'
    #dataset = 'bbc'
    #dataset = 'bbc_title'
    #dataset = 'bbc_abstract'
    #dataset = 'reuters'
    #dataset = 'reuters_title'
    #dataset = 'bbcsport'
    #dataset = 'bbcsport_title'
    #dataset = '20newsclean'
    #dataset = 'ohsumed_sz25'

    vec_dim = 400
    word_vecs='word2vec_400_10_10.bin'
    #word_vecs='glove_6B_300d.bin'
    #word_vecs='GoogleNews-vectors-negative300.bin'
    #word_vecs='ohsumed-full_50_20_2.8.bin'
    
    cluster_file = 'acl2017dataset/' + dataset + '_clusters.txt'
    vec_dic = dataset + '_word_vecs.pkl'

    reuters_r10_categories = ['acq', 'crude', 'earn', 'coffee', 'interest', 'money-fx', 'money-supply', 'ship', 'trade', 'sugar']

    if dataset == 'reuters':
        category_list = reuters_r10_categories
    else:
        category_list = []

    is_result_avail = False;
    for label_file in glob.glob("./"+dataset+"_*_o"):
        print '---------------------------------------------------'
        print 'Method: D2 Clustering'
        print 'Vocabulary Embedding: ' + word_vecs
        d2clustering_metrics(in_file=cluster_file, in_label_file=label_file, filtered_category_list=category_list)
        is_result_avail = True;

    if is_result_avail:
        sys.exit(0)


    if dataset.startswith('reuters'):
        extract_features(in_file=cluster_file, features_file='reuters.terms', filtered_category_list = reuters_r10_categories)
        build_word2vec_index(feature_file='reuters.terms', word_vecs_file=word_vecs, out_dic=vec_dic)
        convert_d2_format(in_file=cluster_file, embedding_dic=vec_dic, embedding_dim_size=vec_dim, weighting_type='tfidf', d2_vocab=dataset + '_cluster.d2s.vocab0', d2s_file=dataset + '_cluster.d2s', filtered_category_list = reuters_r10_categories)
    else:
        extract_features(in_file=cluster_file, features_file=dataset + '.terms')
        build_word2vec_index(feature_file=dataset + '.terms', word_vecs_file=word_vecs, out_dic=vec_dic)
        convert_d2_format(in_file=cluster_file, embedding_dic=vec_dic, embedding_dim_size=vec_dim, weighting_type='tfidf', d2_vocab=dataset + '_cluster.d2s.vocab0', d2s_file=dataset + '_cluster.d2s')

        #
    #build_word2vec_index()
    #convert_d2_format()

