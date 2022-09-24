# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:05:46 2022

@author: kmartin
"""

#Startup Code Begin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
import praw
import json
import itertools
import time
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN as dscan
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier as KNN
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from scipy.stats import zscore,norm,lognorm,skewnorm,laplace,kurtosis
from collections import Counter

#tmp = ((datetime.today()-pd.to_datetime(df2.author_created_utc,unit='s')).dt.days)/365.


def authenticate(c_id,c_secret,pw):
    reddit = praw.Reddit(
        client_id = c_id,
        client_secret = c_secret,
        password = pw,
        user_agent = "***Your User Agent Here***",
        username = "***Your Username Here***,
        check_for_async = False
        )
    return reddit


def load_srs():
    fname = 'E:/Project_Data/k_project/subreddits.txt'
    sl = np.loadtxt(fname,dtype=str)
    return sl
       
    
    

def convert_timestamp(ts):
    date = datetime.fromtimestamp(ts)
    sdate = date.strftime('%Y-%m-%d')
    return sdate



    

def get_comments(reddit,username):
    '''
    :type reddit: praw instance
    :param reddit: current reddit session
    
    :type username: string
    :param username: name of user to gather comments
    '''
    r = reddit.redditor(username)
    c = r.comments.new(limit=1000)
    
    comments,replies = [],[]
    counter = 0
    for x in c:
        print(counter)
        try:
            temp = x.refresh()
            comments.append(temp.body)
            replies.append(temp.replies)
            counter+=1
            
        except praw.exceptions.ClientException:
            counter+=1
            


            
    return comments,replies


# def get_post_comments(reddit,post_str):
#     post = reddit.submission(post_str)
#     c = post.comments
#     c.replace_more(limit=0)
    
#     cl=[]    
#     for x in c:
#         try:
#             cl.append({
#             'author':x.author.name,
#             'author_created_utc':x.author.created_utc,
#             'body':x.body,
#             'created_utc':x.created_utc,
#             'subreddit':x.subreddit.display_name
#             })
            
#             r = x.replies
#             if len(r)>0:
#                 for y in r:
#                     try:
#                         cl.append({
#                         'author':y.author.name,
#                         'author_created_utc':y.author.created_utc,
#                         'body':y.body,
#                         'created_utc':y.created_utc,
#                         'subreddit':y.subreddit.display_name
#                         })
#                     except AttributeError:
#                         continue
#             else:
#                 pass
        
#         except AttributeError:
#             continue
        
#     return cl

# def get_post_comments(post):
#     c = post.comments
#     c.replace_more(limit=0)
    
#     cc,lb = [],[]
#     for x in c:
#         r = [x.body for x in x.replies]
#         print('{}\n'.format(x.body))
#         score = int(input("Enter the label: "))
#         lb.append(score)
#         cc.append(x.body)
#         if len(r)>0:
#             for y in r:
#                 print('\n{}'.format(y))
#                 score = int(input("Enter the label: "))
#                 lb.append(score)
#                 cc.append(y)
#         else:
#             continue
                
        
#     return cl


def select_users(reddit,post_str,rlimit,n):
    '''
    '''
    post = reddit.submission(post_str)
    c = post.comments
    c.replace_more(limit=rlimit)
    
    counter=0
    ulist = []
    while len(ulist)<n:
        try:
            cdate = datetime.fromtimestamp(c[counter].author.created_utc)
            account_age = (datetime.today()-cdate).days/365.0
            if account_age<5.0 and account_age>0.5:
                ulist.append(c[counter].author.name)
            else:
                pass
            
            rlist = c[counter].replies
            if len(rlist)>0:
                for r in rlist:
                    cdate = datetime.fromtimestamp(r.author.created_utc)
                    account_age = (datetime.today()-cdate).days/365.0
                    if account_age<1.0 and account_age>0.3:
                        ulist.append(r.author.name)
                    else:
                        pass
            else:
                pass
            
            counter+=1
        
        except AttributeError:
            counter+=1
        
        except IndexError:
            return np.unique(ulist)
    
    ulist = np.unique(ulist)
    return ulist
        
    
                

def get_user_comments(reddit,username,c_lim):
    #start = time.time()
    r = reddit.redditor(username)
    c = r.comments.new(limit=c_lim)
    
    cl = []
    counter = 0
    for x in c:
        if counter%10==0:
            print(counter)
        else:
            pass
    
        try:
            cl.append({
            'author':x.author.name,
            'author_created_utc':x.author.created_utc,
            'body':x.body,
            'created_utc':x.created_utc,
            'subreddit':x.subreddit.display_name,
            'post_title':x.link_title,
            'score':x.score
            })
            
            counter+=1
        
        except AttributeError:
            continue
        
        
    #end = time.time()
    #ftime = str(datetime.timedelta(seconds=end-start))
    #print(ftime)
    
    return cl


def save_user_comments(clist,n,data_n):
    '''
    '''
    
    df = pd.DataFrame(clist)
    for x in ['author_created_utc','created_utc']:
        df[x] = df[x].astype(np.int64)
    df.to_json('E:/Project_Data/k_project/data{}/{}.json'.format(data_n,n))
    
    print("Done")
    
    
    
def scrape(reddit,ulist,data_n):
    flist = os.listdir('E:/Project_Data/k_project/data{}'.format(data_n))
    flist = [int(x.split('.')[0]) for x in flist]
    flist = sorted(flist)
    
    for i in range(len(ulist)):
        n = (flist[-1]+1)+i
        cl = get_user_comments(reddit,ulist[i],1000)
        save_user_comments(cl,n,data_n)
        
    print('Done!')
    
    
def load_user_comments(user_number):
    fname = r'E:/Project_Data/k_project/user_comments/{}.json'.format(user_number)
    df = pd.read_json(fname)
    
    return df
    
    
def label_user_comments(df):
    '''
    '''
    print(len(df))
    labels = np.zeros(len(df),dtype=np.int64)
    
    for i in range(len(df)):
        print('\n{}'.format(df.loc[i].body))
        try:
            temp = input("Enter the Label: ")
            if temp=='-':
                temp = '0'
            else:
                pass
            
            temp = int(temp)
            
        except ValueError:
            continue
        
        labels[i] = temp
    
    df['label'] = labels
    
    return df
            
            
    
        
def save_comments(user_number,comments,replies):
    '''
    :type user_number: int
    :param user_number: identifying number for the user
    
    :type comments: list
    :param comments: comment body for each comment instance
    
    :type replies: list
    :param replies: reply instance for each comment instance
    '''
    os.makedirs(r'E:/Project_Data/k_project/user_comments/{}'.format(user_number))
    for i in range(len(comments)):
        os.makedirs(r'E:/Project_Data/k_project/user_comments/{}/{}'.format(user_number,i))
        fname = r'E:/Project_Data/k_project/user_comments/{}/{}/{}.txt'
        for j in range(len(replies[i])+1):
            temp = fname.format(user_number,i,j)
            if j==0:
                data = comments[i]
            else:
                data = replies[i][j-1].body
            try:
                with open(temp,'w') as f:
                    f.write(data)
            except UnicodeEncodeError:
                continue         
        

def load_comments():
    dname = r'E:/Project_Data/k_project/user_comments'
    user_numbers = [str(x) for x in sorted([int(y) for y in os.listdir(dname)])]
    
    data = []
    for n in user_numbers:
        data2 = dict()
        temp_dname = dname+r'/{}'.format(n)
        c_list = [str(x) for x in sorted([int(y) for y in os.listdir(temp_dname)])]
        for n2 in c_list:
            temp_dname2 = temp_dname+r'/{}'.format(n2)
            f_list = os.listdir(temp_dname2)
            fname = temp_dname2+r'/{}'
            
            with open(fname.format(f_list[0]),'r') as f:
                user_comment = f.read()
            
            if len(f_list)>1:
                resps = []
                for i in range(1,len(f_list)):
                    with open(fname.format(f_list[i])) as f:
                        resps.append(f.read())
                    
                data2[n2] = (user_comment,resps)
            
            else:
                data2[n2] = (user_comment,None)
                
        data.append(data2)
        
    return data




def label_comments(data,user_number):
    '''
    
    '''       
    d = data[user_number] #dict[key] = (comment,[replies])
    
    dd = {k:[] for k in ['comment_number','body','parent','reply','score']}
    for k in d.keys():
        c,r = d[k][0],d[k][1]
        
        if c=='':
            continue
        else:
        
            print(c)
            s = input("\nComment Score: ")
            
            if r==None:
                dd['comment_number'].append(int(k))
                dd['body'].append(c)
                dd['parent'].append(0)
                dd['reply'].append(0)
                dd['score'].append(s)
            
            else:
                dd['comment_number'].append(int(k))
                dd['body'].append(c)
                dd['parent'].append(1)
                dd['reply'].append(0)
                dd['score'].append(s)
                
                for x in r:
                    if x=='':
                        continue
                    else:
                        print(x)
                        s = input("\nComment Score: ")
                        dd['comment_number'].append(int(k))
                        dd['body'].append(x)
                        dd['parent'].append(0)
                        dd['reply'].append(1)
                        dd['score'].append(s)
            
    return dd


def load_data(user_number=None):
    dname = r'D:\k_project\user_comments2'
    flist = os.listdir(dname)
    
    if user_number!=None:
        fname = dname+r'\{}.json'.format(user_number)
        with open(fname) as f:
            data = json.load(f)['data']
        
        dfs = pd.DataFrame(data)
    
    else:
        dfs = []
        for x in flist:
            fname = dname+r'\{}'.format(x)
            with open(fname) as f:
                data = json.load(f)['data']
        
            dfs.append(pd.DataFrame(data))
        
    
    return dfs


def load_ndata(n):
    fnames = os.listdir('E:/Project_Data/k_project/data{}'.format(n))
    dfs = []
    for x in fnames:
        fn = 'E:/Project_Data/k_project/data{}/{}'.format(n,x)
        dfs.append(pd.read_json(fn))
    
    with open('E:/Project_Data/k_project/smap2.pkl','rb') as f:
        smap = pickle.load(f)
        
        
    return dfs,smap

    

def label_data2(data_frames,user_number):
    '''
    '''
    df = data_frames[user_number]
    scores = []
    for x in df.body:
        print(x)
        s = input("\nComment Score: ")
        try:
            s = int(s)
        except ValueError:
            s = input("\nEnter It Correctly :) ")
        scores.append(s)
        print('\n\n')
        
    scores = np.asarray(scores)
    df['sentiment'] = s
    fname = 'D:/k_project/labeled_comments/{}.json'.format(user_number)
    
    df.to_json(fname)
    
    print("Done")
    
    
def get_wordnet_tag(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    


def preprocess_data(data,new=False):
    '''
    '''
    c = data.body
    sw = nltk.corpus.stopwords.words('english')
    sw = sw+['.',',',';','^','%','&','*','#','@','~','`','``',"'s","'re",'?','-',"'"]
    
    lemmatizer = WordNetLemmatizer()
    clean_comments = []
    for x in c:
        tokens = nltk.word_tokenize(x.lower())
        if len(tokens)>100:
            pct = int(len(tokens)*0.1)
            tokens = tokens[:pct]+tokens[-pct:]
        tags = nltk.pos_tag(tokens)
        lt = []
        for y in tags:
            w = y[0]
            tag = get_wordnet_tag(y[1])
            
            if w in sw:
                continue
            else:
                if tag==None:
                    lt.append(lemmatizer.lemmatize(w))
                else:
                    lt.append(lemmatizer.lemmatize(w,tag))
        
        clean_comments.append(' '.join(lt))
        
    if new==True:
        return clean_comments
    else:
        lb = data.label.values
        lb[lb>0]=1
        lb[lb<0]=-1
        return clean_comments,lb
    
    
    
def preprocess_titles(data):
    '''
    

    Parameters
    ----------
    data : list
        list of raw strings

    Returns
    -------
    None.

    '''
    sents = []
    for x in data:
        temp = []
        tokens = nltk.word_tokenize(x.lower())
        sw = nltk.corpus.stopwords.words('english')
        sw += list(string.punctuation)
        for word,tag in nltk.pos_tag(tokens):
            if word in sw:
                pass
            else:
                if tag.startswith('N'):
                    temp.append(word)
                else:
                    pass
        
        sents.append(temp)
        
    
    return sents


def title_keyword_extraction(processed_titles):
    '''

    Parameters
    ----------
    processed_titles : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    def dummy_tokenizer(data):
        return data
    
    
    tvec = TfidfVectorizer(
        tokenizer = dummy_tokenizer,
        max_df = 0.85,
        min_df = 2,
        lowercase=False)
    
    tvec.fit(processed_titles)
    tfidf = tvec.transform(processed_titles)
    idx = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
    
    
    fnames = np.asarray(tvec.get_feature_names())[idx]
    
    keys = []
    for i in range(len(processed_titles)):
        data_idx = []
        temp = []
        for word in processed_titles[i]:
            try:
                ix = np.where(fnames==word)[0]
                temp.append(ix)
                
                
            except ValueError:
                print(i)
                
        if len(temp)>0:
            m = temp.index(min(temp))
            keys.append(processed_titles[i][m])
            data_idx.append(i)
        else:
            pass
        
    return keys,fnames
                
    
    
def topic_extraction(df,n_comps=10,n_important=50,bigrams=False):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    cc = preprocess_data(df,new=True)
    if bigrams==False:
        cv = CountVectorizer(
            tokenizer=nltk.word_tokenize,
            max_df = 0.75,
            min_df = 2,
            )
    else:
        cv = CountVectorizer(
            tokenizer=nltk.word_tokenize,
            max_df = 0.75,
            min_df = 2,
            ngram_range=(2,2)
            )
    
    lda = LatentDirichletAllocation(n_components=n_comps)
    
    
    tf = cv.fit_transform(cc)
    lda.fit(tf)
    
    lda_comps = lda.components_ #weights of the feature words
    tf_names = np.asarray(cv.get_feature_names()) #ordered list of words
    
    #Get Ordered Features by Topic
    topic_words = [] #nested list
    weights = []
    for i in range(len(lda_comps)):
        w = lda_comps[i]
        wids = np.argsort(w)
        n_words = tf_names[wids][-n_important:]
        topic_words.append(n_words)
        weights.append(w[wids][-n_important:])
        
        
    return topic_words,weights,lda,tf



def plot_topics(topic_words,weights):
    '''
    

    Parameters
    ----------
    topic_words : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    n_topics = len(topic_words)
    n_rows = 2
    
    if n_topics%2==0:
        n_cols = n_topics//2
    else:
        n_cols = n_topics//2 +1
        
    fig,axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        figsize=(30,15)
        )
    
    axes = axes.flatten()
    
    for i in range(n_topics):
        twords = topic_words[i][::-1]
        tweights = weights[i][::-1]
        
        ax = axes[i]
        ax.barh(twords,tweights,height=0.7)
        ax.set_title("Topic {}".format(i+1))
        ax.invert_yaxis()
        ax.tick_params(axis='both',which='major',labelsize=20)
        
        for j in "top right left".split():
            ax.spines[j].set_visible(False)
            
    plt.show()
    
    
def load_word_vectors():
    fname = r'E:\Project_Data\new_project\glove_50d.txt'
    with open(fname,'r',encoding='latin-1') as f:
        gvec = f.read()
        
    gv = gvec.split('\n')
        
    words = []
    vectors = np.empty(shape=(len(gv),50),dtype=np.float64)
    for i in range(len(gv)-1):
        temp = gv[i].split(' ')
        words.append(temp[0])
        vectors[i] = temp[1:]   
    
    
    return words,vectors


     

def train_model():
    data = pd.read_csv('E:/Project_Data/k_project/training_data.csv')
    cc,lb = preprocess_data(data)
    tvec = TfidfVectorizer(tokenizer=nltk.word_tokenize,max_df=0.5,min_df=2,ngram_range=(1,2))
    
    X = tvec.fit_transform(cc)
    
    model = SVC()
    model.fit(X,lb)
    
    return tvec,model
    


def label_data(dfs,tvec,model):
    labels = []
    for x in dfs:
        cc = preprocess_data(x,True)
        X = tvec.transform(cc)
        prd = model.predict(X)
        labels.append(prd)
        
    
    return labels


def load_label_data(start,end):
    dname = r'E:\Project_Data\k_project'
    flist = np.arange(start,end+1)
    dfs = []
    for x in flist:
        fname = dname+'\{}.json'.format(str(x))
        df = pd.read_json(fname)
        dfs.append(df)
        
    
    tvec,model = train_model()
    lbs = label_data(dfs,tvec,model)
    
    for i in range(len(dfs)):
        dfs[i]['label'] = lbs[i]
        
    return dfs



def get_acct_label(df):
    first = datetime.fromtimestamp(df.iloc[-1].created_utc)
    last = datetime.fromtimestamp(df.iloc[0].created_utc)
    days = (last-first).days
    
    score = df.label.sum()/days
    
    return score
        
  
        
def compare_subreddits(dfs,threshold=0):
    '''
    '''
    dfs = [x for x in dfs if len(x)>=100]
    lbs = np.asarray([x.label.sum() for x in dfs])
    # tr = [(datetime.fromtimestamp(x.iloc[0].created_utc) -
    #        (datetime.fromtimestamp(x.iloc[-1].created_utc))).days for x in dfs]
    
    #lbs = lbs/tr
    
    if threshold!=0:
        pidx = np.where(lbs>threshold)[0]
        nidx = np.where(lbs<-threshold)[0]
    else:
        pidx = np.where(lbs>0)[0]
        nidx = np.where(lbs<0)[0]
    
    pdfs = [dfs[i] for i in pidx] #positive dataframes
    ndfs = [dfs[i] for i in nidx] #negative dataframes
    
    psr = list(itertools.chain.from_iterable([x.subreddit.unique() for x in pdfs]))
    nsr = list(itertools.chain.from_iterable([x.subreddit.unique() for x in ndfs]))
    
    pc = Counter(psr)
    nc = Counter(nsr)
    
    return pc,nc
    


def create_interests(dfs,smap):
    '''
    

    Parameters
    ----------
    dfs : list
        list of dataframes loaded from labeled data
    smap : dict
        dictionary of {subreddit:interest...} pairs

    Returns
    -------
    dfs2 : list
        list of dataframes like dfs with new column "interest"

    '''
    
    dfs2 = []
    for i in range(len(dfs)):
        df = dfs[i]
        sr = df.subreddit.str.lower()
        temp = []
        for x in sr:
            try:
                temp.append(smap[x])
            except KeyError:
                temp.append('N/A')
        
        df['interest'] = temp
        dfs2.append(df)
        
    return dfs2
            
     
        
def load_labeled_data(smap_number=1):
    dname = 'E:/Project_Data/k_project'
    flist = os.listdir(dname+'/labeled_data')
    flist2 = os.listdir(dname+'/labeled_data2')
    
    dfs = []
    for x in flist:
        df = pd.read_json(dname+'/labeled_data/{}'.format(x))
        if len(df)>=100:
            dfs.append(df[::-1])
        else:
            pass
        
    for x in flist2:
        df = pd.read_json(dname+'/labeled_data2/{}'.format(x))
        if len(df)>=100:
            df.drop('score',axis=1,inplace=True)
            dfs.append(df[::-1])
        else:
            pass
        
        
    if smap_number==2:
        sfile = dname+'/smap2.pkl'
    else:
        sfile = dname+'/smap.pkl'
        
    with open(sfile,'rb') as f:
        smap = pickle.load(f)
    
        
    #Add Interest Column
    dfs2 = create_interests(dfs,smap)
               
    return dfs2,smap
    




def interest_matrix(dfs,subreddit_map):
    '''
    Parameters
    ----------
    df : Pandas DataFrame Object
        DataFrame to Create Matrix. Should be concatenated labeled data.
    subreddit_map : dict
        Dictionary of subreddit:interest pairs
    
    Returns
    -------
    2-D Array of Percentage Values

    '''
    vlist = list(set(subreddit_map.values()))
    dlist = []
    for df in dfs:
        vdict = {k:0 for k in vlist}
        
        slist = df.subreddit.str.lower()
        for x in slist:
            try:
                k = subreddit_map[x]
                vdict[k]+=1
            except KeyError:
                pass
        dlist.append(vdict)
        
    a = np.zeros((len(dlist),len(vlist)),dtype=np.float64)
    for i in range(len(dlist)):
        for j in range(len(vlist)):
            k = vlist[j]
            v = dlist[i][k]
            a[i][j] = v
            
    #Get Percentages
    a = a/a.sum(axis=1).reshape(-1,1)
    
    idx = [i for i in range(len(a)) if np.isnan(a[i]).any()==False]
    a = a[idx]
    
    return a,idx



def interest_dists(dfs,label):
    vals = []
    for df in dfs:
        temp = df.loc[df.interest==label]
        if len(temp)>0:
            vals.append(temp.label.sum())
        else:
            pass
    
    vals = np.asarray(vals)
    return vals



def all_interest_dists(dfs,smap):
    ilist = list(set(smap.values()))
    idict = {k:[] for k in ilist}
    
    for df in dfs:
        for i in ilist:
            temp = df.loc[df.interest==i]
            if len(temp)>0:
                v = temp.label.sum()
                idict[i].append(v)
            else:
                pass
    
    return idict



def generate_interest_stats(interest_dict):
    '''
    

    Parameters
    ----------
    interest_dict : dict
        dict of form {interest:[values,...],...}

    Returns
    -------
    stat_arr : numpy array size = (number of interests, number of data points)
    
    *NOTE: scipy.stats.kurtosis is using the Fisher definition which subtracts
    3 from the value (giving 0 for a normal distribution).        

    '''
    
    stat_arr = np.full((len(interest_dict),3),np.nan)  #array filled with NaN values
    idx = list(interest_dict.keys())
    for i in range(len(idx)):
        v = np.asarray(interest_dict[idx[i]])
        stat_arr[i][0] = v.mean()
        stat_arr[i][1] = v.std()
        stat_arr[i][2] = kurtosis(v)
    
    return stat_arr,idx



def train_knn():
    '''
    

    Returns
    -------
    None.

    '''
    with open(r'E:/Project_Data/k_project/topic_map.pkl','rb') as f:
        tmap = pickle.load(f)
    
    words,vecs = load_word_vectors() #loads word embeddings
    
    klist = list(tmap.keys())
    topics = {str(i):klist[i] for i in range(len(klist))}
    
    X,lbs = [],[]
    
    for i in range(len(topics)):
        k = topics[str(i)]
        for w in tmap[k]:
            try:
                emb = vecs[words.index(w)]
                X.append(emb)
                lbs.append(i)
            except ValueError:
                continue
    
    X = np.vstack(X)
    
    #Train KNN
    knn = KNN(n_neighbors=3)
    knn.fit(X,lbs)
    
    
    return words,vecs,topics,knn


def get_post_topics(words,vecs,topics,knn,posts):
    '''
    

    Parameters
    ----------
    words : list
        DESCRIPTION.
    vecs : ndarray
        DESCRIPTION.
    topics : dict
        DESCRIPTION.
    knn : sklearn.neighbors._classification.KNeighborsClassifier
        DESCRIPTION.
    posts : list
        ***Preprocessed titles. Nested list of nouns from post titles.

    Returns
    -------
    None.

    '''
    
    new_posts = []
    
    for x in posts:
        temp = []
        for w in x:
            try:
                v = vecs[words.index(w)]
            except ValueError:
                continue
            
            lb = knn.predict(v.reshape(1,-1))[0]
            temp.append(topics[str(lb)])
        
        if len(temp)==0:
            continue
        
        else:
            new_posts.append(temp)
            
            
    return new_posts
        
            
'''
Plots are using:
    
    pd.DataFrame.plot.bar(...,fontsize=50)
    plt.title(...,fontsize=80)
    plt.legend(fontsize=60)
    
'''

    
    
    
            
        
    
    
        
    
            
            
            
        
    

    


        
    
    
        
    
        
    

    
    

        
        
            
    
            
                
    
    
    
    
    
    
    
            
        
        
    
    
    

