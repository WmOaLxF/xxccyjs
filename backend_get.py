from flask import Flask, redirect, url_for, request, render_template
import pkuseg
import os 
import numpy as np 
import re 
from functools import reduce
seg = pkuseg.pkuseg()           # 加载web模型
stop_dir='./stopwords/' #链接: https://pan.baidu.com/s/1lkpBp8JDTN14-Gnczi6_nA  密码: u21o--来自百度网盘超级会员V3的分享
import pandas as pd
from pathlib import Path
from pyecharts.charts import *
from pyecharts import options as opts
from gensim.models import Word2Vec,word2vec
from functools import reduce

DATA_STORE=Path('data.h5') 
word2vec_path='./word2vec/model/'

with pd.HDFStore(DATA_STORE) as store:
    url_=store['course/video/url']
    dp=store['course/text/daopai']  
    meta=store['course/meta']  
    tf_idf_df=store['course/tfidf']
    tf_idf_corr=store['course/tfidf_corr']

def get_vid_url(find_url):
    num=int(re.findall('\d+',find_url)[0])
    return url_.iloc[num].values[0]
def get_vid_num(find_url):
    num=int(re.findall('\d+',find_url)[0])
    return num 
def get_find_url(num):
    if num<10:
        num='url%d'%num
    else:
        num=str(num)
    return num

#加载停用词
def read_f(file):
    if '.txt' not in file: 
        return []
    with open(file) as f: 
        stop_l=[i.strip('\n') for i in f.readlines()]
    return stop_l

# stop_l=reduce(lambda x,y: x+y,map(read_f,[stop_dir+t for t in os.listdir(stop_dir)]))

def my_cut(sentence):
#    return [i for i in seg.cut(sentence) if i not in stop_l]
    return [i for i in seg.cut(sentence)]

idx = pd.IndexSlice
num=1
def get_by_index(num,only_all=False,only_time=False):
    if num<10:
        num='url%d'%num
    else:
        num=str(num)
    with pd.HDFStore(DATA_STORE) as store:
        if only_all:
            all_df = (store['course/text/all/all_%s'%num])
            return all_df 
        if only_time:
            time_df= (store['course/text/detail/detail_%s'%num])
            return time_df 
        else:
            all_df = (store['course/text/all/all_%s'%num])
            time_df= (store['course/text/detail/detail_%s'%num])
            return all_df,time_df 

def get_meta(url_idx):
    return meta.loc[url_idx].tolist()  

def query_sort(input_):
    tokens=my_cut(input_)
    res_sr=reduce(lambda x,y:x+y,map(lambda x:tf_idf_df[x] if x in tf_idf_df.columns else 0,tokens))
    try:
        state=True
        i=3
        while state:
            if res_sr.sort_values(ascending=False).iloc[i-1]!=0:
                state=False
                return tokens,res_sr.sort_values(ascending=False).iloc[:i].index.tolist()#url_index
            else:
                i-=1
    except KeyError:#不在其中
        return tokens,'未找到关键词'          
'''
前端
'''


app = Flask(__name__)

@app.route('/index',methods=['POST','GET'])
def index():
    if request.method=='POST':
        kw=request.form['search']
        if False:
            if kw=='信息存储与检索':
                num=0
                all_df,time_df=get_by_index(num)
                vid_url=get_vid_url(num)
            #return all_df
            #return redirect(vid_url)
        else: 
            
            tokens,contain_=query_sort(kw)
            vid_store=[]
            if type(contain_)==str:#未找到
                vid_store.append((tokens,False,False))
            else:
                vid_store.append((tokens,
                                list(map(get_meta,contain_)),
                                list(map(get_vid_num,contain_))
                                ))
                
            return render_template('play_index.html',
                           vid_store=vid_store
                           )
        #return redirect(url_for('temp'),kw=kw)

@app.route('/video/<int:url>')
def video(url):
    if url<10:
        url_str='url'+str(url)
    else:
        url_str=str(url)
    #w2v_model=Word2Vec.load(word2vec_path+'%s.model'%url_str)
    tf_idf_corr_recommend=tf_idf_corr.loc[url_str].sort_values(ascending=False)[1:4]
    recommand_meta=list(map(get_meta,tf_idf_corr_recommend.index.tolist()))
    recommand_url=list(map(get_vid_num,tf_idf_corr_recommend.index.tolist()))
    recommand_value=tf_idf_corr_recommend.values.tolist()
    return render_template('bofang.html',
    url_=url_,
    url=url,
    recommand_meta=recommand_meta,
    recommand_value=recommand_value,
    recommand_url=recommand_url,
    times_=[0],
    time_dict={}
    )

@app.route('/pro',methods=['POST','GET'])
def pro():
    if request.method=='POST':
        kw=request.form['search']
        url=request.form['url']
        time_df=get_by_index(int(url),only_all=False,only_time=True)
        tokens=my_cut(kw)
        if int(url)<10:
            url_str='url'+str(url)
        else:
            url_str=str(url)
        w2v_model=Word2Vec.load(word2vec_path+'%s.model'%url_str)    
        tf_idf_corr_recommend=tf_idf_corr.loc[url_str].sort_values(ascending=False)[1:4]
        recommand_meta=list(map(get_meta,tf_idf_corr_recommend.index.tolist()))
        recommand_value=tf_idf_corr_recommend.values.tolist()
        recommand_url=list(map(get_vid_num,tf_idf_corr_recommend.index.tolist()))
        time_dict={}
        similar=[]
        similar_to_plot=[]
        for token in tokens:
            #return str(time_df)
            bgtimes=(time_df[ time_df.res.apply(lambda x:True if token in x else False) ].begin_time.values/1000).tolist()
            
            if len(bgtimes):
                time_dict[token]=bgtimes
                try:
                    temp_similar=w2v_model.wv.similar_by_word(token,topn=100)
                    similar+=temp_similar
                except KeyError:
                    similar+=[]
                
            else:
                time_dict[token]=[]
                
        for t in time_dict.values():
            if not t:
                bgtimes=[0]
            else:
                bgtimes=t
        for w in similar:
            w=list(w)
            for nw in similar:
                if w[0]==nw[0]:
                    value=np.mean([w[1],nw[1]])
                    w[1]=value
            similar_to_plot.append(w)
        wc=WordCloud().add('相关词云',similar_to_plot)
        wc.render('./static/temp_wordcloud.html')
        return render_template('bofang.html',
                    url=int(url),
                    url_=url_,
                    recommand_meta=recommand_meta,
                    recommand_value=recommand_value,
                    recommand_url=recommand_url,
                    times_=bgtimes,
                    time_dict=time_dict                
                    )


#@app.route('/index/<int:postID>')
#def show_blog(postID):
#   return 'Blog Number %d' % postID





if __name__ == '__main__':
   app.run(debug=True)
