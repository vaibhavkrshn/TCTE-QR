

## Loading library

import numpy as np
import pandas as pd
import tqdm

from cdlib import algorithms as al, ensemble as en, evaluation as ev, viz 
import networkx as nx




## Defining the deporal descounting function

def temporal_dis(df,dis_cond):

    TempDis_df = df.copy()
    
    ### Hyperbolic discounting
    if (dis_cond == 'hypb_dis'):
        TempDis_df['Ans_Cnt_Wt'] = 1/(1+TempDis_df['Months_Post'])
        TempDis_df['Ans_Scr_Wt'] = TempDis_df['A_Scr']/(1+TempDis_df['Months_Post'])
        
    ### exponential discounting
    elif (dis_cond == 'exp_dis'):
        TempDis_df['Ans_Cnt_Wt'] = 1/(np.exp(TempDis_df['Months_Post']))
        TempDis_df['Ans_Scr_Wt'] = TempDis_df['A_Scr']/(np.exp(TempDis_df['Months_Post']))

    return TempDis_df


def month_diff(a, b):
    return 12 * (a.dt.year - b.dt.year) + (a.dt.month - b.dt.month)



## defing the training dataset
def train_dat(year1,year2,post_df):
    AAAI_df = post_df[(post_df['CreationDate'] > year1) & (post_df['CreationDate'] < year2)].reset_index(drop=True)

    arr_slice = AAAI_df[['OwnerUserId']].values
    unq,unqtags,counts = np.unique(arr_slice.astype(str),return_inverse=True,return_counts=True)
    AAAI_df['Ttl_Posts'] = counts[unqtags]
    AAAI_df['A_Scr'] = AAAI_df.Score.astype(str).astype(int)

    AAAI_df = AAAI_df[AAAI_df['Ttl_Posts'] > 4].reset_index(drop=True)
    
    return AAAI_df


#### Defining TAG graph
def Tag_Graph(TagPair_df,Date1,Date2,t_sup):
    
    df = TagPair_df.loc[(TagPair_df['CreationDate'] >  Date1) &
                         (TagPair_df['CreationDate'] <  Date2)]
    
    df = df.reset_index(drop=True)
    
    arr_slice = df[['Tag-Pair']].values
    unq,unqtags,counts = np.unique(arr_slice.astype(str),return_inverse=True,return_counts=True)
    df['Tag-Pair_Qcnt'] = counts[unqtags]

    ##### Keeping the TAG-pairs where number of questions are above a threshold
    df = df.loc[(df['Tag-Pair_Qcnt'] >= t_sup)]
    df = df.reset_index(drop=True)

    Ttl_num_ques = df.Id.nunique()
    Sample_num_ques = df.Id.nunique()
    
    
    graph_df = df[['Uni_Tag_x','Uni_Tag_y','Tag-Pair_Qcnt']].drop_duplicates(keep='first')
    graph_df.columns = ['source', 'target','weight']
    g = nx.Graph()
    #defining the weighted network
    g.add_weighted_edges_from(graph_df[['source', 'target', 'weight']].values)
    
    
    return g,df



def comm_map(g,coms):

    degree_cen = nx.degree_centrality(g)
    btw_cen = nx.betweenness_centrality(g)
    degr  = pd.DataFrame(nx.degree(g))

    # mapping the Tags with identified topics
    com_map=pd.DataFrame(coms.to_node_community_map()).T
    com_map = com_map.rename_axis('User_Id').reset_index()
    com_map.columns = ['Tag', 'Comm']

    
    degr.columns = ['Tag', 'Deg']
    d_cen = pd.DataFrame(degree_cen.items(), columns=['Tag', 'DegCentrality'])
    b_cen = pd.DataFrame(btw_cen.items(), columns=['Tag', 'BtwCentrality'])

    com_map = pd.merge(com_map,degr, how = 'left',left_on = 'Tag',right_on = 'Tag')
    com_map = pd.merge(com_map,d_cen, how = 'left',left_on = 'Tag',right_on = 'Tag')
    com_map = pd.merge(com_map,b_cen, how = 'left',left_on = 'Tag',right_on = 'Tag') 

    com_map = com_map.sort_values(['Comm', 'Deg','BtwCentrality'], ascending=[True, False, False])

    
    df_com = com_map.copy()

    arr_slice = df_com[['Comm']].values
    unq,unqtags,counts = np.unique(arr_slice.astype(str),return_inverse=True,return_counts=True)
    df_com["Ttl_Tags"] = counts[unqtags]

    # getting the final Topics
    df_com["Sr_No"] = df_com.groupby("Comm")["Deg"].rank(method="first", ascending=False)
    df_com = df_com.loc[df_com['Sr_No'] <2].reset_index()

    
    return com_map,df_com



def commTAG_df(grph_df,train_dat,com_map,com_df):
    
    graph_df_x = grph_df[['Id','Uni_Tag_x']].drop_duplicates()
    graph_df_x.columns = ['Q_Id','Tags']
    graph_df_y = grph_df[['Id','Uni_Tag_y']].drop_duplicates()
    graph_df_y.columns = ['Q_Id','Tags']

    TAG_Qdf = graph_df_x.append(graph_df_y).drop_duplicates().reset_index(drop=True)

    arr_slice = TAG_Qdf[['Q_Id']].values
    unq,unqtags,counts = np.unique(arr_slice.astype(str),return_inverse=True,return_counts=True)
    TAG_Qdf['N_Tags'] = counts[unqtags]



    comm_TAG_df = pd.merge(TAG_Qdf,com_map.iloc[:,[0,1]],how = 'left', left_on='Tags',right_on = 'Tag')
    comm_TAG_df = comm_TAG_df.drop('Tag', 1)
    comm_TAG_df = pd.merge(comm_TAG_df,com_df.iloc[:,[1,2]],how = 'left', left_on='Comm',right_on = 'Comm')
    
    comm_Q_df = comm_TAG_df.filter(items=['Q_Id','N_Tags','Tag'])
    comm_Q_df.columns = ['Q_Id','N_Tags','Tags']
    commTag_Q_df = comm_Q_df.value_counts(['Q_Id','N_Tags','Tags']).reset_index(name='N_CTag')
    
    
    TAG_Adf = train_dat[(train_dat['PostTypeId'] == '2') & (train_dat['A_Scr'] > 0)].reset_index(drop=True)
    TAG_Adf = TAG_Adf.filter(items=['Id', 'CreationDate','A_Scr','OwnerUserId','ParentId'])    
    
    TAG_QnAdf = pd.merge(TAG_Adf,commTag_Q_df,how = 'left',left_on='ParentId',right_on='Q_Id')
    TAG_QnAdf = TAG_QnAdf[-TAG_QnAdf.Q_Id.isnull()]
    
    TAG_QnAdf['wt_Cnt'] = TAG_QnAdf['N_CTag']/TAG_QnAdf['N_Tags']
    TAG_QnAdf['wt_A_Scr'] = TAG_QnAdf['A_Scr']*TAG_QnAdf['wt_Cnt'] 
    
    return TAG_QnAdf



### Returns User activity - temporal discounted on each topic

def comm_Udf(Tag_df,present_dt,temp_dis_type):
    
    TAG_Udf = Tag_df.copy()
    TAG_Udf['U_Tag'] = TAG_Udf['OwnerUserId'] + TAG_Udf['Tags']
    
    arr_slice = TAG_Udf[['U_Tag']].values
    unq,unqtags,counts = np.unique(arr_slice.astype(str),return_inverse=True,return_counts=True)
    TAG_Udf['Ttl_Ans'] = counts[unqtags]   
    TAG_Udf['wt_Ttl_Ans'] = TAG_Udf['wt_Cnt'].groupby(TAG_Udf['U_Tag']).transform('sum')
    
    
    TAG_Udf['ref_date'] = pd.Timestamp(present_dt)
    TAG_Udf['Months_Post'] = month_diff(TAG_Udf['ref_date'], TAG_Udf['CreationDate'])
    TAG_Udf['Days_Ans'] = (TAG_Udf['ref_date'] -  TAG_Udf['CreationDate']).dt.days#.astype('timedelta64[h]')

    TAG_U_TempD_df = temporal_dis(TAG_Udf,temp_dis_type)

    TAG_U_TempD_df['AC_Wt'] = TAG_U_TempD_df['Ans_Cnt_Wt'].groupby(TAG_U_TempD_df['U_Tag']).transform('sum')
    TAG_U_TempD_df['wt_Ans_Cnt_Wt'] = TAG_U_TempD_df['wt_Cnt']/(1+TAG_U_TempD_df['Months_Post'])
    
    ## Temporal weighted score for each user on the Topic
    TAG_U_TempD_df['wt_AC_Wt'] = TAG_U_TempD_df['wt_Ans_Cnt_Wt'].groupby(TAG_U_TempD_df['U_Tag']).transform('sum')

    
    TAG_df = TAG_U_TempD_df.filter(items=['OwnerUserId','Tags','Ttl_Ans','AC_Wt',
                                  'wt_Ttl_Ans','wt_AC_Wt']).drop_duplicates()

    
    return TAG_df

    

from surprise import SVD
from surprise import NMF
from surprise import accuracy
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import Reader#, evaluate#, print_perf



def NMF_Model(train_df,criteria,nfactors):
    
    TAG_df = train_df.filter(items=['OwnerUserId','Tags',criteria])
    TAG_df.columns = ['OwnerUserId','Tags','Rating']
    x, y = TAG_df.Rating.min(), TAG_df.Rating.max()
    
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(x, y))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(TAG_df[['OwnerUserId','Tags','Rating']], reader)

    algo = NMF(n_factors =nfactors, reg_pu=0.01,reg_qi=0.01)

    # retrain on the whole set 
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    TAG_df['AU_Id'] = TAG_df['OwnerUserId'].apply(lambda x: trainset.to_inner_uid(x))
    TAG_df['Tag_Id'] = TAG_df['Tags'].apply(lambda x: trainset.to_inner_iid(x))

    prediction = pd.DataFrame(np.dot(algo.pu, algo.qi.T))
    
    return TAG_df,trainset,prediction



def comm_testDf(start,end,train_df,com_map,df_com,post_dat):
    
    
    test_df = post_dat[(post_dat['CreationDate'] > start) & (post_dat['CreationDate'] < end)].reset_index(drop=True)
    
    testQdf = test_df[test_df['PostTypeId'] == '1'].reset_index(drop=True)
    testQdf = testQdf[testQdf['AcceptedAnswerId'].notnull()]
    testQdf = testQdf.filter(items=['Id','Tags','AcceptedAnswerId'])
    
    testQdf['N_Tags'] = testQdf.Tags.str.count("<")
    testQdf.Tags = testQdf.Tags.apply(lambda x: x.replace('<',''))
    testQdf = testQdf.drop('Tags', axis=1).join(testQdf['Tags'].str.split('>', expand=True).stack().reset_index(level=1, drop=True).rename('Tags'))
    testQdf = testQdf.loc[testQdf['Tags'] != '']
    
    comm_test_df = pd.merge(testQdf,com_map.iloc[:,[0,1]],how = 'left', left_on='Tags',right_on = 'Tag')
    comm_test_df = comm_test_df.drop('Tag', 1)
    comm_test_df = pd.merge(comm_test_df,df_com.iloc[:,[1,2]],how = 'left', left_on='Comm',right_on = 'Comm')
    comm_test_df = comm_test_df[comm_test_df['Tag'].notnull()]
    comm_test_df = comm_test_df[comm_test_df.Tags.isin(train_df.Tags)]
    comm_test_df = comm_test_df.value_counts(['Id','AcceptedAnswerId','Tag','N_Tags']).reset_index(name='N_CTag')
    comm_test_df.columns = ['Id','AcceptedAnswerId','Tags','N_Tags','N_CTag']
    comm_test_df['CTag_wt'] = comm_test_df['N_CTag']/comm_test_df['N_Tags']
    
    testAdf = test_df[test_df['PostTypeId'] == '2'].reset_index(drop=True)
    testAAdf = testAdf[testAdf['Id'].isin(testQdf.AcceptedAnswerId)]
    testAAdf = testAAdf[testAAdf['OwnerUserId'] != '']
    testAAdf = testAAdf[testAAdf['OwnerUserId'].isin(train_df.OwnerUserId)]
    testAAdf = testAAdf.filter(items=['Id','OwnerUserId','ParentId'])
    testAAdf.columns = ['A_Id','OwnerUserId','ParentId']
    
    testAAdf_U = pd.merge(comm_test_df,testAAdf,how='left',left_on='AcceptedAnswerId',right_on='A_Id')   
    testAAdf_U = testAAdf_U[testAAdf_U.A_Id.notnull()]
     
    return testAAdf_U



##### 

def commMF_eval_MatrixApp(test_df,train_df,trainset,prediction):
    
    tag_df = test_df.copy()    
    tag_df['Tags_Id'] = tag_df['Tags'].apply(lambda x: trainset.to_inner_iid(x))
    tag_df['AU_Id'] = tag_df['OwnerUserId'].apply(lambda x: trainset.to_inner_uid(x))
    tag_df['AU_Ind'] = 1


    pivt_Usr_tag_df = tag_df[['Id','AU_Id','AU_Ind']]
    
    all_Users = pd.DataFrame(prediction.index.values)
    all_Users['Id'] = -9
    all_Users['AU_Ind'] = 0
    all_Users.columns = ['AU_Id','Id','AU_Ind']
    
    pivt_allUsr_tag_df = pivt_Usr_tag_df.append(all_Users)

    Usr_Ind = pd.pivot_table(pivt_allUsr_tag_df,values = 'AU_Ind', index=['Id'], columns = 'AU_Id',fill_value=0)
    Usr_Ind = Usr_Ind.iloc[1:,:]

    
    pivt_Topic_df = tag_df[['Id','Tags_Id','CTag_wt']]
    
    all_Topics = pd.DataFrame(prediction.columns.values)
    all_Topics['Id'] = -9
    all_Topics['CTag_wt'] = 0
    all_Topics.columns = ['Tags_Id','Id','CTag_wt']

    pivt_allTopic_df = pivt_Topic_df.append(all_Topics)


    Q_Topic_mat = pd.pivot_table(pivt_allTopic_df, values = 'CTag_wt', index=['Id'], columns = 'Tags_Id').fillna(0)
    Q_Topic_mat = Q_Topic_mat.iloc[1:,:]
    Q_U_mat = Q_Topic_mat.dot(prediction.T)
    Q_U_rank_mat = Q_U_mat.rank(axis = 1, ascending = False).astype(int)

    
    Q_U_rank = Q_U_rank_mat.mul(Usr_Ind)
    Q_U_rank["Rank"] = Q_U_rank.sum(axis=1)

    Q_U_rank['wt_RR'] = 1/Q_U_rank['Rank']
    Q_U_rank['wt_Prec_at_5'] = np.where(Q_U_rank['Rank'] < 6, 1, 0) #### Change for Precision measure at any n
    Q_U_rank['wt_Prec_at_10'] = np.where(Q_U_rank['Rank'] < 11, 1, 0) #### Change for Precision measure at any n

    wt_MRR = round(Q_U_rank['wt_RR'].mean(),4)
    wt_Prec = round(Q_U_rank['wt_Prec_at_5'].mean(),4)
    wt_HIT = round(Q_U_rank['wt_Prec_at_10'].mean(),4)
    
    
    return wt_MRR, wt_Prec, wt_HIT



#### MAIN 

t = input("Enter the CQA folder downloaded from SE archieve (example-askubuntu.com)")

print("StackExchange Community:",t)

### Getting the post history data
post_dat = pd.read_feather('PreprocessedData/{}_Posts.ft'.format(t))
post_dat['CreationDate'] =  pd.to_datetime(post_dat['CreationDate'], format='%Y-%m-%dT%H:%M:%S.%f')
post_dat['Post_Year'] = post_dat['CreationDate'].dt.year

### Getting the TAG graph data
TagCnt_Pair = pd.read_feather('PreprocessedData/{}_TagCnt_Pair.ft'.format(t))   
TagCnt_Pair['CreationDate'] =  pd.to_datetime(TagCnt_Pair['CreationDate'], format='%Y-%m-%dT%H:%M:%S.%f')


##### GRAPH TIME PERIOD
### enter the graph start date
G_date1 = '2015-01-01'

start_date = input("Enter start date (example-'2015-01-01')")
end_date = input("Enter end date (example-'2019-05-01')")

q_dat = post_dat[-post_dat.AcceptedAnswerId.isna()].reset_index(drop=True)
q_dat = q_dat[(q_dat['CreationDate'] > start_date) & (q_dat['CreationDate'] < end_date)].reset_index(drop=True)
q_dat = q_dat[['Id','CreationDate','OwnerUserId','Tags','AcceptedAnswerId']]
q_dat.columns= ['Q_Id','Q_Date','Q_UserId','Tags','AcceptedAnswerId']
### date split - Train, Test data
date_split = q_dat.Q_Date.quantile(0.8)
print('date split - Train, Test data:',date_split)

### enter the temporal discouting type
temp_dis_typ = 'hypb_dis'

train_df = train_dat(start_date,date_split,post_dat)

##### Minimum edge weight for defining TAG graph
t_sup = 5

g,graph_df = Tag_Graph(TagCnt_Pair,G_date1,date_split,t_sup)
coms = al.louvain(g)
CD_algo = 'louvain'
com_map_dat,dat_com = comm_map(g,coms)


# TOPIC MODEL DATA
comm_model_df = commTAG_df(graph_df,train_df,com_map_dat,dat_com)
comm_model_Udf = comm_Udf(comm_model_df,date_split,temp_dis_typ)

TopCommMF_n_users = comm_model_Udf.OwnerUserId.nunique()
TopCommMF_n_tags = comm_model_Udf.Tags.nunique()
TopCommMF_n_ratings = comm_model_Udf.shape[0]



nfac = 10 ## NUMBER OF LATENT FACTORS for Matrix Factorisation

#### CRITERIA : 
################ 'wt_Ttl_Ans', 'wt_AC_Wt'

temporal_ind = input("Input 1 for temporal discounting and 0 for ranking without temporal discounts")
if temporal_ind == 1:
    crt = 'wt_AC_Wt'
else:
    crt = 'wt_Ttl_Ans'

##### Matrix factorisation 
comm_Model_Tag_df,comm_trainset,comm_prediction = NMF_Model(comm_model_Udf,crt,nfac)

### Test data and evaluation
comm_eval_df = comm_testDf(date_split,end_date,comm_Model_Tag_df,com_map_dat,dat_com,post_dat)
Topic_MRR, Topic_Prec5, Topic_Prec10 = commMF_eval_MatrixApp(comm_eval_df,comm_Model_Tag_df,comm_trainset,comm_prediction)

print('MRR:',Topic_MRR,
     'Prec@5:',Topic_Prec5,
     'Prec@10:',Topic_Prec10)




