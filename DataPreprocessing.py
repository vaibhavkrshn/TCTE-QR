

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import xml.etree.ElementTree as ET
import pandas as pd

from tqdm import tqdm



t = input("Enter the CQA folder downloaded from SE archieve (example::askubuntu.com)")

print("StackExchange Community:",t)

file = open('Raw_Data/{}/Posts.xml'.format(t), 'r')
posts = {"posts":[]}

## Parsing XML file
for event, elem in tqdm(ET.iterparse(file)):    

    if event == 'end':
        tag = {}
        if elem.get("Id") is not None:


            post = {}
            post["Id"] = elem.attrib['Id']
            post["PostTypeId"] = elem.attrib['PostTypeId']
            post["CreationDate"] = elem.attrib['CreationDate']
            post["Score"] = elem.attrib['Score']
            post["CommentCount"] = elem.attrib['CommentCount']

            if elem.get("OwnerUserId") is None:
                post["OwnerUserId"] = ""
            else:    
                post["OwnerUserId"] = elem.attrib['OwnerUserId']


            if elem.attrib['PostTypeId'] == '1':

                post["Tags"] = elem.attrib['Tags']
                post["AnswerCount"] = elem.attrib['AnswerCount']
                post["ViewCount"] = elem.attrib['ViewCount']
                if elem.get("AcceptedAnswerId") is not None:
                    post["AcceptedAnswerId"] = elem.attrib['AcceptedAnswerId']

            elif elem.attrib['PostTypeId'] == '2': 

                post["ParentId"] = elem.attrib['ParentId']

            else:
                post["Body"] = elem.attrib['Body']


            posts["posts"].append(post)

            elem.clear()

df_posts = pd.DataFrame(posts["posts"])

df_posts.to_feather('PreprocessedData/{}_Posts.ft'.format(t))  # dataset to save




## Loading the TAG synonym Dataset
TagSyn_dat = pd.read_csv('Raw_Data/{}_Tag_Syn.csv'.format(t))

#### SUBSET QUESTION DATA
########################################
all_ques_data = df_posts.loc[(df_posts['PostTypeId'] == '1')]
all_ques_tags = all_ques_data.filter(items = ['Id','CreationDate','OwnerUserId','Tags','Post_Year'])
all_ques_tags.Tags = all_ques_tags.Tags.apply(lambda x: x.replace('<',''))


### Getting unique TAGs for each question
df_Ques = all_ques_tags.copy() #
df_Ques = df_Ques.drop('Tags', axis=1).join(df_Ques['Tags'].str.split('>', expand=True).stack().reset_index(level=1, drop=True).rename('Tags'))
df_Ques = df_Ques.loc[df_Ques['Tags'] != '']

df_Ques = pd.merge(df_Ques,TagSyn_dat[['SourceTagName','TargetTagName']],how = 'left',
             left_on = 'Tags', right_on = 'SourceTagName')
df_Ques['Uni_Tag'] = np.where(df_Ques['SourceTagName'].isnull(), df_Ques['Tags'], df_Ques['TargetTagName'])
df_Ques = df_Ques.filter(items = ['Id','CreationDate','OwnerUserId','Post_Year','Uni_Tag'])



Tag_dat = df_Ques.drop_duplicates(["Id","Uni_Tag"],keep='first').reset_index(drop=True)
Tag_dat['CreationDate'] =  pd.to_datetime(Tag_dat['CreationDate'], format='%Y-%m-%dT%H:%M:%S.%f')

#### Adding serial number to the TAG used in the Question 
Tag_dat["Tag_Sr_No"] = Tag_dat.groupby("Id")["CreationDate"].rank(method="first", ascending=True)



## Getting TAG-pairs for each question
TagCnt_Pair = pd.merge(Tag_dat,Tag_dat[['Id','Uni_Tag','Tag_Sr_No']], how = 'left',left_on = 'Id',right_on = 'Id')
TagCnt_Pair = TagCnt_Pair.loc[TagCnt_Pair['Tag_Sr_No_x'] < TagCnt_Pair['Tag_Sr_No_y']].reset_index(drop=True)
TagCnt_Pair['Tag-Pair'] = TagCnt_Pair['Uni_Tag_x'] + '::' + TagCnt_Pair['Uni_Tag_y']

arr_slice = TagCnt_Pair[['Tag-Pair']].values
unq,unqtags,counts = np.unique(arr_slice.astype(str),return_inverse=True,return_counts=True)

## Computing the number of questions for each TAG-pair
TagCnt_Pair['Tag-Pair_Qcount'] = counts[unqtags]

TagCnt_Pair.to_feather('PreprocessedData/{}_TagCnt_Pair.ft'.format(t))




