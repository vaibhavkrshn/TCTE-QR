# TCTE-QR


## Introduction

This is the repo of "Question routing via activity-weighted modularity-enhanced factorization". Bibtex citation format:

@article{krishna2022question,
  title={Question routing via activity-weighted modularity-enhanced factorization},
  author={Krishna, Vaibhav and Vasiliauskaite, Vaiva and Antulov-Fantulin, Nino},
  journal={Social Network Analysis and Mining},
  volume={12},
  number={1},
  pages={155},
  year={2022},
  publisher={Springer}
}

## Install Dependencies

Package dependencies are listed in "requirements.txt". You can run

$ pip install -r requirements.txt

to install the packages that are needed.


## Preparing Data

* The archive of the dataset is available [here](https://archive.org/download/stackexchange). Download the dataset and unzip the 7z files into ./Raw_Data/. 

* You will also need Tag synonym dataset which is available [here](https://data.stackexchange.com/superuser/query/new). Download the dataset 'TagSynonyms' and save it as '['name of the SE community']_Tag_Syn.csv' into ./Raw_Data/ 

For example, download for the SuperUser SE community, store the xml files in Raw_Data/SuperUser folder and the Tag synonym dataset as SuperUser_Tag_Syn.csv into Raw_Data folder

* Then run
$ python preprocessing.py

to process the xml files to dataframes and save them into ./PreprocessedData/.

## Run

Execute the Main_Code.py to get the performance on the dataset.
