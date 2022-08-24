# bot_detector
This repository contains tests made for checking the effect of a few types of clustering pre-processing for bot detection on Twitter. It's part of an Msc program at USP (Universidade de São Paulo)

Initial data used for training comes from datasets available at https://botometer.iuni.iu.edu/bot-repository/datasets.html

Most datasets have only the label and will need to have their data retrieved through the Twitter API. 

`tidy_data.py` will transform the datasets as available from the repository to a usable file.

`read_from_twitter.py` builds the data from Twitter.

`tidy_cresci.py` and `tidy_users.py` are then applied to the data with features.

`run_experiment.ipynb` contains the code to run the experiments, while the `results_10pct_` files have the code to run all results.


Datasets that have their data retrieved through the Twitter API:
* "botometer-feedback-2019.tsv"
* "botwiki-2019.tsv"
* "celebrity-2019.tsv"
* "cresci-rtbust-2019.tsv"
* "cresci-stock-2018.tsv"
* "gilani-2017.tsv"
* "midterm-2018.tsv"
* "pronbots.tsv"
* "varol-2017.dat"
* "vendor-purchased-2019.tsv"

Datasets that already included their relevant variables: 

* "cresci-2015"

    TFP (the fake project): 100% humans
    E13 (elections 2013): 100% humans
    INT (intertwitter): 100% fake followers
    FSF (fastfollowerz): 100% fake followers
    TWT (twittertechnology): 100% fake followers
    
* "cresci-2017"

 'fake_followers.csv', 'genuine_accounts.csv',
       'social_spambots_1.csv', 'social_spambots_2.csv',
       'social_spambots_3.csv', 'traditional_spambots_1.csv',
       'traditional_spambots_2.csv', 'traditional_spambots_3.csv',
       'traditional_spambots_4.csv'
