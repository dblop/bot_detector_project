# bot_detector
This is my personal project for a bot detector on Twitter.

Initial data used for training comes from datasets available at https://botometer.iuni.iu.edu/bot-repository/datasets.html

Most datasets have only the label and will need to have their data retrieved through the Twitter API. 

`tidy_data.py` will transform the datasets as available from the repository to a usable file.

`read_from_twitter.py` builds the data from Twitter.



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

* "verified-2019.tsv" -> STILL NEED TO GET THIS DATA! Only humans though
* "caverlee-2011" -> STILL NEED TO GET THIS DATA! content_polluters.txt and legitimate_users.txt, first column is user_id


Datasets that already included their relevant variables: 

* "cresci-2015"

    TFP (the fake project): 100% humans
    E13 (elections 2013): 100% humans
    INT (intertwitter): 100% fake followers
    FSF (fastfollowerz): 100% fake followers
    TWT (twittertechnology): 100% fake followers
    
* "cresci-2017"

