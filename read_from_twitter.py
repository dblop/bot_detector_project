import tweepy
import time
consumer_key = "7ud8FIZYfbNeTeXZhemTwl08T"
consumer_secret = "aKsSRvFOiUdPjHbL2sKcwF55WBTWb4GAvM5BSFH6stvLyamyrJ"

access_token = "1180902059077521408-z01YFValEgCwAq6xsIgVgUt4daprM9"
access_secret = "s6NDmZfODm14R7ZUfGQ1lqkzD5idAELeBMwfsERQamcuf"

#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)


#try:
#    redirect_url = auth.get_authorization_url()
#except tweepy.TweepError:
#    print('Error! Failed to get request token.')

# Example w/o callback (desktop)proba
#verifier = raw_input('Verifier:')


api = tweepy.API(auth)

#start = time.time()
user = api.get_user('alvarodias_')

import pandas as pd

dict_data = {'statuses_count': [user.statuses_count], 'followers_count': [user.followers_count], 'friends_count': [user.friends_count], 'favourites_count': [user.favourites_count], 'default_profile': [user.default_profile] , 'geo_enabled': [user.geo_enabled] , 'profile_use_background_image': [user.profile_use_background_image] , 'verified': [user.verified] , 'protected': [user.protected]  }
df_a = pd.DataFrame(dict_data)
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

rf = load('randomforest.joblib') 
result = rf.predict_proba(df_a)
print(result)
print(result[0][1])
#stuff = api.user_timeline(screen_name = 'realDonaldTrump', count = 100, include_rts = True)

#stop = time.time()
#dif = stop-start


#public_tweets = api.home_timeline()
#for tweet in public_tweets:
#    print(tweet.text)


