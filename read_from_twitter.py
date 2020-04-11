import tweepy
import time
import pandas as pd
import os.path
import datetime
from tweepy.error import RateLimitError, TweepError

consumer_key = "7ud8FIZYfbNeTeXZhemTwl08T"
consumer_secret = "aKsSRvFOiUdPjHbL2sKcwF55WBTWb4GAvM5BSFH6stvLyamyrJ"

access_token = "1180902059077521408-z01YFValEgCwAq6xsIgVgUt4daprM9"
access_secret = "s6NDmZfODm14R7ZUfGQ1lqkzD5idAELeBMwfsERQamcuf"

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)


api = tweepy.API(auth)

count = 0 
with open("data.txt") as f:
    for line in f:
        count += 1
        start = time.time()

        data = line.split(",")
        dataset = data[0]
        user_id = data[1]
        label = data[2].strip('\n')
        
        dict_data = dict()
        dict_data["dataset"] = dataset
        dict_data["label"] = label

        try:
            user = api.get_user(user_id=user_id)
        except RateLimitError:
            dict_data["user_id"] = user_id
            time.sleep(60)
            df_a = pd.DataFrame(dict_data, index=[0])
            df_a.to_csv("retry_later_users.csv",mode="a",index=False,header=False)
            print(f"Iteration {count} hit a RateLimitError")
            continue            
        except TweepError as e:
            dict_data["user_id"] = user_id
            # error codes and messages:
            # https://developer.twitter.com/en/docs/basics/response-codes
            # if there's any non-Twitter error we don't get the message!
            try:
                dict_data["error"] = e.response.text
            except:
                dict_data["error"] = "Error without message"
            df_a = pd.DataFrame(dict_data, index=[0])
            df_a.to_csv("suspended_users.csv",mode="a",index=False,header=False)
            continue
        
        #
        params = list(user.__dict__.keys())
        params.remove("_api")
        params.remove("_json")
        params.remove("status")
        params.remove("entities")
        params.remove("created_at")
        
        #API doesnt return the null profile_banner_url if there's none
        dict_data["profile_banner_url"] = None
        for param in params:
            dict_data[param] = getattr(user,param)

        df_a = pd.DataFrame(dict_data, index=[0])


        df_a.to_csv("users.csv",mode="a",index=False,header=False)
        
        stop = time.time()
        dif = stop-start
        print(f"Iteration {count} took {dif} seconds")
