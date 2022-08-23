import tweepy
import time
import pandas as pd
import os.path
import datetime
from tweepy.error import RateLimitError, TweepError

consumer_key = ""
consumer_secret = ""

access_token = ""
access_secret = ""

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)


api = tweepy.API(auth)



param_list = ["dataset","label","profile_banner_url","id","id_str","name","screen_name","location","profile_location","description","url","protected","followers_count","friends_count","listed_count","favourites_count","utc_offset","time_zone","geo_enabled","verified","statuses_count","lang","contributors_enabled","is_translator","is_translation_enabled","profile_background_color","profile_background_image_url","profile_background_image_url_https","profile_background_tile","profile_image_url","profile_image_url_https","profile_link_color","profile_sidebar_border_color","profile_sidebar_fill_color","profile_text_color","profile_use_background_image","has_extended_profile","default_profile","default_profile_image","following","follow_request_sent","notifications","translator_type","created_at"]


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
        #Initialize everything as null. API isn't consistent about null values and might not return the param otherwise.
        for param in param_list:
            dict_data[param] = None

        dict_data["dataset"] = dataset
        dict_data["label"] = label

        try:
            user = api.get_user(user_id=user_id)
        except RateLimitError:
            dict_data["id_str"] = user_id
            time.sleep(60)
            df_a = pd.DataFrame(dict_data, index=[0])
            df_a.to_csv("retry_later_users.csv",mode="a",index=False,header=False)
            print(f"Iteration {count} hit a RateLimitError")
            continue            
        except TweepError as e:
            dict_data["id_str"] = user_id
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
        
        params = list(user.__dict__.keys())

        for param in params:
            if param in param_list:
                dict_data[param] = getattr(user,param)

        df_a = pd.DataFrame(dict_data, index=[0])


        df_a.to_csv("users.csv",mode="a",index=False,header=False)
        
        stop = time.time()
        dif = stop-start
        print(f"Iteration {count} took {dif} seconds")
