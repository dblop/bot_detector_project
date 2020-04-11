import pandas as pd
import numpy as np



bot_accounts = pd.concat([pd.read_csv('data/social_spambots_1.csv/users.csv'), pd.read_csv('data/social_spambots_2.csv/users.csv'), pd.read_csv('data/social_spambots_3.csv/users.csv')],sort=True)
clean_accounts = pd.read_csv('data/genuine_accounts.csv/users.csv')

requiredColumns = ['id','screen_name', 'statuses_count', 'followers_count', 'friends_count','favourites_count',
                   'default_profile','geo_enabled','profile_use_background_image','verified', 'protected']
bot_accounts = bot_accounts[requiredColumns]
clean_accounts = clean_accounts[requiredColumns]


bot_accounts = bot_accounts.fillna(0)
clean_accounts = clean_accounts.fillna(0)


bot_accounts["label"] = 1
clean_accounts["label"] = 0
full_dataset = pd.concat([bot_accounts, clean_accounts])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(full_dataset.drop("label",1),full_dataset["label"] , test_size = 0.2, random_state = 42)

columns_to_stdardize = ["statuses_count","followers_count","friends_count","favourites_count"]

def pre_processing_df(df):
    train_mean = df[columns_to_stdardize].mean()
    train_std = df[columns_to_stdardize].std()
    df[columns_to_stdardize] = (df[columns_to_stdardize] - train_mean) / train_std
    return train_mean,train_std

pre_processing_df(X_train)


from sklearn.ensemble.forest import RandomForestClassifier

 rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)


 rf.fit(X_train,y_train)

y_test_predicted = rf.predict(X_test)

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_test_predicted)

rf.feature_importances_