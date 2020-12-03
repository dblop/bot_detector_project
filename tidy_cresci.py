import csv

#list of attributes being used for other datasets
param_list = ["dataset","label","profile_banner_url","id","id_str","name","screen_name","location","profile_location","description","url","protected","followers_count","friends_count","listed_count","favourites_count","utc_offset","time_zone","geo_enabled","verified","statuses_count","lang","contributors_enabled","is_translator","is_translation_enabled","profile_background_color","profile_background_image_url","profile_background_image_url_https","profile_background_tile","profile_image_url","profile_image_url_https","profile_link_color","profile_sidebar_border_color","profile_sidebar_fill_color","profile_text_color","profile_use_background_image","has_extended_profile","default_profile","default_profile_image","following","follow_request_sent","notifications","translator_type","created_at"]
#params_list of cresci datasets:                              "id","name","screen_name","statuses_count","followers_count","friends_count","favourites_count","listed_count","url","lang","time_zone","location","default_profile","default_profile_image","geo_enabled","profile_image_url","profile_banner_url","profile_use_background_image","profile_background_image_url_https","profile_text_color","profile_image_url_https","profile_sidebar_border_color","profile_background_tile","profile_sidebar_fill_color","profile_background_image_url","profile_background_color","profile_link_color","utc_offset","is_translator","follow_request_sent","protected","verified","notifications","description","contributors_enabled","following","created_at","timestamp","crawled_at","updated","test_set_2"


#remake file
with open(f'./users_cresci.csv','w') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=param_list)
    writer.writeheader()



#same pattern - cresci-2017
datasets = ["fake_followers.csv","genuine_accounts.csv","social_spambots_1.csv","social_spambots_2.csv","social_spambots_3.csv" \
,"traditional_spambots_1.csv","traditional_spambots_2.csv","traditional_spambots_3.csv","traditional_spambots_4.csv"]


for dataset in datasets:
    with open(f'./data/cresci-2017.csv/datasets_full.csv/{dataset}/users.csv', 'r') as infile, open(f'./users_cresci.csv','a') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=param_list,extrasaction='ignore')
        #writer.writeheader()
        for row in csv.DictReader(infile):
            row["dataset"] = dataset
            row["label"] = "human" if dataset == "genuine_accounts.csv" else "bot"
            row["id_str"] = row["id"]
            writer.writerow(row)        


#same pattern - cresci-2015
datasets = ["E13_users.csv","FSF_users.csv","INT_users.csv","TFP_users.csv","TWT_users.csv"]

#yes, some of them have weird NUL values. Probably encoding shenanigans.
for dataset in datasets:
    with open(f'./data/cresci-2015.csv/{dataset}', 'r') as infile:
        data = infile.read().replace('\0' , '')
    with open(f'./data/cresci-2015.csv/{dataset}_new', 'w') as outfile:
        outfile.write(data)


for dataset in datasets:
    with open(f'./data/cresci-2015.csv/{dataset}_new', 'r') as infile, open(f'./users_cresci.csv','a') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=param_list,extrasaction='ignore')
        #writer.writeheader()
        for row in csv.DictReader(infile):
            row["dataset"] = dataset
            row["label"] = "human" if dataset == "E13_users.csv" or dataset == "TFP_users.csv" else "bot"
            row["id_str"] = row["id"]
            writer.writerow(row)        

