import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from joypy import joyplot
sns.set_theme(style="whitegrid")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score



def build_df(users_total,indomain_datasets):
    X = users_total[["dataset","label","statuses_count","followers_count","friends_count","favourites_count", \
            "listed_count","default_profile","geo_enabled","profile_use_background_image","verified","protected"]].fillna(0)

    X["label"] = X["label"].apply(lambda x: 0 if x == 'human' else 1)

    X[["statuses_count","followers_count","friends_count","favourites_count","listed_count"]] = X[["statuses_count","followers_count","friends_count","favourites_count","listed_count"]].apply(lambda x: np.log10(x+1))

    X.reset_index()

    #X_indomain = X[X["dataset"].isin(["fake_followers.csv","genuine_accounts.csv","social_spambots_1.csv","social_spambots_2.csv",
    #                                  "social_spambots_3.csv","traditional_spambots_1.csv","traditional_spambots_2.csv","traditional_spambots_3.csv"])].copy()
    X_indomain = X[X["dataset"].isin(indomain_datasets)].copy()
       
    
    X_train_indomain, X_test_indomain, y_train_indomain, y_test_indomain = train_test_split(X_indomain.drop(["dataset","label"], axis=1), X_indomain["label"], test_size=0.25, random_state=42)

    transformer = MinMaxScaler().fit(X_train_indomain)

    X_train_indomain = pd.DataFrame(transformer.transform(X_train_indomain),columns=["statuses_count","followers_count","friends_count","favourites_count", \
                "listed_count","default_profile","geo_enabled","profile_use_background_image","verified","protected"])
    X_test_indomain = pd.DataFrame(transformer.transform(X_test_indomain),columns=["statuses_count","followers_count","friends_count","favourites_count", \
                "listed_count","default_profile","geo_enabled","profile_use_background_image","verified","protected"])


    X_outdomain = X[~X["dataset"].isin(indomain_datasets)].copy()

    y_outdomain = X_outdomain[["dataset","label"]]
    X_outdomain = pd.DataFrame(transformer.transform(X_outdomain.drop(["dataset","label"], axis=1)),columns=["statuses_count","followers_count","friends_count","favourites_count", \
                "listed_count","default_profile","geo_enabled","profile_use_background_image","verified","protected"]).copy()

    
    return X_train_indomain, X_test_indomain, y_train_indomain, y_test_indomain,X_outdomain,y_outdomain




def run_experiment(X_train_indomain,X_test_indomain,y_train_indomain,y_test_indomain,X_outdomain,y_outdomain,n_clusters,experiment_id):


    df_results = pd.DataFrame(columns=['experiment_id', 'model', 'auc_indomain','auc_outdomain',"n_clusters","datasets_outdomain"])


    #Train baseline RandomForest using indomain data
    clf_ = RandomForestClassifier(random_state=42)
    clf = CalibratedClassifierCV(base_estimator=clf_)

    clf.fit(X_train_indomain,y_train_indomain)


    ## Baseline model: Get AUC for indomain and outdomain
    y_test_scored = clf.predict_proba(X_test_indomain)

    auc_baseline = roc_auc_score(y_test_indomain,y_test_scored[:,1])

    y_test_scored_out = clf.predict_proba(X_outdomain)

    auc_out_baseline = roc_auc_score(y_outdomain["label"],y_test_scored_out[:,1])

    df_results = df_results.append({'experiment_id': experiment_id,"model" : "baseline", "auc_indomain" : auc_baseline,"auc_outdomain" : auc_out_baseline,"n_clusters" : n_clusters,"datasets_outdomain" : ', '.join(y_outdomain.dataset.unique())}, ignore_index=True)
    
    ## Do clustering on the selected columns (using indomain data)

    df_to_cluster = X_train_indomain[["friends_count","followers_count"]]
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    clusterer.fit(df_to_cluster)

    X_train_cluster = pd.DataFrame(clusterer.predict(X_train_indomain[["friends_count","followers_count"]]),columns=["cluster"])

    X_train_cluster_dict = dict()
    clf_ = dict()

    
    ## Then train a RandomForestClassifier for each cluster
    X_train_indomain_copy = X_train_indomain.copy()
    X_train_indomain_copy["index"] = X_train_indomain_copy.index

    for current_cluster in range(0,n_clusters):

        X_train_cluster_dict[current_cluster] = X_train_cluster[X_train_cluster["cluster"] == current_cluster].copy()
        
        X_train_cluster_dict[current_cluster]["index"] = X_train_cluster_dict[current_cluster].index

        X_train_cluster_dict[current_cluster] = pd.merge(X_train_indomain_copy, X_train_cluster_dict[current_cluster], on="index", how="inner")
        y_train_segment = np.take(y_train_indomain,X_train_cluster_dict[current_cluster]["index"],axis=0)

        clf_[current_cluster] = RandomForestClassifier(random_state=42)
        clf_[current_cluster].fit(X_train_cluster_dict[current_cluster].drop(columns=["index","cluster"]).copy(),y_train_segment)


    ## Now predict ALL indomain data using ALL the RandomForestClassifier trained by each cluster, and select the one with highest indomain AUC
    scores = dict()
    scores_out = dict()
    # best_cluster = 0
    # max_auc = 0
    for current_cluster in range(0,n_clusters):
        scores[current_cluster] = clf_[current_cluster].predict_proba(X_test_indomain)
        auc = roc_auc_score(y_test_indomain,scores[current_cluster][:,1])

        scores_out[current_cluster] = clf_[current_cluster].predict_proba(X_outdomain)
        auc_out = roc_auc_score(y_outdomain["label"],scores_out[current_cluster][:,1])

        df_results = df_results.append({'experiment_id': experiment_id,"model" : current_cluster, "auc_indomain" : auc,"auc_outdomain" : auc_out,"n_clusters" : n_clusters,"datasets_outdomain" : ', '.join(y_outdomain.dataset.unique())}, ignore_index=True)

        #not doing anything with this for now
        # if auc > max_auc:
        #     max_auc = auc
        #     best_cluster = current_cluster
        
    
    return df_results

def get_valid_combinations(users_total,max_datasets_num=8):

    datasets = users_total["dataset"].unique()

    total_len = len(users_total)
    total_len_bot = len(users_total[users_total["label"]=='bot'])
    total_len_human = len(users_total[users_total["label"]=='human'])

    lengths = dict()
    lengths_bot = dict()
    lengths_human = dict()
    for dataset in datasets:
        lengths[dataset] = len(users_total[users_total["dataset"]==dataset])
        lengths_bot[dataset] = len(users_total[(users_total["dataset"]==dataset) & (users_total["label"]=='bot')])
        lengths_human[dataset] = len(users_total[(users_total["dataset"]==dataset) & (users_total["label"]=='human')])

    valid_combinations = list()
    for i in range(2,max_datasets_num):
        for comb in itertools.combinations(datasets,i):
            current_len = 0
            current_bot = 0
            current_human = 0


            for dataset in comb:
                current_len = current_len + lengths[dataset]
                current_bot = current_bot + lengths_bot[dataset]
                current_human = current_human + lengths_human[dataset]

            if (current_len > 0.5 * total_len and current_len < 0.8 * total_len) and (current_bot > 0.5 * total_len_bot and current_bot < 0.8 * total_len_bot) and (current_human > 0.5 * total_len_human and current_human < 0.8 * total_len_human):
                valid_combinations.append(comb)
    
    return valid_combinations






users = pd.read_csv("users2.csv")
users_cresci = pd.read_csv("users_cresci.csv")
users_total = users.append(users_cresci, ignore_index=True)


# 122659 valid combinations
valid_combinations = get_valid_combinations(users_total,max_datasets_num=10)


current_exp = 0
for indomain in valid_combinations:
    print(f'indomain datasets: {indomain}')
    print(" ")
    df_results = pd.DataFrame(columns=['experiment_id', 'model', 'auc_indomain','auc_outdomain',"n_clusters","datasets_outdomain"])

    X_train, X_test, y_train, y_test,X_outdomain,y_outdomain = build_df(users_total,indomain)
    df_results = df_results.append(run_experiment(X_train, X_test, y_train, y_test,X_outdomain,y_outdomain,2,current_exp),ignore_index=True)
    current_exp = current_exp + 1
    df_results = df_results.append(run_experiment(X_train, X_test, y_train, y_test,X_outdomain,y_outdomain,3,current_exp),ignore_index=True)
    df_results.to_csv("df_results.csv",mode="a",index=False,header=False)
    current_exp = current_exp + 1

