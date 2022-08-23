import pandas as pd

data = pd.read_csv("df_results.csv",names=["experiment_id","model","auc_indomain","auc_outdomain","n_clusters","datasets_outdomain"])

grp_baseline = data[data["model"]=="baseline"].groupby("experiment_id",as_index=False).agg({"auc_indomain":"max"})
grp_baseline = grp_baseline.rename(columns={"auc_indomain": "auc_indomain_max_baseline"})

grp_clusters = data[data["model"]!="baseline"].groupby("experiment_id",as_index=False).agg({"auc_indomain":["max","min"]})
grp_clusters.columns = ["_".join(col_name).rstrip('_') for col_name in grp_clusters.columns.to_flat_index()]
grp_clusters = grp_clusters.rename(columns={"auc_indomain_max": "auc_indomain_max_cluster","auc_indomain_min":"auc_indomain_min_cluster"})

data = data.merge(grp_baseline,on=["experiment_id"])
data = data.merge(grp_clusters,on=["experiment_id"])

data["best_cluster"] = data[["auc_indomain","auc_indomain_max_cluster"]].apply(lambda x: 1 if x[0]==x[1] else 0,axis=1)
data["worst_cluster"] = data[["auc_indomain","auc_indomain_min_cluster"]].apply(lambda x: 1 if x[0]==x[1] else 0,axis=1)
data["cluster>baseline"] = data[["auc_indomain","auc_indomain_max_baseline"]].apply(lambda x: 1 if x[0]>x[1] else 0,axis=1)

data["num_datasets"] = data["datasets_outdomain"].apply(lambda x: x.count(",")+1)

dataset_dict = dict()
dataset_dict['E13_users.csv'] = 1481
dataset_dict['FSF_users.csv'] = 1169
dataset_dict['INT_users.csv'] = 1337
dataset_dict['TFP_users.csv'] = 469
dataset_dict['TWT_users.csv'] = 845
dataset_dict['botometer-feedback-2019.tsv'] = 461
dataset_dict['botwiki-2019.tsv'] = 654
dataset_dict['celebrity-2019.tsv'] = 5793
dataset_dict['cresci-rtbust-2019.tsv'] = 653
dataset_dict['cresci-stock-2018.tsv'] = 13161
dataset_dict['fake_followers.csv'] = 3351
dataset_dict['genuine_accounts.csv'] = 3474
dataset_dict['gilani-2017.tsv'] = 2491
dataset_dict['midterm-2018.tsv'] = 7729
dataset_dict['pronbots.tsv'] = 1876
dataset_dict['social_spambots_1.csv'] = 991
dataset_dict['social_spambots_2.csv'] = 3457
dataset_dict['social_spambots_3.csv'] = 464
dataset_dict['traditional_spambots_1.csv'] = 1000
dataset_dict['traditional_spambots_2.csv'] = 100
dataset_dict['traditional_spambots_3.csv'] = 403
dataset_dict['traditional_spambots_4.csv'] = 1128
dataset_dict['varol-2017.new3'] = 2208
dataset_dict['vendor-purchased-2019.tsv'] = 736

total = 0
for key in dataset_dict.keys():
    total += dataset_dict[key]

dataset_dict["total"] = total
    
data["num_examples_outdomain"] = data["datasets_outdomain"].apply(lambda x: sum([dataset_dict[y.strip()] for y in x.split(",")]))
data["num_examples_indomain"] = dataset_dict["total"] - data["num_examples_outdomain"]
