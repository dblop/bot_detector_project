import pandas as pd
import numpy as np



def test_split(variable, value, dataframe):
    """
    Divide o dataset em dois a partir de um par de threshold: variable,value
    """
    left = dataframe[dataframe[variable] < value]
    right = dataframe[dataframe[variable] >= value]
    return left, right

def gini_index(dataframe,label):
    """
    Calcula o Gini Index para o dataframe fornecido. Função usada por gini_split()
    """
    size_0 = len(dataframe[dataframe[label]==0]) 
    size_1 = len(dataframe[dataframe[label]==1])
    size_total = len(dataframe)
    if size_total == 0:
        #caso em que o split perfeito aconteceu e o algoritmo passa um dataset vazio para gini_index()
        return 0
    gini_index = 1.0 - (size_1 / size_total)**2 - (size_0 / size_total)**2
    return gini_index

def gini_split(dataframe_left,dataframe_right,label):
    """
    Calcula o Gini Index de um split, considerando a contribuição proporcional de cada nó filho
    """
    size_total = len(dataframe_left)+len(dataframe_right)
    gini_split = gini_index(dataframe_left,label)*len(dataframe_left)/size_total + gini_index(dataframe_right,label)*len(dataframe_right)/size_total
    return gini_split

def get_split(dataframe,label):
    """
    Seleciona qual o melhor split para o dataframe. Testa todas as colunas excluindo label, com 99 valores diferentes de percentil da variável.
    Feito apenas para variáveis numéricas.
    """
    columns_to_try = dataframe.columns.drop(label)
    best_column = ""
    best_value = float("inf")
    best_gini = float("inf")
    for column in columns_to_try:
        values_to_try = np.percentile(dataframe[column],range(1,100))
        for value in values_to_try:
            left, right = test_split(variable=column,value=value,dataframe=dataframe)
            current_gini = gini_split(dataframe_left=left,dataframe_right=right,label=label) 
            if current_gini < best_gini:
                best_gini = current_gini
                best_value = value
                best_column = column    
    return best_column,best_value,best_gini




class Tree:
    def __init__(self):
        self.id = None
        self.parent = None
        self.left = None
        self.right = None
        self.variable = None
        self.value = None
        self.gini = None
        self.depth = None
        self.prediction = None


def build_tree(dataframe,label,max_depth,parent,pct_sample=0.5,random_seed=1):
    """
    Algoritmo recursivo para construir uma árvore de decisão usada na Random Forest.
    Recebe um dataframe, amostra suas colunas, seleciona o melhor split e chama a si mesma nos nós filhos (caso necessário)
    """
    node = Tree()
    sampled = dataframe.drop(label,axis=1).sample(frac=pct_sample,axis=1,random_state=random_seed)
    sampled = pd.concat([sampled,dataframe[label]],axis=1)
    if parent is None:
        node.depth = 0
    else:
        node.parent = parent
        node.depth = parent.depth+1
    best_column,best_value,best_gini = get_split(sampled,label)
    node.variable = best_column
    node.value = best_value
    node.gini = best_gini

    dataframe_left = dataframe[dataframe[best_column] < best_value]
    dataframe_right = dataframe[dataframe[best_column] >= best_value]

    node.prediction_left = dataframe_left[label].mean()
    node.prediction_right = dataframe_right[label].mean()   
    if node.depth >= max_depth:
        return node
    if len(dataframe_left):
        if not all([x == dataframe_left.loc[:,label].iloc[0] for x in dataframe_left.loc[:,label]]):
            node.left = build_tree(parent=node,dataframe=dataframe_left,label=label,max_depth=max_depth,pct_sample=pct_sample,random_seed=random_seed+2)
    if len(dataframe_right):
        if not all([x == dataframe_right.loc[:,label].iloc[0] for x in dataframe_right.loc[:,label]]):
            node.right = build_tree(parent=node,dataframe=dataframe_right,label=label,max_depth=max_depth,pct_sample=pct_sample,random_seed=random_seed+100)
                
    return node


def predict(tree,row):
    """
    Percorre uma árvore e retorna o valor predito para um exemplo
    """
    current_node = tree
    while True:
        if row[current_node.variable] < current_node.value:
            prediction = current_node.prediction_left
            if current_node.left is None:
                break
            else:
                current_node = current_node.left
        else:
            prediction = current_node.prediction_right
            if current_node.right is None:
                break
            else:
                current_node = current_node.right

    return prediction




def myRandomForest(dataframe,label,max_depth,pct_sample,n_trees,random_seed=1):
    """
    Gera uma Random Forest. Amostra o conjunto de dados por árvore e treina um conjunto de árvores apropriadas para o método.
    """
    roots = list()
    rand = random_seed
    for i in range(0,n_trees):
        sampled = dataframe.sample(frac=pct_sample,axis=0,random_state=rand)
        roots.append( build_tree(dataframe=sampled,label=label,max_depth=max_depth,parent=None) )
        rand = rand + 1
    return roots


def predictRandomForest(forest,row):
    """
    Percorre todas as árvores da Random Forest e retorna o valor médio como predição
    """
    predictions = list()
    for root in forest:
        predictions.append(predict(root,row))
    prediction = sum(predictions) / len(predictions)
    return prediction