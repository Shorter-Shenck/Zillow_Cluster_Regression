#### Import Section
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import env
import wrangle_zillow
from os.path import exists

from itertools import product
from scipy.stats import levene , pearsonr, spearmanr, mannwhitneyu, f_oneway, ttest_ind,ttest_1samp
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, f_regression, SelectKBest

import warnings
warnings.filterwarnings("ignore")

# expore init
def explore_inital_guidance(train):
    """  
    takes in train and creates a previous made list of features to heatmap in the guidance of exploration
    """
    #list to heatmatp corr values
    heatmap_list = ['basementsqft','area','garagetotalsqft','latitude','longitude',
                    'lotsizesquarefeet','structuretaxvaluedollarcnt','home_value',
                    'landtaxvaluedollarcnt','logerror','age','home_size','est_tax_rate',
                    "bathrooms","bedrooms","openness"]

    #Creates and plotts the correlation values
    fig,ax = plt.subplots(figsize=(18,12))

    kwargs = {'alpha':1,'linewidth':5,'linestyle':'--','linecolor':'white'}

    sns.heatmap(train[heatmap_list].corr(),
                cmap="Spectral",mask=(np.triu(np.ones_like(train[heatmap_list].corr(),
                dtype=bool))),annot=True,vmin=-1, vmax=1)
    ax.add_patch(patches.Rectangle((9, 10),1.0,35.0,edgecolor='red',fill=False,lw=4) )
    ax.add_patch(patches.Rectangle((9,9),-10.0,1.0,edgecolor='red',fill=False,lw=4) )
    plt.xticks(rotation = 60)
    plt.title("Are there features that correlate higher than others?")
    plt.show()


# q2 explore heating varible
def explore_heating(train):
    ''' 
    specific made function, takes in train and runs a tailored function that 
    does a 1 sample ttest based on predefined expectations and plots a the sub-cat in relation to non sub-cat for that feature
    '''
    # sets variables
    target = "logerror"
    alpha = .05
    population_name = "heating"
    sample_name = "None"
    
    #sets null hypothesis
    H0 = f"{sample_name} as a sample has equal mean values to {population_name} as a population regarding {target}"
    Ha = f"{sample_name} as a sample does not have equal mean values to {population_name} as a population regarding {target}"

    #runs test and prints results
    t, p = ttest_1samp( train[train[population_name] == sample_name][target], train[target].mean())
    if p > alpha:
        print("\n We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
    else:
        print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))

    #creates a temp df that assists in plotting the feature
    temp1 = train.copy()
    temp1["Heating Compared"] = np.where(temp1[population_name] == sample_name,"Heating - No", "Heating - Yes")

    #does the plotting
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=temp1, x="Heating Compared", y="logerror",
        linewidth=3, edgecolor=".5", facecolor=(0, 0, 0, 0),)
    plt.axhline(y=temp1.logerror.mean(),label=f"LogError Mean {round(temp1.logerror.mean(),3)}",color="black")
    plt.legend()
    plt.title("Heating Compared in relation to LogError")
    plt.show()

#q3 house size cluster

def explore_house_size_clusters(train):
    ''' 
    specific made function, takes in train and runs a tailored function that 
    does a 1 sample ttest based on predefined expectations and plots a the sub-cat in relation to non sub-cat for that feature
    '''
    # sets variables
    target = "logerror"
    alpha = .05
    population_name = "cluster house_sizing"
    sample_name = 4
    H0 = f"{sample_name} as a sample has equal mean values to {population_name} as a population regarding {target}"
    Ha = f"{sample_name} as a sample does not have equal mean values to {population_name} as a population regarding {target}"
    
    #runs test and prints results
    t, p = ttest_1samp( train[train[population_name] == sample_name][target], train[target].mean())
    if p > alpha:
        print("\n We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
    else:
        print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))
    
    #creates a temp df that assists in plotting the feature
    temp1 = train.copy()
    temp1["House Sizing Clusters"] = np.where(temp1[population_name] == sample_name,"House Sizeing (cluster-4)", "House Sizeing (cluster-others)")
    #does the plotting
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=temp1, x="House Sizing Clusters", y="logerror",
        linewidth=3, edgecolor=".5", facecolor=(0, 0, 0, 0),)
    plt.axhline(y=temp1.logerror.mean(),label=f"LogError Mean {round(temp1.logerror.mean(),3)}",color="black")
    plt.legend()
    plt.title("House Size Clusters in relation to LogError")
    plt.show()

# Q4 house locale cluster

def explore_house_locale_cluster(train):    
    ''' 
    specific made function, takes in train and runs a tailored function that 
    does a 1 sample ttest based on predefined expectations and plots a the sub-cat in relation to non sub-cat for that feature
    '''
    # sets variables
    target = "logerror"
    alpha = .05
    population_name = 'cluster house_locale'
    sample_name = 1
    
    #sets null hypothesis
    H0 = f"{sample_name} as a sample has equal mean values to {population_name} as a population regarding {target}"
    Ha = f"{sample_name} as a sample does not have equal mean values to {population_name} as a population regarding {target}"

    #runs test and prints results
    t, p = ttest_1samp( train[train[population_name] == sample_name][target], train[target].mean())
    if p > alpha:
        print("\n We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
    else:
        print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))

    #creates a temp df that assists in plotting the feature
    temp1 = train.copy()
    temp1["House Locale Compared"] = np.where(temp1[population_name] == sample_name,"Cluster 1", "Other Clusters")

    #does the plotting
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=temp1, x="House Locale Compared", y="logerror",
        linewidth=3, edgecolor=".5", facecolor=(0, 0, 0, 0),
    )
    plt.axhline(y=temp1.logerror.mean(),label=f"LogError Mean {round(temp1.logerror.mean(),3)}",color="black")
    plt.legend()
    plt.title("House Locale Compared Compared in relation to LogError")
    plt.show()

#aux score cluster

def score_clusters(train,target):
    '''
    takes in your three sets and runs one sided ttest on the cluster groups to see if they were 
    averaging significantly different averages
    '''

    #creates a data frame, to append the iteration of result from 1 sample to population ttest result of clustering

    cluster_test_results = pd.DataFrame(columns=['test', 'population', "variable", "comparing on", "test_value", "p_val"])
    alpha = .05

    #loops through columns to find any with the name cluster in it
    for i in train.columns[train.columns.to_series().str.contains('cluster')].tolist():

        #sets the population name to the feature/column name
        population_name = f"{i}"
        #sets the sample name to the next unique name in the feature/column
        for sample_name in train[population_name].unique():
    
            t, p = ttest_1samp( train[train[population_name] == sample_name][target], train[target].mean())

            cluster_test_results.loc[len(cluster_test_results.index)] = ["One Sample TTest - 2 sided", 
                                                                f"{population_name}",
                                                                f"{population_name}({sample_name})",
                                                                target,
                                                                t,
                                                                p]       
  
    #puts out sorted dataframe                              
    cluster_test_results.sort_values(by=["p_val"])
    return(cluster_test_results)

def viz_clusters(train,target):
    '''  
    takes in dataframe and target string
    visualizes features with cluster in the name, tailored for single use case
    '''
    #plots a violin plot of the clusters for quick review
    cluster_features = train.columns[train.columns.to_series().str.contains('cluster')].tolist()
    l = 0
    plt.figure(figsize=(12,24))
    for cluster in cluster_features:
        l += 1
        plt.subplot(len(cluster_features),2,l)
        plot_order = train[cluster].sort_values(ascending=True).unique()
        sns.violinplot(    x=train[cluster], 
                        y=train[target], 
                        data=train, 
                        order = plot_order,
                        color ="grey",
                        inner="box"
                        #notch=True,
                        )
        plt.axhline(train[target].mean(),label=f"mean line - {round(train[target].mean(),0)}")
        plt.ylim([-.25,.25])
        plt.legend()
        plt.title(f"value of {target} sorted by {cluster}")
    plt.show() 

def explore_ttest_cluster(train,population_name="cluster house_sizing"):
    ''' 
    specific made function, takes in train and string name of cluster and runs a tailored function that 
    does a 1 sample ttest based on predefined expectations and plots a the sub-cat in relation to non sub-cat for that feature
    '''

    # sets variables
    target = "logerror"
    alpha = .05
    sample_name = int(score_clusters(train,target)\
                    [score_clusters(train,target)\
                        ["variable"].\
                            str.contains(population_name)].\
                                sort_values(by=["p_val"])\
                                    ["variable"].tolist()[0][-2:-1])

    #sets null hypothesis
    H0 = f"{sample_name} as a sample has equal average values to {population_name} as a population regarding {target}"
    Ha = f"{sample_name} as a sample does not have equal average values to {population_name} as a population regarding {target}"
    #runs test and prints results
    t, p = ttest_1samp( train[train[population_name] == sample_name][target], train[target].mean())
    if p > alpha:
        print("\n We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
    else:
        print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))
    #creates a temp df that assists in plotting the feature
    temp1 = train.copy()
    temp1[population_name] = np.where(temp1[population_name] == sample_name,f"{population_name}-{sample_name}", f"{population_name}-others")
    #does the plotting
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=temp1, x=population_name, y=target,
        linewidth=3, edgecolor=".5", facecolor=(0, 0, 0, 0),
    )
    plt.axhline(y=temp1.logerror.mean(),label=f"{target} Mean {round(temp1.logerror.mean(),3)}",color="black")
    plt.legend()
    plt.title(f"{population_name} in relation to {target}")
    plt.show()