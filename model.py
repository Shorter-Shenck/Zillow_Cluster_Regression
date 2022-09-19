#### Import Section
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import env
import wrangle_zillow
from os.path import exists

from itertools import product

from random import randint
from scipy.stats import levene , pearsonr, spearmanr, mannwhitneyu, f_oneway, ttest_ind
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, f_regression, SelectKBest

import warnings
warnings.filterwarnings("ignore")


def modeling_prep (train, train_scaled, validate, validate_scaled, test, test_scaled):
   """
   Purpose
      To return X, y subsets for training, validation, and testing of models

   Parameters
      train/validate/test: dataframes containing appropriate subsets of data
      train_scaled, validate_scaled, test_scaled: dataframes contianing scaled versions of approrpriate subsets of data

   Returns
      X_train, y_train, X_val, y_val, X_test, y_test: dataframes containing appropriate subsets of data
   """

   # create X,y for train, validate and test subsets
   X_train = train_scaled.drop(columns=['logerror'])
   y_train = train.logerror
   X_val = validate_scaled.drop(columns=['logerror'])
   y_val = validate.logerror
   X_test = test_scaled.drop(columns=['logerror'])
   y_test = test.logerror

   #shift y subsets into a data frame
   y_train = pd.DataFrame(y_train)
   y_val = pd.DataFrame(y_val)
   y_test = pd.DataFrame(y_test)

   #add baseline predictions
   y_train['pred_median'] = y_train.logerror.median()
   y_val['pred_median'] = y_val.logerror.median()
   y_test['pred_median'] = y_test.logerror.median()

   #get dummies for X subsets
   dummy_columns = ['county',
               'poolcnt',
               'home_size',
               'aircon',
               'heating',
               'cluster house_tax',
               'cluster house_details',
               'cluster house_sizing',
               'cluster house_locale',
               ]
   X_train = pd.get_dummies(X_train, columns=dummy_columns, drop_first=True)
   X_val = pd.get_dummies(X_val, columns=dummy_columns, drop_first=True)
   X_test = pd.get_dummies(X_test, columns=dummy_columns, drop_first=True)

   #add column after dummy creation to insure feature count match
   #X_train.insert(25, 'heating_Gravity', 0)

   return X_train, y_train, X_val, y_val, X_test, y_test

def select_kbest(X, y, k): 
    """
    Purpose
        To return the top features selecting by the SelectKBest function

    Parameters
       X: dataframe containing X subset of features for the data subset
       y: dataframe with series containing the target variable
       k: the number of features for the function to suggest 
    Returns
       f_top_features: list of the top features selected by SelectKBest function
    """
    # initilize selector object
    f_selector = SelectKBest(f_regression, k=k)

    #fit object --> will find top 2 as requested
    f_selector.fit(X, y)

    # create mask
    feature_mask = f_selector.get_support()

    # use mask to show list of feature support
    f_top_features = X.iloc[:,feature_mask].columns.tolist()

    return f_top_features

def rfe (X, y, n):
    """
    Purpose
        To return the top features selected by the RFE function

    Parameters
       X: dataframe containing X subset of features for the data subset
       y: dataframe with series containing the target variable
       n: the number of features for the function to select 
    Returns
       rfe_tip_features: list of the top features selected by SelectKBest function
    """
    #initialize  regression object
    lm = LinearRegression()

    # initilize RFE object with n features
    rfe = RFE(lm, n_features_to_select=n)

    #fit object onto data
    rfe.fit(X, y)

    #create boolean mask for columns model selects 
    feature_mask = rfe.support_

    # use mask to show list of selected features
    rfe_top_features = X.iloc[:, feature_mask].columns.tolist()

    return rfe_top_features

def get_features(X_train, y_train):
    """
    Purpose
        create a list of feature combinations to feed into the various models

    Parameters
       X_train: dataframe containing X subset of features for the data subset
       y_train: dataframe with series containing the target variable
       
    Returns
       feat_combos: list feature combinations
    """
    #create lists of features

    tax_feat = ['cluster house_tax_0.25', 'cluster house_tax_0.5', 'cluster house_tax_0.75', 'cluster house_tax_1.0',
                  'cluster house_locale_0.25', 'cluster house_locale_0.5', 'cluster house_locale_0.75', 'cluster house_locale_1.0',
                  'cluster house_sizing_0.25', 'cluster house_sizing_0.5', 'cluster house_sizing_0.75', 'cluster house_sizing_1.0',
                  'area', 'poolcnt_1.0', 'home_value', 'county_Orange County', 'county_Ventura County']

    details_feat = ['cluster house_details_0.3333333333333333', 'cluster house_details_0.6666666666666666', 'cluster house_details_1.0',
                  'cluster house_tax_0.25', 'cluster house_tax_0.5', 'cluster house_tax_0.75', 'cluster house_tax_1.0',
                  'cluster house_locale_0.25', 'cluster house_locale_0.5', 'cluster house_locale_0.75', 'cluster house_locale_1.0',
                  'area', 'poolcnt_1.0', 'home_value', 'county_Orange County', 'county_Ventura County']

    sizing_feat = ['cluster house_sizing_0.25', 'cluster house_sizing_0.5', 'cluster house_sizing_0.75', 'cluster house_sizing_1.0',
                  'cluster house_locale_0.25', 'cluster house_locale_0.5', 'cluster house_locale_0.75', 'cluster house_locale_1.0',
                  'area', 'poolcnt_1.0', 'home_value', 'county_Orange County', 'county_Ventura County']

    locale_feat = ['cluster house_locale_0.25', 'cluster house_locale_0.5', 'cluster house_locale_0.75', 'cluster house_locale_1.0',
                  'cluster house_details_0.3333333333333333', 'cluster house_details_0.6666666666666666', 'cluster house_details_1.0',
                  'area', 'poolcnt_1.0', 'home_value', ]

    feat_rfe = rfe(X_train, y_train.logerror, 10)
    print(feat_rfe)
    feat_sk_best = select_kbest(X_train, y_train.logerror, 10)
    print(feat_sk_best)

    #combine lists of features into large list feature all selected combinations
    feat_combos = [tax_feat, details_feat, sizing_feat, locale_feat, feat_sk_best]

    return feat_combos

def pf_mod(X, y, selectors, scores, fit_train=None, fit_y_train=None):
    """
    Purpose
       to create, train, and score linear regression models using polynomial features
    Parameters
       X: dataframe containing X subset of features for the data subset
       y: dataframe with series containing the target variable
       selectors: list of different feature and degree combinations for use with models
       fit_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
       fit_y_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
    Returns
       pf_description: dataFrame containing the scores, features, and parameters of the created models
    """
    #create empty data frame to hold model descriptions    
    #pf_descriptions = pd.DataFrame({}, columns=['Name','RMSE', 'Features', 'Parameters'])
    
    #loop through selector combinations to pull out different features and degree levels
    for idx, combo in enumerate(selectors):
        #create features object
        pf = PolynomialFeatures(degree=combo[1])
        #initialize model object
        lm = LinearRegression(normalize=True)
        #fit object on X_train subset depeneding on its position as parameter or the optional variant
        if fit_train is not None:
            fit_pf = pf.fit_transform(fit_train[combo[0]])
            X_pf = pf.transform(X[combo[0]])  
            lm.fit(fit_pf, fit_y_train.logerror)
        else:
            X_pf = pf.fit_transform(X[combo[0]])
            lm.fit(X_pf, y.logerror)

        model_label = f'Polynomial_{idx+1}'

        #predict
        if model_label in y:
            model_label = f'Polynomial_{randint(50,100)}'
            y[model_label] = lm.predict(X_pf)
        else:
            y[model_label] = lm.predict(X_pf)
         
        #calculate train rmse
        rmse = mean_squared_error(y.logerror, y[model_label], squared=False)

        description = pd.DataFrame([[model_label, rmse, combo[0], f'Degree: {combo[1]}']], columns=['Name', 'RMSE', 'Features', 'Parameters'])
        #pf_descriptions = pd.concat([pf_descriptions, description])
        scores = pd.concat([scores, description], ignore_index=True)

    return scores, y

def ols_mod(X, y, selectors, scores, fit_x_train=None, fit_y_train=None):
    """
    Purpose
       to create, train, and score ordinary least squares linear regression modelss
    Parameters
       X: dataframe containing X subset of features for the data subset
       y: dataframe with series containing the target variable
       selectors: list of different feature and degree combinations for use with models
       fit_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
       fit_y_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
    Returns
       pf_description: dataFrame containing the scores, features, and parameters of the created models
    """

    #loop through selector combinations to pull out different features and degree levels
    for idx, features in enumerate(selectors):  
        #create model object
        lm = LinearRegression()
        
        #fit object on X_train subset depeneding on its position as parameter or the optional variant
        if fit_x_train is not None:
            lm.fit(fit_x_train[features], fit_y_train.logerror)
        else:   
            lm.fit(X[features], y.logerror)

        #create mdoel label
        model_label = f'OLS_{idx+1}'

        if model_label in y:
            model_label = f'OLS_{randint(50,100)}'
            y[model_label] = lm.predict(X[features])
        else:
            y[model_label] = lm.predict(X[features]) 

        #calc trian rmse
        rmse = mean_squared_error(y.logerror, y[model_label], squared=False)

        description = pd.DataFrame([[model_label, rmse, features, 'N/A']], columns=['Name', 'RMSE', 'Features', 'Parameters'])
        scores = pd.concat([scores, description], ignore_index=True)

    return scores, y

def lars_mod(X, y, selectors, scores, fit_x_train=None, fit_y_train=None):
   """
   Purpose
      to create, train, and score linear regression models using polynomial features
   Parameters
      X: dataframe containing X subset of features for the data subset
      y: dataframe with series containing the target variable
      selectors: list of different feature and degree combinations for use with models
      fit_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
      fit_y_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
   Returns
      pf_description: dataFrame containing the scores, features, and parameters of the created models
   """

   #create empty data frame to hold model descriptions    
   lars_descriptions = pd.DataFrame({}, columns=['Name','RMSE', 'Features', 'Parameters'])

   #loop through selector combinations to pull out different features and degree levels
   for idx, selector in enumerate(selectors):  
      #create model object
      lars = LassoLars(alpha=selector[1])
      #create mdoel label
      model_label = f'LARS_{idx+1}'

      if fit_x_train is not None:
         lars.fit(fit_x_train[selector[0]], fit_y_train.logerror)
      else:   
         lars.fit(X[selector[0]], y.logerror)

      #predict train
      if model_label in y:
         model_label = f'LARS_{randint(50,100)}'
         y[model_label] = lars.predict(X[selector[0]])
      else:
         y[model_label] = lars.predict(X[selector[0]]) 

      #calc trian rmse
      rmse = mean_squared_error(y.logerror, y[model_label], squared=False)

      description = pd.DataFrame([[model_label, rmse, selector[0], f'Alpha: {selector[1]}']], columns=['Name', 'RMSE', 'Features', 'Parameters'])
      scores = pd.concat([scores, description], ignore_index=True)

   return scores, y


def GLM_mod(X, y, selectors, scores, fit_x_train=None, fit_y_train=None):
   """
   Purpose
      to create, train, and score linear regression models using polynomial features
   Parameters
      X: dataframe containing X subset of features for the data subset
      y: dataframe with series containing the target variable
      selectors: list of different feature and degree combinations for use with models
      fit_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
      fit_y_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
   Returns
      pf_description: dataFrame containing the scores, features, and parameters of the created models
   """
   
   #create empty data frame to hold model descriptions    
   glm_descriptions = pd.DataFrame({}, columns=['Name','RMSE', 'Features', 'Parameters'])

   #create empty data frame to hold model descriptions    
   for idx, selector in enumerate(selectors):  
      #create model object
      glm = TweedieRegressor(power=selector[1][0], alpha=selector[1][1])

      #create model label
      model_label = f'GLM_{idx+1}'

      #fit mode 
      #glm.fit(X, y.logerror)
      #fit object on X_train subset depeneding on its position as parameter or the optional variant
      if fit_x_train is not None:
         glm.fit(fit_x_train[selector[0]], fit_y_train.logerror)
      else:   
         glm.fit(X[selector[0]], y.logerror)

      #predict train
      if model_label in y:
         model_label = f'GLM_{randint(50,100)}'
         y[model_label] = glm.predict(X[selector[0]])
      else:
         y[model_label] = glm.predict(X[selector[0]])
         
      #calc rmse
      rmse = mean_squared_error(y.logerror, y[model_label], squared=False)

      description = pd.DataFrame([[model_label, rmse, selector[0], f'Power,Alpha: {selector[1]}']], columns=['Name', 'RMSE', 'Features', 'Parameters'])
      scores = pd.concat([scores, description], ignore_index=True)

   return scores, y 

def score_on_train(X_train, y_train): 
   """
   Purpose
      to create, train, and score linear regression models using diffent feature sets
   Parameters
      X: dataframe containing X subset of features for the data subset
      y: dataframe with series containing the target variable
      selectors: list of different feature and degree combinations for use with models
      fit_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
      fit_y_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
   Returns
      pf_description: dataFrame containing the scores, features, and parameters of the created models
   """
   y_train = y_train[['logerror', 'pred_median']]

   #calc rmse
   rmse = mean_squared_error(y_train.logerror, y_train.pred_median, squared=False)

   #create empty dataframe to hold model descriptions
   scores = pd.DataFrame([['pred_median', rmse, 0, 'N/A', 'N/A']], columns=['Name','RMSE', 'r^2 score','Features', 'Parameters'])
   
   #create lists of features
   feat_combos = get_features(X_train, y_train)

   #create a lists of parameters
   pf_parameters = [2]
   lars_parameters = [0, .1]
   glm_parameters = [(0,0), (0,.25), (0,.5), (0,.75), (0,1)]

   #use list with product to create tuples of feature/parameter combination to feed into model
   pf_selectors = list(product(feat_combos, pf_parameters))
   lars_selectors =  list(product(feat_combos, lars_parameters))
   glm_selectors =  list(product(feat_combos, glm_parameters))

   #run ols model with feature combinations
   scores, holder = pf_mod(X_train, y_train, pf_selectors, scores)
   scores, holder = ols_mod(X_train, y_train, feat_combos, scores)
   scores, holder = lars_mod(X_train, y_train, lars_selectors, scores)
   scores, holder = GLM_mod(X_train, y_train, glm_selectors, scores)


   for idx, model in enumerate(y_train.drop(columns='logerror').columns):
      scores.iat[(idx),2] = explained_variance_score(y_train['logerror'], y_train[model])

   return scores

def score_on_validate(train_scores, X_val, y_val, X_train, y_train): 
   """
   Purpose

   Parameters
      X: dataframe containing X subset of features for the data subset
      y: dataframe with series containing the target variable
      selectors: list of different feature and degree combinations for use with models
      fit_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
      fit_y_train: optional, X_train subset of data to fit the model when needing to score validation or test subsets
   Returns
      pf_description: dataFrame containing the scores, features, and parameters of the created models
   """
   y_val = y_val[['logerror', 'pred_median']]
   
   #create new dataframe that is the top 10 (for and runs them)
   validate_scores = train_scores.copy(deep=True)
   validate_scores = validate_scores.set_index('Name')

   #calc rmse
   rmse = mean_squared_error(y_val.logerror, y_val.pred_median, squared=False)

   #create empty dataframe to hold model descriptions
   model_descriptions = pd.DataFrame([['pred_median', rmse, 0, 'N/A', 'N/A']], columns=['Name','RMSE', 'r^2 score','Features', 'Parameters'])
   
   for model in validate_scores.index:
      #empty list to hold feature combinations
      feat_combos = []
  
      #create a lists of parameters
      pf_selectors = []
      lars_selectors = []
      glm_selectors = []

      if model.startswith('Pol'):
         features = validate_scores.loc[model]['Features']
         degree = int(validate_scores.loc[model]['Parameters'][-1])
         pf_selectors.append((features, degree))
         model_descriptions, y_val = pf_mod(X_val, y_val, pf_selectors, model_descriptions, X_train, y_train)
      elif model.startswith('GLM'):
         pow_alpha = eval(validate_scores.loc[model]['Parameters'][13:])
         features = validate_scores.loc[model]['Features']
         glm_selectors.append((features,pow_alpha))
         model_descriptions, y_val = GLM_mod(X_val, y_val, glm_selectors, model_descriptions, X_train, y_train)
      elif model.startswith('LARS'):
         features = validate_scores.loc[model]['Features']
         alpha = eval(validate_scores.loc[model]['Parameters'][7:])
         lars_selectors.append((features, alpha))
         model_descriptions, y_val = lars_mod(X_val, y_val, lars_selectors, model_descriptions, X_train, y_train)
      elif model.startswith('OLS_'):
         feat_combos.append((validate_scores.loc[model]['Features']))
         model_descriptions, y_val = ols_mod(X_val, y_val, feat_combos, model_descriptions, X_train, y_train)
      
      model_descriptions.iat[-1, 0] = model
      
   for idx, model in enumerate(y_val.drop(columns='logerror').columns):
      model_descriptions.iat[idx,2] = explained_variance_score(y_val['logerror'], y_val[model])

   return model_descriptions, y_val

def score_on_test(X_test, y_test, X_train, y_train):
    #reset y_test variable to remove scores from previous model
    y_test = y_test[['logerror', 'pred_median']]

    #find rmse of baseline predictions 
    test_pred_rmse = mean_squared_error(y_test.logerror, y_test.pred_median, squared=False)

    #creates dataframe to hold model score on test set
    test_score = pd.DataFrame([['pred_median', test_pred_rmse, 0, 'N/A', 'N/A']], columns=['Name','RMSE', 'r^2 score','Features', 'Parameters'])

    #select parameters or features for final modeling on test set
    #test_selectors = [(['bedrooms', 'area', 'county_Orange County', 'home_size_large'], 2)]
    test_features = ['basementsqft', 'area', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'home_value',
                     'landtaxvaluedollarcnt', 'tax_per_sqft', 'poolcnt_1.0', 'cluster house_details_0.3333333333333333', 'cluster house_details_1.0']
    test_parameters = (0,0)
    test_selectors = [(test_features, test_parameters)]

    #fit and use the model that scored highest on validate set
    #test_score, holder = pf_mod(X_test, y_test, test_selectors, test_score, X_train, y_train)
    test_score, holder = GLM_mod(X_test, y_test, test_selectors, test_score, X_train, y_train)

    #adds correct model name to data frame
    test_score.iat[1, 0] = 'GLM_1'

    #add r^2 score 
    test_score.iat[1,2] = explained_variance_score(y_test['logerror'], y_test.iloc[:,2])
    print(y_test.iloc[:,2])

    return test_score