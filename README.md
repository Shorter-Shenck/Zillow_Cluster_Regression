# Zillow_Cluster_Regression

## Project Objective 
> Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 

* Create modules that faciliate project repeatability, as well as final report readability

> Ask/Answer exploratory questions of data and attributes to understand drivers of home value  

* Utilize charts, statistical tests, and various clustering models to drive linear regression models; improving baseline model

> Construct models to predict `logerror`
* Log Error: log(zestimate) - log(Saleprice)
* A calculated value taking difference between the log of Zillow's Zestimate model estimate and the log of a properties actual sale price.

> Make recommendations to a *fictional* data science team about how to improve predictions

> Refine work into report in form of jupyter notebook. 

> Present walkthrough of report in 5 minute recorded presentation

* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer panel questions about all project areas

## Project Business Goals
> Construct ML Regression model that accurately predicts log error of *Single Family Properties* using clustering techniques to guide feature selection for modeling</br>

> Find key drivers of log error</br>

> Deliver report that the data science team can read through and replicate, while understanding what steps were taken, why and what the outcome was.

> Make recommendations on what works or doesn't work in predicting log error, and insights gained from clustering

## Deliverables
> Github repo with a complete README.md, a final report (.ipynb), other supplemental artifacts and modules created while working on the project (e.g. exploratory/modeling notebook(s))</br>
> 5 minute recording of a presentation of final notebook</br>

## Data Dictionary
|       Target             |           Datatype       |     Definition      |
|:-------------------------|:------------------------:|-------------------:|  
logerror                   | 51736 non-null  float64  | Log(Zestimate) - Log(SalePrice): Difference in estimated and actual

|       Feature            |           Datatype       |     Definition      |
|:------------------------|:------------------------:|-------------------:|  
basementsqft               | 51736 non-null  float64  | Square Feet (Sqft) of basement
bathrooms                  | 51736 non-null  float64  | Count of bathrooms
bedrooms                   | 51736 non-null  float64  | Count of bedrooms
area                       | 51736 non-null  float64  | Finished Sqft of Home
county                     | 51736 non-null  object   | Name of County Home is in
garagecarcnt               | 51736 non-null  float64  | Count of Cars rated for Garage
garagetotalsqft            | 51736 non-null  float64  | Sqft of Garage
latitude                   | 51736 non-null  float64  | Angular distance (East/West) of a home (degrees and minutes)
longitude                  | 51736 non-null  float64  | Angular distance (North/South) of a home (degrees and minutes)
lotsizesquarefeet          | 51736 non-null  float64  | Sqft of Lot (land)
poolcnt                    | 51736 non-null  float64  | Count of Pools
structuretaxvaluedollarcnt | 51736 non-null  float64  | Tax Value of Structure/Home
home_value                 | 51736 non-null  float64  | Tax Value of Property
landtaxvaluedollarcnt      | 51736 non-null  float64  | Tax Value of Land/Lot
aircon                     | 51736 non-null  object   | Air Conditioning Type
heating                    | 51736 non-null  object   | Heating Type
age                        | 51736 non-null  float64  | FE: Years since House Built
openness                   | 51736 non-null  float64  | FE: Area / (Bedrooms + Bathrooms): Relative size
tax_per_sqft               | 51736 non-null  float64  | FE: Home Value / Sqft: Relative value
home_size                  | 51736 non-null  category | FE: Binned Category of Area
est_tax_rate               | 51736 non-null  float64  | FE: Estimated Tax Rate of County
cluster house_tax          | 51736 non-null  int32    | FE: Clusters based on `tax_per_sqft`,`est_tax_rate`, and `openness`
cluster house_details      | 51736 non-null  int32    | FE: Clusters based on `lotsizesquarefeet`, `garagetotalsqft`, `poolcnt`
cluster house_sizing       | 51736 non-null  int32    | FE: Clusters based on `area`, `bathrooms`, and `bedrooms`
cluster house_locale       | 51736 non-null  int32    | FE: Clusters based on `latitude`,`longitude`, and `age`
-----                    

# Initial Questions and Hypotheses
## Question 1 -  Does logerror and Area(sqft) have a signifcant relationship?
* ${H_0}$: There is not a significant relationship in LogError and Area 
* ${H_a}$: There is a significant relationship in LogError and Area 
> Conclusion: There is enough evidence to reject our null hypothesis.

## Question 2 - Is there a difference in log error between the different types of heating?
* ${H_0}$: There is no significant difference in mean LogError of the sample No Heating compared to the population  
* ${H_a}$: There is significant difference in mean LogError of the sample No Heating compared to the population  
> Conclusion: There is enough evidence to reject our null hypothesis.

## Question 3 -  Does the Home Size clustering reveal any differences between the clusters?
* ${H_0}$: There is no significant difference in mean LogError of the sample Home Size (Cluster 4) compared to the population  
* ${H_a}$: There is significant difference in mean LogError of the sample Home Size (Cluster 4) compared to the population  
> Conclusion: There is enough evidence to reject our null hypothesis. 
 
## Question 4 - Does Locale clustering reveal any differences between the clusters?
* ${H_0}$: There is no significant difference in mean LogError of the sample House Locale (Cluster 0) compared to the population  
* ${H_a}$: There is significant difference in mean LogError of the sample House Locale (Cluster 0) compared to the population  
> Result: There is enough evidence to reject our null hypothesis.


## Summary of Key Findings and Takeaways
* Low correlation values among all features with `logerror` led to relying on clustering for bulk of feature differientation
* Cluster creation was able to show difference in log error prediction
* Feature sets informed by clustering performed best on model through validation phase
    * Best model utilized Clustering based on tax information, size, and locale
* Model gain on predictive performance vs. baseline prediction using median `logerror` was minimal on test set
    * Baseline Prediction RMSE: 0.1531
    * Model RMSE: 0.1529 (Lower is better) 
-----
</br></br></br>

# Pipeline Walkthrough
## Plan
> Create and build out project README  
> Create required as well as supporting project modules and notebooks
* `env.py`, `wrangle_zillow.py`,  `model.py`,  `Final Report.ipynb`
* `wrangle.ipynb`,`model.ipynb` ,
> Decide which colums to import   
> Deal with null values  
> Decide how to deal with outliers  

> Clustering
- Decide on which features to use when crafting clusters
- Create cluster feature sets
- Add cluster labels as features  
> Statistical testing based on clustering
- Create functions that iterate through statistical tests
- Organize in Explore section 
> Explore
- Visualize cluster differences to gauge impact
- Rank clusters based on statistical weight
> Modeling
* Create functions that automate iterative model testing
    - Adjust parameters and feature makes
* Handle acquire, explore, and scaling in wrangle
> Verify docstring is implemented for each function within all notebooks and modules 
 

## Acquire
> Acquired zillow 2017 data from appropriate sources
* Create local .csv of raw data upon initial acquisition for later use
* Take care of any null values -> Decide on impute or elimination
* Eliminated 
> Add appropriate artifacts into `wrangle_zillow.py`

## Prepare
> Univariate exploration: 
* Basic histograms/boxplot for categories
> Take care of outliers  
> Handle any possible threats of data leakage
* Removed log error bins to prevent leakage

> Feature Engineering **shifted to accomodate removal of outliers*
* `age`: Feature that reflects the age of the property
* `home_size`: Feature places homes into size categories based on home sqft
* `openness`: Ratio of home sqft to combined number of bed/bathrooms
* `est_tax_rate`: Created to estimate a tax rate based off the home_value divided by the tax rate
* Cluster modeling: 
> - `house_tax` :|: `tax_per_sqft`, `est_tax_rate`, `openness`
> - `house_details` :|: `lotsizesquarefeet`, `garagetotalsqft`, `poolcnt`
> - `house_sizing` :|: `area`, `bathrooms`, `bedrooms`
> - `house_locale` :|: `latitude`, `longitude` , `age`

> Split data  
> Scale data  
> Collect and collate section *Takeaways*  
> Add appropirate artifacts into `wrangle.py` or `explore.py`

## Explore
* Removed Year Built, and Tax Amount
> Bivariate exploration
* Investigate and visualize *all* features against log error
> Identify possible areas for feature engineering
* 
> Multivariate:
* Visuals exploring features as they relate to home value
> Statistical Analysis:
* 
* 
> Collect and collate section *Takeaways*

## Model
> Ensure all data is scaled  
> Create dummy vars of categorical columns  
> Set up comparison dataframes for evaluation metrics and model descriptions    
> Set Baseline Prediction and evaluate RMSE and r^2 scores  
> Explore various models and feature combinations.
* For initial M.V.P of each model include only single cluster label features
> Choose **Four** Best Models to add to final report

>Choose **one** model to evaluate on Test set
* GLM 1
* Power: 3
* Alpha: 0
* Features: All features and cluster labels 

> Collect and collate section *Takeaways*

## Deliver
> Create project report in form of jupyter notebook  
> Finalize and upload project repository with appropriate documentation 
> Created recorded presentation for delivery
----
</br>

## Project Reproduction Requirements
> Requires personal `env.py` file containing database credentials  
> Steps:
* Fully examine this `README.md`
* Download `wrangle_zillow.py, explore_zillow.py`, `model.py`, and `Final Report.ipynb` to working directory
* Create and add personal `env.py` file to directory. Requires user, password, and host variables
* Run `Final Report.ipynb`