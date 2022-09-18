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

> Present walkthrough of report in 5 minute presentation to classmates and instructors

* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer panel questions about all project areas

## Project Business Goals
> Construct ML Regression model that accurately predicts log error of *Single Family Properties* using clustering techniques to guide feature selection for modeling</br>

>Find key drivers of log error</br>

> Deliver report that the data science team can read through and replicate, while understanding what steps were taken, why and what the outcome was.

> Make recommendations on what works or doesn't work in predicting log error, and insights gained from clustering

## Deliverables
> Github repo with a complete README.md, a final report (.ipynb), other supplemental artifacts and modules created while working on the project (e.g. exploratory/modeling notebook(s))</br>
> 5 minute recording of a presentation of final notebook</br>

## Data Dictionary **
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
cluster house_tax          | 51736 non-null  int32    | FE: Groups of ["tax_per_sqft","est_tax_rate","openness"]
cluster house_details      | 51736 non-null  int32    | FE: Groups of ["lotsizesquarefeet",'garagetotalsqft',"poolcnt"]
cluster house_sizing       | 51736 non-null  int32    | FE: Groups of ['area','bathrooms','bedrooms']
cluster house_locale       | 51736 non-null  int32    | FE: Groups of ["latitude","longitude","age"],
                    

## Initial Questions and Hypotheses
### Question 1 -  Does logerror and Area(sqft) have a signifcant relationship
* ${H_0}$: There is not a significant relationship in LogError and Area 
* ${H_a}$: There is a significant relationship in LogError and Area 
> Conclusion: There is enough evidence to reject our null hypothesis.

### Question 2 - Is there a difference in the the sample of no Heating for Log Error?
* ${H_0}$: There is no significant difference in mean LogError of the sample No Heating compared to the population  
* ${H_a}$: There is significant difference in mean LogError of the sample No Heating compared to the population  
> Conclusion: There is enough evidence to reject our null hypothesis.

### Question 3 -  Does the Cluster group of Home Size have any clusters that are significantly different in relation to logerror?
* ${H_0}$: There is no significant difference in mean LogError of the sample Home Size (Cluster 4) compared to the population  
* ${H_a}$: There is significant difference in mean LogError of the sample Home Size (Cluster 4) compared to the population  
> Conclusion: There is enough evidence to reject our null hypothesis. 
 
### Question 4 -  Does the Cluster group of House Locale have any clusters that are significantly different in relation to logerror?
* ${H_0}$: There is no significant difference in mean LogError of the sample House Locale (Cluster 1) compared to the population  
* ${H_a}$: There is significant difference in mean LogError of the sample House Locale (Cluster 1) compared to the population  
> Result: There is enough evidence to reject our null hypothesis.


## Summary of Key Findings and Takeaways
> - A combination of features and clusters will likely give us the best result to predict log error
> - These features may include: Area, Heating, Cluster of Home Size Area(sqft), Bedroom, Bathroom, Home Locale, Location, and age will all prove to have a level of significance in helping predict our target variable of log error in our modeling

## Pipeline Walkthrough
### Plan
> Create and build out project README
> Create required as well as supporting project modules and notebooks
* `env.py`, `wrangle.py`,  `model.py`,  `Final Report.ipynb`
* `wrangle.ipynb`,`model.ipynb` ,
> ~~Which columsn to import~~
> ~~Deal with nulls~~
> ~~include or remove outliers for exploration~~
> Explore
> Clustering stuff
> Statistical testing based on clustering 
> Modeling stuff 
* Handle acquire, explore, and scaling in wrangle

### Acquire
> Acquired zillow 2017 data from appropriate sources
* Create local .csv of raw data upon initial acquisition for later use
* Take care of any null values -> Decide on impute or elimination
> Add appropriate artifacts into `wrangle.py`

### Prepare
> Univariate exploration: 
* Basic histograms/boxplot for categories
> Take care of outliers
> Handle any possible threats of data leakage

> Feature Engineering *shifted to accomodate removal of outliers*
* Age: Columns that have the Age 
* Size: Column created to categorize homes by size
* Openness: Column created to measure the 'opennes'
* Estimated Tax Rate: Created to estimate a tax rate based off the home_value divided by the tax rate
* A cluster model: 
> - house_tax :|: [tax_per_sqft,est_tax_rate,openness]
> - house_details :|: [lotsizesquarefeet,garagetotalsqft,poolcnt]
> - house_sizing :|: [area,bathrooms,bedrooms]
> - house_locale :|: [latitude,longitude,age]

> Split data
> Scale data
> Collect and collate section *Takeaways*
> Add appropirate artifacts into `wrangle.py` or `explore.py`

### Explore
* Removed Year Built, and Tax Amount
> Bivariate exploration
* Investigate and visualize *all* features against home value
> Identify possible areas for feature engineering
* 
> Multivariate:
* Visuals exploring features as they relate to home value
> Statistical Analysis:
* Answer questions from *Initial Questions and Hyptheses* 
* Answer questions from *Univariate* and *Bivariate* exploration
> Collect and collate section *Takeaways*

### Model
> Ensure all data is scaled
> Create dummy vars of categorical columns
> Set up comparison dataframes for evaluation metrics and model descriptions  
> Set Baseline Prediction and evaluate accuracy  
> Explore various models and feature combinations.
* For initial M.V.P of each model include only `area`, `bedrooms`, `bathrooms` as features
> Choose **Four** Best Models ran on Validation Set

>Choose **one** model to test

> Collect and collate section *Takeaways*

### Deliver
> Create project report in form of jupyter notebook  
> Finalize and upload project repository with appropriate documentation 
* Verify docstring is implemented for each function within all notebooks and modules 
> Present to audience of CodeUp instructors and classmates


## Project Reproduction Requirements
> Requires personal `env.py` file containing database credentials  
> Steps:
* Fully examine this `README.md`
* Download `acquire.py, explore.py, model.py, prepary.py, and final_report.ipynb` to working directory
* Create and add personal `env.py` file to directory. Requires user, password, and host variables
* Run ``final_report.ipynb`