# Zillow_Cluster_Regression

## Project Objective 
> Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 

* Create modules that faciliate project repeatability, as well as final report readability

> Ask/Answer exploratory questions of data and attributes to understand drivers of home value  

* Utilize charts, statistical tests, and various clustering models with 

> Construct models to predict `logerror`
* Log Error: log(zestimate) - log(Saleprice)
* A calculated value taking difference between the log of Zillow's Zestimate model estimate and the log of a properties actual sale price.

> Make recommendations to a *fictional* data science team about how to improve predictions

> Refine work into report in form of jupyter notebook. 

> Present walkthrough of report in 5 minute presentation to classmates and instructors

* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer panel questions about all project areas

## Project Business Goals
> Construct ML Regression model that accurately predicts log error of *Single Family Properties* using clustering techniques to guide feature selection </br>

>Find key drivers of log error</br>

> Deliver report that the data science team can read through and replicate, while understanding what steps were taken, why and what the outcome was.

> Make recommendations on what works or doesn't work in predicting log error, and insights gained from clustering

## Deliverables
> Github repo with a complete README.md, a final report (.ipynb), other supplemental artifacts and modules created while working on the project (e.g. exploratory/modeling notebook(s))</br>
> 5 minute recording of a presentation of final notebook</br>

## Data Dictionary **
|Target|Datatype|Definition|
|:-----|:-----|:-----|
|home_value|xxx non-null: uint8| property tax assessed values

|Feature|Datatype|Definition|
|:-----|:-----|:-----|
bathrooms       | 4225 non-null   float64 | number of bathrooms in home
bedrooms        | 4225 non-null   float64 | number of bedrooms in home
county          | 4225 non-null   object  | county of home based on FIPS code
area            | 4225 non-null   float64 | area of home in calculated square feet
home_size       | 4225 non-null   object | size of home grouped in category by square feet
home_age        | 4225 non-null   int64  | age of home
decades         | 4225 non-null   object | decade in which home was constructed
est_tax_rate    | 4225 non-null   float64 | calculated estimnated tax rate bbased on home value and paid tax


## Initial Questions and Hypotheses
### Question 1
 > Is there a difference in `garagetotlsqrefeet` between different `logerrorbins`?
* ${\alpha}$ = .05

* ${H_0}$: 

* ${H_a}$: 

 * Conclusion: 

### Question 2
 > Did the k = 3 cluster ('openness') object featuring bedrooms/homesize/bathrooms show difference in logerror mean
 * ${\alpha}$ = .05

* ${H_0}$: 

* ${H_a}$: 

 * Conclusion: 

 ### Question 3
 > Second cluster - location, age 
* ${\alpha}$ = .05

* ${H_0}$: 

* ${H_a}$: 

 * Conclusion: 
 
 ### Question 4 
 >
* ${\alpha}$ = .05

* ${H_0}$: 

* ${H_a}$: 

 * Conclusion: 


## Summary of Key Findings and Takeaways
* 
* 
* 
* 

## Pipeline Walkthrough
### Plan
> Create and build out project README
> Create required as well as supporting project modules and notebooks
* `env.py`, `wrangle.py`,  `model.py`,  `Final Report.ipynb`
* `wrangle.ipynb`,`model.ipynb` ,
> Which columsn to import 
> Deal with nulls
> include or remove outliers for exploration
> Explore
> Clustering stuff
* A cluster model: 
    - k=3: square feet, bathroom, bedrooms
    - k=3: pool, garage, acreage
    - k=3: estimated tax rate, openness value, land price per square foot
    - k=3:  
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
* Decades: Columns featuring the decade in which the home was 
* Age: Columns that have the Age 
* Size: Column created to categorize homes by size
* Openness: Column created to measure the 'opennes'
* Estimated Tax Rate: Created to estimate a tax rate based off the home_value divided by the tax rate
* Some calculation involving tax value vs lotsize and structure size/taxsssssss......
> Split data
> Scale data
> Collect and collate section *Takeaways*
> Add appropirate artifacts into `wrangle.py`

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
> Create dummy vars
> Set up comparison dataframes for evaluation metrics and model descriptions  
> Set Baseline Prediction and evaluate accuracy  
> Explore various models and feature combinations.
* For initial M.V.P of each model include only `area`, `bedrooms`, `bathrooms` as features
> Four Best Models ran on Validation Set


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
* Download `acquire.py, model.py, prepary.py, and final_report.ipynb` to working directory
* Create and add personal `env.py` file to directory. Requires user, password, and host variables
* Run ``final_report.ipynb`