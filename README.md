# Zillow_Cluster_Regression

## Project Objective 
> Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 

* Create modules that faciliate project repeatability, as well as final report readability

> Ask/Answer exploratory questions of data and attributes to understand drivers of home value  

* Utilize charts and statistical tests

> Construct models to predict assessed home value for single family properties using regression techniques

> Make recommendations to a *fictional* data science team about how to improve predictions

> Refine work into report in form of jupyter notebook. 

> Present walkthrough of report in 5 minute presentation to classmates and instructors

* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer panel questions about all project areas

## Project Business Goals
> Construct ML Regression model that accurately predicts property tax assessed values (`taxvaluedollarcnt` - rename) of *Single Family Properties* using attributes of the properties? </br>

>Find key drivers of property value for single familty properties</br>

> Deliver report that the data science team can read through and replicate, while understanding what steps were taken, why and what the outcome was.

> Make recommendations on what works or doesn't work in predicting these homes' values

## Deliverables
> Github repo with a complete readme.md, a final report (.ipynb), acquire & prepare modules (.py), other supplemental artifacts created while working on the project (e.g. exploratory/modeling notebook(s))</br>
> 5 minute presentation of your final notebook</br>

## Data Dictionary
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
> Is there difference in median home value between counties?  

${H_0}$: There is no significant difference in median home value between counties   

${H_a}$: There is significant difference in median home value between counties    

${\alpha}$: .05

> Result: There is enough evidence to reject our null hypothesis.

> Why do some properties have a much higher value than others when they are located so close to each other? 

* ${\alpha}$ = .05

* ${H_0}$: There is no relationship between home value and area

* ${H_a}$: There is a relationship between home value and area

* Conclusion: There is enough evidence to reject our null hypothesis. This conclusion holds across all counties within the data

> Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location?

* ${\alpha}$ = .05

* ${H_0}$: There is no relationship between home value and estimated tax rate

* ${H_a}$: There is a relationship between home value and estimated tax rate

* Conclusion: There is enough evidence to reject our null hypothesis. This conclusion holds across all size categories

 >Is having one bathroom worse than having two bedrooms?
* ${\alpha}$ = .05

* ${H_0}$: The mean home value of homes with 1 bathroom is equal or greater than those with 2 bedrooms 

* ${H_a}$: The mean home value of homes with 1 bathroom is less than those with 2 bedrooms

 * Conclusion: There is enough evidence to reject our null hypothesis. We can confirm that having one bathroom is worse than having two bedrooms.

## Summary of Key Findings and Takeaways
* Bedrooms, bathrooms, home_size/area and lower age support higher home values
* Large numbers of homes in LA County 
* Estimated Tax rate has negative correlation with home value

## Pipeline Walkthrough
### Plan
> Create and build out project README
> Create required as well as supporting project modules and notebooks
* `env.py`, `wrangle.py`,  `model.py`,  `Final Report.ipynb`
* `wrangle.ipynb`,`model.ipynb` ,
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
* Estimated Tax Rate: Created to estimate a tax rate based off the home_value divided by the tax rate
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
>Three Best Models ran on Validation Set
* Polynomials 2 
    - Degrees: 3
    - Features: All
* Polynomial 12
    - Degrees: 3
    - Fetures: estimated tax rate, area, home age, and county
* Polynomial 10
    - Degrees: 3
    - Features: estimated tax rate, area, bathrooms, home age, and county

>Choose **one** model to test
* Polynomial 10
    - Degrees: 3
    - Features: estimated tax rate, area, bathrooms, home age, and county
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