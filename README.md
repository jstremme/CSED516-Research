## Gradient Descent for Logistic Regression in Spark
### Joel Stremmel

**See final_report.docx for the final write-up of my analysis.**

### Project Goal:
I implement a simple Gradient Descent function in Spark from scratch, using the Python Spark interface in order to train a Logistic Regression model.  I compare this to a from-scratch, sequential implementation of Logistic Regression in Python which runs on a single machine.  I  also examine the MLlib implementation of Logistic Regression as a benchmark.  **In my final report I compare the execution time required to train these models on 2, 4, and 8 worker nodes on an AWS spark cluster to achieve the same level of accuracy.**

### Dataset Requirements:
For this analysis, I use the Chicago Food Inspections dataset from HealthData.gov, containing food inspection information for food facilities in Chicago since 2010.  The classification task is simple: predict if an establishment will pass or fail a food inspection based on the establishment category (bakery, cafe, etc.), the standardized latitude and longitude, and the documented (anticipated) risk of failing.  Before processing, the dataset is 219 MB in CSV format.

### Compute Resources:
To complete this analysis, I use my local Python environment, as well as AWS Spark clusters of 2, 4, and 8 m5.xlarge workers with 4vCores and 16 GiB memory.

### Programming Resources Used

Documentation for Python can be found here: https://docs.python.org/3.7/ and documentation for Jupyter Notebook can be found here: http://jupyter-notebook.readthedocs.io/en/latest/.

The following Python packages were used for this analysis:
- pandas 0.24.2
- requests 2.21.0   
- matplotlib 3.0.3
- numpy 1.17.2
- pyspark 2.3.2 
- scikit-learn 0.21.3

### Data Detail and Licenses

**Chicago Food Inspections data from HealthData.gov: https://healthdata.gov/dataset/food-inspections.**

At the time of writing, this dataset has 194,684 records of food inspections, and represents food inspections in the city of Chicago going back to 2010 as individual records.  Each record contains a facility type, name, address, date, zip code, longitude, latitude, risk status of the business, and inspection result, in addition to specific violations.  **The dataset is licensed under the ODbL (Open Database License): http://opendefinition.org/licenses/odc-odbl/.**  Of particular note, the dataset provides detailed location information allowing me to identify geographic patterns associated with passing or failing food inspections in the city of Chicago, and link to additional information about city neighborhoods to assess correlations with economic status.  

While this dataset has been made freely available by the city of Chicago, one ethical consideration is that the names of restaurants are made public in this dataset.  That said, it is precisely for public health reasons that this information is made public, so I do not remove these identifiers from my analysis.  However, I do not call attention to or disparage specific establishments over others, instead focusing on group trends and considering assumptions and possible confounding effects when summarizing my findings.

To address the potential confounding effects of economic status, I include median household income by zip code using the American Community Five Year Survey which I access through the US Census Data API.  See details on these data resources here:

- US Census Data API Terms of Service: https://www.census.gov/data/developers/about/terms-of-service.html
- API Key Signup: https://api.census.gov/data/key_signup.html
- API User Guide: https://www.census.gov/data/developers/guidance/api-user-guide.html
- American Community and Five Year Survey: https://www.census.gov/data/developers/data-sets/acs-5year.html
- American Community and Five Year Survey Variables: https://api.census.gov/data/2017/acs/acs5/variables.html
- Blogpost I Read to Help with Setup: https://towardsdatascience.com/getting-census-data-in-5-easy-steps-a08eeb63995d

**Disclaimers**
- As required by census.gov: "This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau."
- The `data` directory includes data licensed under the [odbl license](http://opendefinition.org/licenses/odc-odbl/) and that license is reproduced in `LICENSE.txt` in accordance with the requirements of the odbl license.

**Example Records**

| Inspection ID | DBA Name             | AKA Name             | License # | Facility Type        | Risk            | Address              | City    | State | Zip     | Inspection Date | Inspection Type       | Results         | Violations                                                                                                                                                                                                                              | Latitude           | Longitude    | Location                                 |
|---------------|----------------------|----------------------|-----------|----------------------|-----------------|----------------------|---------|-------|---------|-----------------|-----------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------|------------------------------------------|
| 2320831       | OGDEN PLAZA INC.     | OGDEN PLAZA INC.     | 2475982.0 | Grocery Store        | Risk 3 (Low)    | 3459 W OGDEN AVE     | CHICAGO | IL    | 60623.0 | 10/31/19        | Canvass               | Out of Business |                                                                                                                                                                                                                                         | 41.85526591        | -87.71240156 | (-87.71240156240032, 41.85526590922669)  |
| 2320793       | TACO MARIO'S LIMITED | TACO MARIO'S LIMITED | 2622418.0 | Mobile Food Preparer | Risk 2 (Medium) | 2300 S THROOP ST     | CHICAGO | IL    | 60608.0 | 10/30/19        | License               | Pass            |                                                                                                                                                                                                                                         | 41.85045102        | -87.65879786 | (-87.65879785567869, 41.85045102427)     |
| 2320830       | THE HOXTON, CHICAGO  | THE HOXTON, CHICAGO  | 2694640.0 | Restaurant           | Risk 2 (Medium) | 200 N GREEN ST       | CHICAGO | IL    | 60607.0 | 10/31/19        | License               | Pass            | 36. THERMOMETERS PROVIDED & ACCURATE - Comments: MUST PROVIDE THERMOMETERS IN ALL REFRIGERATION UNITS AND MAINTAIN.                                                                                                                     | 41.885699200000005 | -87.64878909 | (-87.64878908937915, 41.885699197163355) |
| 2320717       | ROCKS LAKEVIEW       | ROCKS LAKEVIEW       | 2304161.0 | Restaurant           | Risk 1 (High)   | 3463-3467 N BROADWAY | CHICAGO | IL    | 60657.0 | 10/29/19        | Canvass Re-Inspection | Pass            | 47. FOOD & NON-FOOD CONTACT SURFACES CLEANABLE, PROPERLY DESIGNED, CONSTRUCTED & USED - Comments: NOTED TORN RUBBER GASKET INSIDE THE PREP SERVICE COOLER AT THE KITCHEN PREP. INSTRUCTED TO DETAIL REPAIR AND MAINTAIN AND/OR REPLACE. | 41.94497417        | -87.64565976 | (-87.64565975587642, 41.94497417145062)  |
| 2320618       | A BEAUTIFUL RIND     | A BEAUTIFUL RIND     | 2670347.0 |                      | Risk 1 (High)   | 2211 N MILWAUKEE AVE | CHICAGO | IL    | 60647.0 | 10/28/19        | License               | Not Ready       |                                                                                                                                                                                                                                         | 41.92107616        | -87.69413786 | (-87.69413785909323, 41.921076157561416) |


### References

- Spark RDD paper with pseudocode for parallelized Logistic Regression: M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. Ma, M. McCauley, M. J. Franklin, S. Shenker, and I. Stoica. Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing. In Proceedings of NSDI, pages 15–28, 2012.

- Modeling techniques and ideas come from: Hastie, Trevor, Tibshirani, Robert and Friedman, Jerome. The Elements of Statistical Learning. New York, NY, USA: Springer New York Inc., 2001.

- The problem of identifying establishments likely to fail inspections has been addressed in part by the city of Chicago, using the data they have collected and made available.  Their approach focuses primarily on predicting establishments likely to fail inspections as a way of triaging which establishments need attention.  Their research is located here: https://github.com/Chicago/food-inspections-evaluation.

- Blogposts have covered the way the city of Chicago breaks up food establishments by risk of foodborne illness according to the type of food they serve and the way they serve it.  This blogpost details the way in which inspectors visit high-risk establishments more often than low-risk ones and has informed my thinking about the problem at hand: http://redlineproject.org/foodinspections.php.
