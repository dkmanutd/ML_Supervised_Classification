# importing timeit
import timeit

code_to_test = """

# Student Name : Deepak Kumar
# Cohort       : Haight

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

import pandas as pd                                  # data science essentials
from sklearn.model_selection import train_test_split # train_test_split
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import roc_auc_score            # auc score
from sklearn.preprocessing import StandardScaler     # standard scaler
from sklearn.tree import DecisionTreeClassifier      # classification trees
from sklearn.tree import export_graphviz             # exports graphics
from sklearn.externals.six import StringIO           # saves objects in memory
from IPython.display import Image                    # displays on frontend
import pydotplus                                     # interprets dot objects

################################################################################
# Load Data
################################################################################

file = 'Apprentice_Chef_Dataset.xlsx'
original_df = pd.read_excel(file)                    # reading in the dataset

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

#display_tree user-defined function
####################################
def display_tree(tree, feature_df, height = 500, width = 800):
    
    #PARAMETERS
    #----------
    #tree       : fitted tree model object
    #    fitted CART model to visualized
    #feature_df : DataFrame
    #    DataFrame of explanatory features (used to generate labels)
    #height     : int, default 500
    #    height in pixels to which to constrain image in html
    #width      : int, default 800
    #    width in pixels to which to constrain image in html
    #

    # visualizing the tree
    dot_data = StringIO()

    
    # exporting tree to graphviz
    export_graphviz(decision_tree      = tree,
                    out_file           = dot_data,
                    filled             = True,
                    rounded            = True,
                    special_characters = True,
                    feature_names      = feature_df.columns)


    # declaring a graph object
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


    # creating image
    img = Image(graph.create_png(),
                height = height,
                width  = width)
    
    return img
####################################


# Since the objective of the analysis is to understand how Apprentice Chef can diversify their revenue stream, I'm adding some new features.

# Adding a new feature to further understand the dataset:
original_df['PRICE_PER_ORDER'] = original_df['REVENUE']/original_df['TOTAL_MEALS_ORDERED']
    # with this, I want to understand how much each person is paying per order

# Setting outlier thresholds:
REVENUE_hi = 2500
AVG_TIME_PER_SITE_VISIT_hi = 200
AVG_PREP_VID_TIME_lo = 50
AVG_PREP_VID_TIME_hi = 275
AVG_CLICKS_PER_VISIT_lo = 8
AVG_CLICKS_PER_VISIT_hi = 20
FOLLOWED_RECOMMENDATIONS_PCT_hi = 80
TOTAL_MEALS_ORDERED_lo = 25
TOTAL_MEALS_ORDERED_hi = 150
UNIQUE_MEALS_PURCH_hi = 9
CONTACTS_W_CUSTOMER_SERVICE_hi = 12
CANCELLATIONS_BEFORE_NOON_hi = 7
MOBILE_LOGINS_lo = 4.5
MOBILE_LOGINS_hi = 6.5
PC_LOGINS_lo = 1
PC_LOGINS_hi = 2
WEEKLY_PLAN_hi = 20
EARLY_DELIVERIES_hi = 5
LATE_DELIVERIES_hi = 10
LARGEST_ORDER_SIZE_lo = 1
LARGEST_ORDER_SIZE_hi = 7
MASTER_CLASSES_ATTENDED_hi = 2
TOTAL_PHOTOS_VIEWED_hi = 600
MEDIAN_MEAL_RATING_hi = 4

## Feature Engineering (outlier thresholds)

# Developing features (columns) for outliers:

#Revenue
original_df['out_REVENUE'] = 0
condition_hi = original_df.loc[0:,'out_REVENUE'][original_df['REVENUE'] > REVENUE_hi]

original_df['out_REVENUE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#AVG_TIME_PER_SITE_VISIT
original_df['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_TIME_PER_SITE_VISIT']\
                              [original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]

original_df['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#AVG_PREP_VID_TIME
original_df['out_AVG_PREP_VID_TIME'] = 0
condition_hi = original_df.loc[0:,'out_AVG_PREP_VID_TIME']\
                              [original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]
condition_lo = original_df.loc[0:,'out_AVG_PREP_VID_TIME']\
                              [original_df['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_lo]

original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#AVG_CLICKS_PER_VISIT
original_df['out_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT']\
                              [original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]
condition_lo = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT']\
                              [original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#FOLLOWED_RECOMMENDATIONS_PCT
original_df['out_FOLLOWED_RECOMMENDATIONS_PCT'] = 0
condition_hi = original_df.loc[0:,'out_FOLLOWED_RECOMMENDATIONS_PCT']\
                              [original_df['FOLLOWED_RECOMMENDATIONS_PCT'] > FOLLOWED_RECOMMENDATIONS_PCT_hi]

original_df['out_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#TOTAL_MEALS_ORDERED
original_df['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED']\
                              [original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]
condition_lo = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED']\
                              [original_df['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_lo]

original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#UNIQUE_MEALS_PURCH
original_df['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = original_df.loc[0:,'out_UNIQUE_MEALS_PURCH']\
                              [original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]

original_df['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#CONTACTS_W_CUSTOMER_SERVICE
original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE']\
                              [original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#CANCELLATIONS_BEFORE_NOON
original_df['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_BEFORE_NOON']\
                              [original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]

original_df['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#MOBILE_LOGINS
original_df['out_MOBILE_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_MOBILE_LOGINS']\
                              [original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_hi]
condition_lo = original_df.loc[0:,'out_MOBILE_LOGINS']\
                              [original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_lo]

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#PC_LOGINS
original_df['out_PC_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_PC_LOGINS']\
                              [original_df['PC_LOGINS'] > PC_LOGINS_hi]
condition_lo = original_df.loc[0:,'out_PC_LOGINS']\
                              [original_df['PC_LOGINS'] < PC_LOGINS_lo]

original_df['out_PC_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_PC_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#WEEKLY_PLAN
original_df['out_WEEKLY_PLAN'] = 0
condition_hi = original_df.loc[0:,'out_WEEKLY_PLAN']\
                              [original_df['WEEKLY_PLAN'] > WEEKLY_PLAN_hi]

original_df['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#EARLY_DELIVERIES
original_df['out_EARLY_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_EARLY_DELIVERIES']\
                              [original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]

original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#LATE_DELIVERIES
original_df['out_LATE_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_LATE_DELIVERIES']\
                              [original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]

original_df['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#LARGEST_ORDER_SIZE
original_df['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = original_df.loc[0:,'out_LARGEST_ORDER_SIZE']\
                              [original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]
condition_lo = original_df.loc[0:,'out_LARGEST_ORDER_SIZE']\
                              [original_df['PRODUCT_CATEGORIES_VIEWED'] < LARGEST_ORDER_SIZE_lo]

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#MASTER_CLASSES_ATTENDED
original_df['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = original_df.loc[0:,'out_MASTER_CLASSES_ATTENDED']\
                              [original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_hi]

original_df['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#TOTAL_PHOTOS_VIEWED
original_df['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED']\
                              [original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_hi]

original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#MEDIAN_MEAL_RATING
original_df['out_MEDIAN_MEAL_RATING'] = 0
condition_hi = original_df.loc[0:,'out_MEDIAN_MEAL_RATING']\
                              [original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# placeholder list for the e-mail domains
placeholder_lst = []

# looping over each email address
for index, col in original_df.iterrows():
    
    # splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)     

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)

# renaming columns
email_df.columns = ['concatenate' , 'email_domain']

# concatenating email_domain with Apprentice Chef DataFrame
original_df = pd.concat([original_df, email_df.loc[: ,'email_domain']],
                   axis = 1) # because we are concatenating by column

# creating email domains
professional_domains = ['@mmm.com', '@amex.com', '@apple.com', '@boeing.com',
                       '@caterpillar.com', '@chevron.com', '@cisco.com',
                       '@cocacola.com', '@disney.com','@dupont.com',
                       '@exxon.com', '@ge.org', '@goldmansacs.com',
                       '@homedepot.com', '@ibm.com', '@intel.com', '@jnj.com',
                       '@jpmorgan.com', '@mcdonalds.com', '@merck.com',
                       '@microsoft.com', '@nike.com', '@pfizer.com', '@pg.com',
                       '@travelers.com', '@unitedtech.com', '@unitedhealth.com',
                       '@verizon.com', '@visa.com', '@walmart.com']
personal_domains  = ['@gmail.com', '@yahoo.com', '@protonmail.com']
junk_domains = ['@me.com', '@aol.com', '@hotmail.com', '@live.com', '@msn.com',
                '@passport.com']

# placeholder list
placeholder_lst = []

# looping to group observations by domain type
for domain in original_df['email_domain']:
        if '@' + domain in professional_domains:
            placeholder_lst.append('professional')
        elif '@' + domain in personal_domains:
            placeholder_lst.append('personal')
        elif '@' + domain in junk_domains:
            placeholder_lst.append('junk')
        else:
            print('Unknown')


# concatenating with original DataFrame
original_df['domain_group'] = pd.Series(placeholder_lst)

# checking results
original_df['domain_group'].value_counts()


# One Hot encoding the categorical variable 'domain_group'

one_hot_domain_group = pd.get_dummies(original_df['domain_group'])

# dropping categorical variables after they've been encoded
original_df = original_df.drop('domain_group', axis = 1)

# joining codings together
original_df = original_df.join([one_hot_domain_group])

# Declaring explanatory variables:
original_df_data = original_df.drop(['CROSS_SELL_SUCCESS', 'NAME', 'EMAIL', 
                                     'FIRST_NAME', 'FAMILY_NAME', 'email_domain'],
                            axis = 1)

# Declaring response variable:
original_df_response = original_df.loc[:, 'CROSS_SELL_SUCCESS']


################################################################################
# Train/Test Split
################################################################################

# INSTANTIATING StandardScaler()
scaler = StandardScaler()


# FITTING the data
scaler.fit(original_df_data)


# TRANSFORMING the data
X_scaled     = scaler.transform(original_df_data)


# converting to a DataFrame
X_scaled_df  = pd.DataFrame(X_scaled) 


# train-test-split with the scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
            X_scaled_df,
            original_df_response,
            random_state = 222,
            test_size = 0.25,
            stratify = original_df_response)


################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# INSTANTIATING a Decision Tree Classifier object, but with specific parameters 
tree_pruned = DecisionTreeClassifier(max_depth = 3,
                      random_state = 222)

# FITTING the training data
tree_pruned_fit = tree_pruned.fit(X_train_scaled, y_train_scaled)

# PREDICTING on new data
tree_pruned_pred = tree_pruned_fit.predict(X_test_scaled)


################################################################################
# Final Model Score (score)
################################################################################

# SCORING the results
test_score = roc_auc_score(y_true  = y_test_scaled,
                                 y_score = tree_pruned_pred).round(4)
"""
    
# calculating execution time
elapsed_time = timeit.timeit(code_to_test, number=3)/3
print(elapsed_time)