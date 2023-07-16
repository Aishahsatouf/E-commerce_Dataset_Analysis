import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib as mpl
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


# Set the default font settings for matplotlib
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "Arial"   
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1

# Define a helper function for creating default plots
def default_plot(ax, spines): 
    ax = plt.gca()
    # Remove unnecessary axes and ticks (top and right)
    ax.spines["top"].set_visible(False)   
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    
    # Set the ticks facing outward
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    
   # Adjust the position of spines  
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points

     # Turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        
    if 'right' in spines:
        ax.yaxis.set_ticks_position('right')

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')

    return ax

# Load the datasets
order_items    = pd.read_csv("data/olist_order_items_dataset.csv")
orders         = pd.read_csv("data/olist_orders_dataset.csv")
order_payments = pd.read_csv("data/olist_order_payments_dataset.csv")
products       = pd.read_csv("data/olist_products_dataset.csv")
customers      = pd.read_csv("data/olist_customers_dataset.csv")
sellers        = pd.read_csv("data/olist_sellers_dataset.csv")
reviews        = pd.read_csv("data/olist_order_reviews_dataset.csv")
product_category_translation = pd.read_csv("data/product_category_name_translation.csv")



# Merge the datasets into a consolidated dataset
merged = order_items.merge(orders, on='order_id') \
                    .merge(order_payments, on=['order_id']) \
                    .merge(products, on='product_id') \
                    .merge(customers, on='customer_id') \
                    .merge(sellers, on='seller_id') \
                    .merge(product_category_translation, on='product_category_name')\
                    .merge(reviews, on="order_id")
                     
# Save the consolidated dataset to a CSV file
merged.to_csv('data/brazilian_ecommerce_dataset.csv', index=False)

# Print the statistical summary of the consolidated dataset
print(merged.describe())

# Load the dataset from the CSV file
df = pd.read_csv("brazilian_ecommerce_dataset.csv")

#######################################################################################
# Preprocess the data

# Remove outliers from the 'price' column
df = df[(df['price'] >= df['price'].quantile(0.05)) & (df['price'] <= df['price'].quantile(0.95))]
### getting the demand values############
# Extract the year from 'order_purchase_timestamp' and assign it to 'Year' column
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['Year'] = df['order_purchase_timestamp'].dt.year
# Create a new column for the month of the order
df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')


# Calculate the sum of product quantity ordered per month
monthly_demand = df.groupby(['product_id', 'order_month'])['order_item_id'].sum().reset_index()
# Merge the monthly demand back to the original dataset
df = pd.merge(df, monthly_demand, on=['product_id', 'order_month'], how='left')
# Rename the new column as 'demand'
df.rename(columns={'order_item_id_y': 'demand'}, inplace=True)
# Drop unnecessary columns
df.drop(['order_item_id_x'], axis=1, inplace=True)



#####################################################################################################################################
# Perform price, freight, and review score comparison

# Perform left join operations
import plotly.express as px
# Perform left join operations
products_data_english = pd.merge(products, product_category_translation, on="product_category_name", how="left")
products_data_eng_items = pd.merge(order_items, products_data_english, on="product_id", how="left")

# Filter, group by, and select top 10
top10_product = products_data_eng_items[~products_data_eng_items["product_category_name_english"].isna()] \
    .groupby("product_category_name_english") \
    .size() \
    .sort_values(ascending=False) \
    .head(10) \
    .reset_index(name="count")

# Access the top 10 product category names
top10_product_category_names = top10_product["product_category_name_english"].tolist()
# Perform left join operations
products_details1 = pd.merge(products, product_category_translation , on="product_category_name", how="left")
products_orders_details1 = pd.merge(order_items , products_details1, on="product_id", how="left")

# Calculate volume
products_orders_details1["volume"] = (
    products_orders_details1["product_length_cm"] * 
    products_orders_details1["product_height_cm"] * 
    products_orders_details1["product_width_cm"]
)

products_orders_items1 = pd.merge(products_orders_details1, orders, on="order_id", how="left")
products_orders_items1 = pd.merge(products_orders_items1, customers, on="customer_id", how="left")
products_orders_items1 = pd.merge(products_orders_items1, sellers, on="seller_id", how="left")
products_orders_items1 = pd.merge(products_orders_items1, reviews, on="order_id", how="left")

# Filter and group by
top_10_product_category_names = top10_product["product_category_name_english"].tolist()
products_orders_items1["product_category_name_english"] = products_orders_items1["product_category_name_english"].apply(
    lambda x: x if x in top_10_product_category_names else "Others"
)

p5 = products_orders_items1.groupby(["product_category_name_english", "review_score"]).agg(
    fv=("freight_value", "mean"),
    pv=("price", "mean")
).reset_index()

# Create a 3D scatter plot for price ,Catgory name of the products & the review scores
fig = px.scatter_3d(
    p5, x="review_score", y="pv", z="fv", color="product_category_name_english",
    color_discrete_map={"Others": "red"}, height=900, width=1500
)

fig.update_layout(
    scene=dict(
        xaxis=dict(title="Mean Review Score"),
        yaxis=dict(title="Mean Price"),
        zaxis=dict(title="Mean Freight Value")
    ),
    xaxis=dict(title="State"),
    yaxis=dict(title="Delivery days")
)

# Uncomment the line below to display the plot
# fig.show()

###the prices and the frieght values will differ upon the differinces in  product's categories so mainly thers might be a relation between prices and categories of the products which leads eventually to different satisfaction levels
#############################################################################################################################################
# ####################################################################################################################################
# Analyze orders variation throughout the year:
# Convert order_purchase_timestamp to date and find the minimum and maximum order dates
orders.order_purchase_timestamp = pd.to_datetime(orders.order_purchase_timestamp).dt.date
min_order_date = orders.order_purchase_timestamp.min()
max_order_date = orders.order_purchase_timestamp.max()
print('First registered order: ', min_order_date)
print('Last registered order: ', max_order_date)

# Create a list of days between the beginning and the end of the dataset
order_dates = [min_order_date + dt.timedelta(days= i) for i in range((max_order_date - min_order_date).days)]

# Convert dates to numerical values for creating a histogram
to_timestamp = np.vectorize(lambda x: (x - dt.date(1970, 1, 1)).total_seconds())
time_stamps = to_timestamp(orders.order_purchase_timestamp)
order_histogram = np.histogram(time_stamps, bins = len(order_dates))[0]

# Create a time series plot of the number of daily orders
ax = plt.subplots()
ax = default_plot(ax, ['bottom', 'left'])
ax.set_ylabel('Orders')
ax.set_xlabel('Date')
plt.ylim(0, 1200)
plt.xticks(rotation= 30)
dates_and_counts = list(zip(order_dates, order_histogram))
ax.scatter(max(dates_and_counts, key=lambda x: x[1])[0], max(dates_and_counts, key=lambda x: x[1])[1], marker='o', color='r')
ax.text(max(dates_and_counts, key=lambda x: x[1])[0], max(dates_and_counts, key=lambda x: x[1])[1] + 20, 'Highest Demand')
ax.plot(order_dates, order_histogram, 'k', lw=0.75);
plt.tight_layout()
plt.show()
plt.savefig('orders_timeseries.png', dpi= 300)
###Clearly, the sale increases significantly when there is an event during a particular month.

###########################################################################################################################################
# Scattered plot
# Create a scatter plot of demand vs price
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
## the figure with demand square values and the prices values is giving a plot that is not well defined so I used the logarithmic  to get rid of the outliers 
# demand_sequare =np.square(df['demand'])
# price_col = df.price 
demand_log=np.log(df['demand'])
price_log=np.log(df['price'])
# Create scatter plot
fig, ax = plt.subplots()
ax.scatter(demand_log,df.price, c='black')
line = mlines.Line2D([-1,0], [-1, 0], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
# Add labels and title
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Price VS Quantity')
plt.show()

##the demand and prices are in  an exponential  relation  
##############################################################################################################################################
#training preprocessing
# Clean the dataset for model training
df = df.drop(['order_id','shipping_limit_date','customer_id','seller_id', 'freight_value','order_purchase_timestamp','order_month','order_status','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','customer_unique_id','customer_city','seller_zip_code_prefix','review_id','review_comment_title','review_comment_message','review_creation_date','review_answer_timestamp'], axis=1)
df = df.dropna()
print(f"the describtive analysis of the data frame:\n {df.describe()}")
from sklearn.preprocessing import LabelEncoder
##### Convert categorical variables to numerical values
# Convert categorical variables to numerical values using LabelEncoder
label_encoder = LabelEncoder()

# Iterate over each column in the DataFrame
for column in df.columns:
    # Check if the column has object/string datatype
    if df[column].dtype == 'object':
        # Apply Label Encoding to convert string values to numeric labels
        df[column] = label_encoder.fit_transform(df[column])
# Print the descriptive analysis of the cleaned dataset
print('cleaned data description \n')
print(df.describe())



##########################################################################################################################################
# Get the correlation heatmap to identify important features
correlation = df[['price','payment_sequential','payment_type','payment_installments','payment_value','product_name_lenght','product_description_lenght','product_photos_qty','product_weight_g','product_length_cm','product_height_cm','product_width_cm','customer_zip_code_prefix','seller_city','seller_state','product_category_name_english','review_score','Year','demand']] 
mask = np.triu(np.ones_like(correlation.corr(), dtype=bool))
heatmap = sns.heatmap(correlation.corr(), vmin=-1 , vmax=1, annot=True, mask=mask,square=True,annot_kws={'fontsize':9}) 
plt.show()
#########################################################################################################################################

# Train a linear regression model and print its summary
from sklearn import linear_model
reg = linear_model.LinearRegression()
import statsmodels.formula.api as smf
reg_model='price~payment_sequential+payment_type+payment_installments+payment_value+product_description_lenght+product_photos_qty+product_weight_g+product_length_cm+product_height_cm+product_width_cm+customer_zip_code_prefix+seller_city+seller_state+product_category_name_english+review_score+Year+demand'
reg_output=smf.ols(reg_model,df).fit()
print(reg_output.summary())

##########################################################################################################################################
# Split the data into training and testing sets
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(df.drop('price', axis=1), df['price'], test_size=0.2, random_state=42)

# Convert PeriodArray to numerical representation
X_train_numeric = X_train.astype('int64').values
X_test_numeric=X_test.astype('int64').values

# Train a Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_numeric, y_train)

# Make predictions on the test set and evaluate the model
y_pred = rf.predict(X_test_numeric)
mae = mean_absolute_error(y_test, y_pred)
print('Random Forest Model:\nThe accuracy_score function is suitable for classification tasks, where the predicted values are discrete labels or categories. Since what we have here is a regression task, we should use evaluation metrics \ndesigned for continuous variables, such as mean squared error (MSE), mean absolute error (MAE), or R-squared.')
print("The Random Forest Mean absolute error:\n", mae)
# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared score For the Random Forest:", r2)

#####################################################################################################################################################################




