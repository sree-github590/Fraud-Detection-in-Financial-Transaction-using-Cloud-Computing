# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests


from sklearn.metrics import  confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix, auc
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler, PowerTransformer

import streamlit as st

import warnings
warnings.filterwarnings("ignore")
def check_auth():
    response = requests.get("http://localhost:5000/check_auth")
    return response.json().get("authenticated", False)

# Redirect to login page if not authenticated
if not check_auth():
    st.warning("You are not authenticated. Please log in.")
    st.stop()


# Title
st.title('Bank Card Fraud Detection')


# Read Data into a Dataframe
df = pd.read_csv('creditcard.csv')


# --- 1 CHECKBOX ---
# Print description of the initial data and shape
if st.sidebar.checkbox('Show the initial data set'):
    st.header("Understanding dataset")

    st.write('Initial data set: \n', df)
    st.write('Data decription: \n', df.describe())
    st.write('Shape of the dataframe: ',df.shape)
    st.text('The dataset consists of 284,807 rows and 31 columns.\nThere is no zero value in the data.')

    st.header("Checking missing and outlier values")

    # Check missing values
    st.write('Missing values: ', df.isnull().values.sum())

    # Checking the number of missing values in each column
    st.write('The number of missing values in each column: ', df.isnull().sum())

    # Percentage of null values
    percent_missing = (df.isnull().sum().sort_values(ascending = False) / len(df)) * 100
    st.write('Percentage of null values: ', percent_missing)

    # Check if there are any duplicate rows
    st.write('Duplicate rows: ', df.duplicated(keep=False).sum())

    # Delete duplicate rows
    df = df.drop_duplicates() 
    st.write('Deleting duplicate rows was successful. This is a new data set:', df)
# --- 1 CHECKBOX ---



# --- 2 CHECKBOX ---
if st.sidebar.checkbox('Show the analysis'):
    
    fraud = df[df.Class == 1]
    valid = df[df.Class == 0]

    outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100

    st.header('Univariate analysis')

    st.write('Fraud Cases: ', len(fraud))
    st.write('Valid Cases: ', len(valid))
    st.write('Compare the values for both transactions: \n', df.groupby('Class').mean())
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)


    # Method to compute countplot of given dataframe parameters:
    # - data(pd.Dataframe): Input Dataframe
    # - feature(str): Feature in Dataframe
    def countplot_data(data, feature):
        plt.figure(figsize=(10,10))
        sns.countplot(x=feature, data=data)
        plt.show()

    # Method to construct pairplot of the given feature wrt data parameters:
    # - data(pd.DataFrame): Input Dataframe
    # - feature1(str): First Feature for Pair Plot
    # - feature2(str): Second Feature for Pair Plot
    # - target: Target or Label (y)
    def pairplot_data_grid(data, feature1, feature2, target):
        sns.FacetGrid(data, hue=target).map(plt.scatter, feature1, feature2).add_legend()
        plt.show()

    st.subheader('Transaction ratio:')
    st.pyplot(countplot_data(df, df.Class))

    st.subheader('The relationship of fraudulent transactions with the amount of money:\n')
    st.pyplot(pairplot_data_grid(df, "Time", "Amount", "Class"))
    


    st.header('Bivariate Analysis')
    
    st.write('Fraud: ', df.Time[df.Class == 1].describe())
    st.write('Not fraud: ', df.Time[df.Class == 0].describe())

    
    def graph1():
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
        bins = 50

        ax1.hist(df.Time[df.Class == 1], bins = bins)
        ax1.set_title('Fraud')

        ax2.hist(df.Time[df.Class == 0], bins = bins)
        ax2.set_title('Not Fraud')

        plt.xlabel('Time (Sec.)')
        plt.ylabel('Number of Transactions')
        plt.show()


    def graph2():
        f, axes = plt.subplots(ncols=2, figsize=(16,10))
        colors = ['#C35617', '#FFDEAD']

        sns.boxplot(x="Class", y="Amount", data=df, palette = colors, ax=axes[0], showfliers=True)
        axes[0].set_title('Class vs Amount')

        sns.boxplot(x="Class", y="Amount", data=df, palette = colors, ax=axes[1], showfliers=False)
        axes[1].set_title('Class vs Amount without outliers')

        plt.show()

    
    def graph3():
        fig, ax = plt.subplots(1, 2, figsize=(18,4))

        amount_val = df['Amount'].values
        time_val = df['Time'].values

        sns.distplot(amount_val, ax=ax[0], color='b')
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
        ax[0].set_xlim([min(amount_val), max(amount_val)])

        sns.distplot(time_val, ax=ax[1], color='r')
        ax[1].set_title('Distribution of Transaction Time', fontsize=14)
        ax[1].set_xlim([min(time_val), max(time_val)])

        plt.show()


    st.pyplot(graph1())
    st.pyplot(graph2())
    st.pyplot(graph3())



    st.header('Multivariate Analysis')


    # Plot relation with different scale
    def graph4(): 
        df1 = df[df['Class']==1]
        df2 = df[df['Class']==0]
        fig, ax = plt.subplots(1,2, figsize=(15, 5))

        ax[0].scatter(df1['Time'], df1['Amount'], color='red', marker= '*', label='Fraudrent')
        ax[0].set_title('Time vs Amount')
        ax[0].legend(bbox_to_anchor =(0.25, 1.15))

        ax[1].scatter(df2['Time'], df2['Amount'], color='green', marker= '.', label='Non Fraudrent')
        ax[1].set_title('Time vs Amount')
        ax[1].legend(bbox_to_anchor =(0.3, 1.15))

        plt.show()


    def graph5():
        sns.lmplot(x='Time', y='Amount', hue='Class', markers=['x', 'o'], data=df, height=6)
    

    # plot relation in same scale
    def graph6():
        g = sns.FacetGrid(df, col="Class", height=6)
        g.map(sns.scatterplot, "Time", "Amount", alpha=.7)
        g.add_legend()
    

    st.pyplot(graph4())
    st.pyplot(graph5())
    st.pyplot(graph6())  
# --- 2 CHECKBOX ---


# --- 3 CHECKBOX ---
if st.sidebar.checkbox('Model building on imbalanced data'):
    # --- TRAIN AND TEST SPLIT ---
    st.header('Train and test split')


    # Putting feature variables into X
    X = df.drop(['Class'], axis=1)

    # Putting target variable to y
    y = df['Class']


    # Splitting data into train and test set 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)

    st.write('X_train: ', X_train.shape)
    st.write('y_train: ', y_train.shape)
    st.write('X_test: ', X_test.shape)
    st.write('y_test: ', y_test.shape)
    # --- TRAIN AND TEST SPLIT ---


    
    # --- FEATURE SCALING ---
    # Instantiate the Scaler
    scaler = StandardScaler()

    # Fit the data into scaler and transform
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])


    # Transform the test set
    X_test['Amount'] = scaler.transform(X_test[['Amount']])


    # Checking the Skewness
    # Listing the columns
    cols = X_train.columns
    
    
    # Plotting the distribution of the variables (skewness) of all the columns
    def skewness(): 
        k = 0
        plt.figure(figsize=(17,28))
        for col in cols :    
            k = k + 1
            plt.subplot(6, 5,k)    
            sns.distplot(X_train[col])
            plt.title(col+' '+str(X_train[col].skew()))
    

    st.header('Checking the Skewness')
    st.pyplot(skewness())
    # --- FEATURE SCALING ---



    # --- Mitigate skwenes with PowerTransformer ---
    # Instantiate the powertransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)

    # Fit and transform the PT on training data
    X_train[cols] = pt.fit_transform(X_train)

    # Transform the test set
    X_test[cols] = pt.transform(X_test)

    
    def newSkewness():
        k=0
        plt.figure(figsize=(17,28))
        for col in cols :    
            k=k+1
            plt.subplot(6, 5,k)    
            sns.distplot(X_train[col])
            plt.title(col+' '+str(X_train[col].skew()))
    

    st.header('Mitigate skwenes with PowerTransformer')
    st.pyplot(newSkewness())
    # --- Mitigate skwenes with PowerTransformer ---   
# --- 3 CHECKBOX ---



# --- 4 CHECKBOX ---
if st.sidebar.checkbox('Analysis of algorithms'):
    # --- TRAIN AND TEST SPLIT ---
    # Putting feature variables into X
    X = df.drop(['Class'], axis=1)

    # Putting target variable to y
    y = df['Class']


    # Splitting data into train and test set 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)
    # --- TRAIN AND TEST SPLIT ---


    # --- FEATURE SCALING ---
    # Instantiate the Scaler
    scaler = StandardScaler()


    # Fit the data into scaler and transform
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])


    # Transform the test set
    X_test['Amount'] = scaler.transform(X_test[['Amount']])


    # Checking the Skewness
    # Listing the columns
    cols = X_train.columns
    # --- FEATURE SCALING ---


    # --- Mitigate skwenes with PowerTransformer ---
    # Instantiate the powertransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)

    # Fit and transform the PT on training data
    X_train[cols] = pt.fit_transform(X_train)

    # Transform the test set
    X_test[cols] = pt.transform(X_test)
    # --- Mitigate skwenes with PowerTransformer ---


    def visualize_confusion_matrix(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Oranges',
                    xticklabels=['No Credit Card Fraud Dection','Credit Card Fraud Dection'], 
                    yticklabels=['No Credit Card Fraud Dection','Credit Card Fraud Dection'])
        plt.title('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
        plt.ylabel('True Values')
        plt.xlabel('Predicted Values')
        plt.show()
        
        st.write("\n")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        return

    
    def ROC_AUC(Y, Y_prob):
        # caculate roc curves
        fpr, tpr, threshold = roc_curve(Y, Y_prob)
        # caculate scores
        model_auc = roc_auc_score(Y, Y_prob)
        # plot roc curve for the model
        plt.figure(figsize=(16, 9))
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='Model - AUC=%.3f' % (model_auc))
        # show axis labels and the legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show(block=False)
        return



    
    
    # --- START Decision tree ---
    st.header('Decision tree')

    
    # --- START Training the Decision tree Model on the Training set ---
    st.subheader('Training the Decision tree Model on the Training set')


    DTR_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    DTR_model.fit(X_train, y_train)
    y_train_pred = DTR_model.predict(X_train)
    y_test_pred = DTR_model.predict(X_test)
    acc4 = accuracy_score(y_test, y_test_pred)


    # Train Score
    st.write('Recall score: %0.4f'% recall_score(y_train, y_train_pred))
    st.write('Precision score: %0.4f'% precision_score(y_train, y_train_pred))
    st.write('F1-Score: %0.4f'% f1_score(y_train, y_train_pred))
    st.write('Accuracy score: %0.4f'% accuracy_score(y_train, y_train_pred))
    st.write('AUC: %0.4f' % roc_auc_score(y_train, y_train_pred))


    st.pyplot(visualize_confusion_matrix(y_train, y_train_pred))

    st.pyplot(ROC_AUC(y_train, y_train_pred))
    # --- END Training the Decision tree Model on the Training set ---


    # --- START Training the Decision tree Model on the Testing set ---
    st.subheader('Training the Decision tree Model on the Testing set')


    st.write('Recall score: %0.4f'% recall_score(y_test, y_test_pred))
    st.write('Precision score: %0.4f'% precision_score(y_test, y_test_pred))
    st.write('F1-Score: %0.4f'% f1_score(y_test, y_test_pred))
    st.write('Accuracy score: %0.4f'% accuracy_score(y_test, y_test_pred))
    st.write('AUC: %0.4f' % roc_auc_score(y_test, y_test_pred))


    st.pyplot(visualize_confusion_matrix(y_test, y_test_pred))

    st.pyplot(ROC_AUC(y_test, y_test_pred))
    # --- END Training the Decision tree Model on the Testing set ---


    # Result
    st.header('Results')
    st.subheader('Training set')
    st.text('- Recall score: 1.0000\n- Precision score: 1.0000\n- F1-Score: 1.0000\n- Accuracy score: 1.0000\n- AUC: 1.0000')
    

    st.subheader('Testing set')
    st.text('- Recall score: 0.6889\n- Precision score: 0.7561\n- F1-Score: 0.7209\n- Accuracy score: 0.9992\n- AUC: 0.8443')
    # --- END Decision tree ---

# --- 4 CHECKBOX ---



# --- 5 CHECKBOX ---
if st.sidebar.checkbox('Manual transaction verification'):

    # separate legitimate and fraudulent transactions
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]

    # undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    # split data into training and testing sets
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # train logistic regression model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # evaluate model performance
    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)

    # create Streamlit app
    st.title("Manual transaction verification")
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    # create input fields for user to enter feature values
    input_df = st.text_input('Input All features')
    input_df_lst = input_df.split(',')
    # create a button to submit input and get prediction
    submit = st.button("Submit")

    if submit:
        # get input feature values
        features = np.array(input_df_lst, dtype=np.float64)
        # make prediction
        prediction = model.predict(features.reshape(1,-1))
        # display result
        if prediction[0] == 0:
            st.write("Legitimate transaction")
        elif prediction[0] == 1:
            st.write("Fraudulent transaction")
# --- 5 CHECKBOX --