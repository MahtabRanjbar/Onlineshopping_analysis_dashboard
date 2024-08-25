import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set up page configuration
st.set_page_config(layout="wide", page_title="Online Shoppers EDA")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('online_shoppers_intention.csv')
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
sections = st.sidebar.radio(
    "Select Section",
    ["Overview", "Revenue Distribution", "Category Analysis", "Visitor Behavior", "Traffic Analysis", "Summary Stats"]
)

# Define the top_values function
def top_values(df, column, top=5, all_values=False):
    if all_values:
        slices = df[column].value_counts().values
        labels = df[column].value_counts().index.values.astype(str)
    else:
        top_values = df[column].value_counts()[:top]
        sum_others = sum(df[column].value_counts()[top:])
        slices = np.append(top_values.values, sum_others)
        labels = np.append(top_values.index.values.astype(str), 'Other')
    return slices, labels

# 1. Overview Section
if sections == "Overview":
    st.title("Online Shoppers Intention EDA")
    st.write("This dashboard provides an exploratory data analysis (EDA) of online shoppers' behavior. It includes visualizations to understand customer revenue, browsing behavior, and purchase patterns.")
    
    st.header("Dataset Preview")
    st.dataframe(df.head())
    
    st.subheader("Dataset Information")
    buffer = st.empty()
    buffer.code(df.info(buf=None))

# 2. Revenue Distribution Section
elif sections == "Revenue Distribution":
    st.title("Revenue Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue and Weekend Purchase Distribution")
        fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
        sns.countplot(x='Revenue', palette='pastel', data=df, ax=axs[0])
        axs[0].set_title('Buy or Not', fontsize=20)
        axs[0].set_xlabel('Revenue or Not', fontsize=15)
        axs[0].set_ylabel('Count', fontsize=15)

        sns.countplot(x='Weekend', palette='inferno', data=df, ax=axs[1])
        axs[1].set_title('Purchase on Weekends', fontsize=20)
        axs[1].set_xlabel('Weekend or Not', fontsize=15)
        axs[1].set_ylabel('Count', fontsize=15)
        st.pyplot(fig)

    with col2:
        st.subheader("Traffic and Regional Distribution")
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        sns.countplot(x='TrafficType', palette='inferno', data=df, ax=axs[0])
        axs[0].set_title('Distribution of Different Traffics', fontsize=20)
        axs[0].set_xlabel('TrafficType Codes', fontsize=15)
        axs[0].set_ylabel('Count', fontsize=15)

        sns.countplot(x='Region', palette='inferno', data=df, ax=axs[1])
        axs[1].set_title("Distribution of User's Location", fontsize=20)
        axs[1].set_xlabel('Region Codes', fontsize=15)
        axs[1].set_ylabel('Count', fontsize=15)
        st.pyplot(fig)

# 3. Category Analysis Section
elif sections == "Category Analysis":
    st.title("Category Analysis")

    fig, axs = plt.subplots(2, 2, figsize=(25, 15))
    
    # Browser Analysis
    slices, labels = top_values(df, "Browser")
    colors = ["orange", "yellow", "pink", "crimson", "lightgreen", "cyan", "blue"]
    axs[0, 0].pie(slices, colors=colors, labels=labels, shadow=True, startangle=90, rotatelabels=True)
    axs[0, 0].set_title("Different Browsers", fontsize=20)
    labels = [f"{i} - {j:1.1f}%" for i, j in zip(labels, slices * 100 / slices.sum())]
    axs[0, 0].legend(labels, loc="best")

    # Operating Systems Analysis
    slices, labels = top_values(df, "OperatingSystems", top=3)
    colors = ["violet", "magenta", "pink", "blue"]
    axs[0, 1].pie(slices, colors=colors, labels=labels, shadow=True, autopct="%.1f%%", startangle=90)
    axs[0, 1].set_title("Different Operating Systems", fontsize=20)
    axs[0, 1].legend()

    # Month Analysis
    slices, labels = top_values(df, 'Month', all_values=True)
    colors = ['yellow', 'pink', 'lightblue', 'crimson', 'lightgreen', 'orange', 'cyan', 'magenta', 'violet', 'pink', 'lightblue', 'red']
    axs[1, 0].pie(slices, colors=colors, labels=labels, explode=[0]*10, shadow=True, autopct='%.2f%%')
    axs[1, 0].set_title('Month', fontsize=30)
    axs[1, 0].legend(loc='upper left')

    # Visitor Type Analysis
    slices, labels = top_values(df, 'VisitorType', all_values=True)
    colors = ['lightGreen', 'green', 'pink']
    axs[1, 1].pie(slices, colors=colors, labels=labels, explode=[0, 0, 0.1], shadow=True, autopct='%.2f%%')
    axs[1, 1].set_title('Different Visitors', fontsize=30)
    axs[1, 1].legend()

    st.pyplot(fig)

# 4. Visitor Behavior Section
elif sections == "Visitor Behavior":
    st.title("Visitor Behavior Analysis")
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Administrative Duration vs Revenue
    sns.boxenplot(x='Administrative_Duration', y='Revenue', palette='pastel', orient='h', data=df, ax=axs[0, 0])
    axs[0, 0].set_title('Admin. Duration vs Revenue', fontsize=20)
    axs[0, 0].set_xlabel('Admin. Duration', fontsize=15)
    axs[0, 0].set_ylabel('Revenue', fontsize=15)

    # Informational Duration vs Revenue
    sns.boxenplot(x='Informational_Duration', y='Revenue', data=df, palette='rainbow', orient='h', ax=axs[0, 1])
    axs[0, 1].set_title('Info. Duration vs Revenue', fontsize=20)
    axs[0, 1].set_xlabel('Info. Duration', fontsize=15)
    axs[0, 1].set_ylabel('Revenue', fontsize=15)

    # Product Related Duration vs Revenue
    sns.boxenplot(x='ProductRelated_Duration', y='Revenue', data=df, palette='inferno', orient='h', ax=axs[1, 0])
    axs[1, 0].set_title('Product Related Duration vs Revenue', fontsize=20)
    axs[1, 0].set_xlabel('Product Related Duration', fontsize=15)
    axs[1, 0].set_ylabel('Revenue', fontsize=15)

    # Exit Rates vs Revenue
    sns.boxenplot(x='ExitRates', y='Revenue', data=df, palette='dark', orient='h', ax=axs[1, 1])
    axs[1, 1].set_title('Exit Rates vs Revenue', fontsize=20)
    axs[1, 1].set_xlabel('Exit Rates', fontsize=15)
    axs[1, 1].set_ylabel('Revenue', fontsize=15)

    st.pyplot(fig)

# 5. Traffic Analysis Section
elif sections == "Traffic Analysis":
    st.title("Traffic and Visitor Analysis")
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), sharey=True)
    
    # Page Values vs Revenue
    sns.stripplot(x='PageValues', y='Revenue', data=df, palette='spring', orient='h', ax=axs[0, 0])
    axs[0, 0].set_title('Page Values vs Revenue', fontsize=20)
    axs[0, 0].set_xlabel('Page Values', fontsize=15)
    axs[0, 0].set_ylabel('Revenue', fontsize=15)

    # Bounce Rates vs Revenue
    sns.stripplot(x='BounceRates', y='Revenue', data=df, palette='autumn', orient='h', ax=axs[0, 1])
    axs[0, 1].set_title('Bounce Rates vs Revenue', fontsize=20)
    axs[0, 1].set_xlabel('Bounce Rates', fontsize=15)
    axs[0, 1].set_ylabel('Revenue', fontsize=15)

    # Page Values vs Revenue (Boxplot)
    sns.boxplot(x='PageValues', y='Revenue', data=df, palette='spring', orient='h', ax=axs[1, 0])
    axs[1, 0].set_title('Page Values vs Revenue (Boxplot)', fontsize=20)
    axs[1, 0].set_xlabel('Page Values', fontsize=15)
    axs[1, 0].set_ylabel('Revenue', fontsize=15)

    # Bounce Rates vs Revenue (Boxplot)
    sns.boxplot(x='BounceRates', y='Revenue', data=df, palette='autumn', orient='h', ax=axs[1, 1])
    axs[1, 1].set_title('Bounce Rates vs Revenue (Boxplot)', fontsize=20)
    axs[1, 1].set_xlabel('Bounce Rates', fontsize=15)
    axs[1, 1].set_ylabel('Revenue', fontsize=15)

    st.pyplot(fig)

# 6. Summary Stats Section
elif sections == "Summary Stats":
    st.title("Summary Statistics and Pair Plots")

    st.subheader("Distribution Plots")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(df['BounceRates'], bins=20, ax=ax1, kde=True)
    ax1.set_title('Bounce Rates')
    
    sns.histplot(df['ExitRates'], bins=20, ax=ax2, kde=True)
    ax2.set_title('Exit Rates')
    
    sns.histplot(df['PageValues'], bins=20, ax=ax3, kde=True)
    ax3.set_title('Page Values')
    
    st.pyplot(fig)
    
    st.subheader("Monthly Revenue vs Visitor Type")
    fig, ax = plt.subplots(figsize=(18, 12))
    orderlist = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sns.countplot(data=df, x='Month', hue='Revenue', ax=ax, order=orderlist)
    ax.set_title("Monthly Revenue vs Visitor Type")
    st.pyplot(fig)
    
    st.subheader("Pair Plot of Bounce and Exit Rates")
    pairplot_fig = sns.pairplot(df, x_vars=['BounceRates', 'ExitRates'], y_vars=['BounceRates', 'ExitRates'], hue='Revenue', diag_kind='kde')
    st.pyplot(pairplot_fig.fig)


