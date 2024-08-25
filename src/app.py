import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Set the page configuration for Streamlit
st.set_page_config(layout="wide", page_title="Online Shoppers EDA", initial_sidebar_state="collapsed")

# Function to load data with caching to improve performance
@st.cache_data
def load_data():
    """Load the dataset from a CSV file."""
    return pd.read_csv('data/online_shoppers_intention.csv')


# Load the dataset
df = load_data()

# Title and description for the dashboard
st.title("Online Shoppers Purchase Intention Analysis")
st.markdown("""
This dashboard presents an exploratory data analysis (EDA) of online shoppers' purchase intentions.
Navigate through the tabs to explore different aspects of the dataset.
""")


# Function to get top values of a column with optional aggregation of others
def top_values(df, column, top=5, all_values=False):
    """
    Returns slices and labels for a pie chart based on the value counts of a column.

    Parameters:
    - df: DataFrame containing the data.
    - column: Column name to analyze.
    - top: Number of top values to show.
    - all_values: Whether to include all values or group others.

    Returns:
    - slices: Values for the pie chart.
    - labels: Labels for the pie chart.
    """
    if all_values:
        slices = df[column].value_counts().values
        labels = df[column].value_counts().index.values.astype(str)
    else:
        top_values = df[column].value_counts()[:top]
        sum_others = sum(df[column].value_counts()[top:])
        slices = np.append(top_values.values, sum_others)
        labels = np.append(top_values.index.values.astype(str), 'Other')
    return slices, labels


# Define tabs for different sections of the dashboard
tabs = st.tabs([
    "Overview",
    "Revenue & Visitor Analysis",
    "Traffic & Behavior Analysis",
    "Multivariate & Correlation Analysis",
    "Clustering Analysis",
    "Modeling Dashboard"
])

# Overview Tab
with tabs[0]:
    # Ensure correct data types
    df['Month'] = df['Month'].astype('category')
    df['VisitorType'] = df['VisitorType'].astype('category')
    df['Weekend'] = df['Weekend'].astype('bool')
    df['Revenue'] = df['Revenue'].astype('bool')

    # Overview container
    with st.container():
        st.title("Dataset Overview")

        # Interactive Summary Cards
        st.subheader("Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Unique Values", df.nunique().sum())

        # Column Summary in collapsible sections
        with st.expander("View Column Summary"):
            for col in df.columns:
                st.markdown(f"**{col}**")
                st.write(f"Data Type: {df[col].dtype}")
                st.write(f"Unique Values: {df[col].nunique()}")
                st.write(f"Missing Values: {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.2f}%)")
                st.markdown("---")

        # Data Preview with Filtering Options
        st.subheader("Data Preview")
        cols_to_display = st.multiselect("Select columns to display", options=df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[cols_to_display].head(10))

        # Filter numeric columns only
        numeric_cols_to_display = df[cols_to_display].select_dtypes(include=np.number).columns.tolist()

        # Column Descriptions with Dynamic Tooltips
        st.subheader("Column Descriptions")
        column_descriptions = {
            "Administrative": "Number of administrative pages visited by the user.",
            "Administrative_Duration": "Time spent on administrative pages.",
            "Informational": "Number of informational pages visited by the user.",
            "Informational_Duration": "Time spent on informational pages.",
            "ProductRelated": "Number of product-related pages visited by the user.",
            "ProductRelated_Duration": "Time spent on product-related pages.",
            "BounceRates": "The percentage of visitors who enter the site and then leave ('bounce') without continuing to view other pages.",
            "ExitRates": "The percentage of exits from the site that occurred from a specific page.",
            "PageValues": "The average value for a web page that a user visited before completing an e-commerce transaction.",
            "SpecialDay": "Closeness of the site visit to a special day (e.g., Mother’s Day).",
            "Month": "Month of the year.",
            "OperatingSystems": "Operating system used by the visitor.",
            "Browser": "Browser used by the visitor.",
            "Region": "Geographical region of the visitor.",
            "TrafficType": "Source of the traffic (e.g., direct, organic search, paid search, etc.).",
            "VisitorType": "Type of visitor (e.g., Returning Visitor, New Visitor).",
            "Weekend": "Indicates if the visit was on a weekend.",
            "Revenue": "Indicates whether the visit resulted in a transaction (purchase)."
        }

        # Create a DataFrame for column descriptions
        description_df = pd.DataFrame(list(column_descriptions.items()), columns=['Column Name', 'Description'])

        # Display the DataFrame with styling
        st.dataframe(description_df.style.set_properties(**{
            'text-align': 'left',
            'font-size': '14px',
            'color': 'white',
            'background-color': '#2e2e2e',
            'border-color': '#444'
        }).set_table_styles([{
            'selector': 'thead th',
            'props': [('background-color', '#444'), ('color', 'white'), ('font-weight', 'bold')]
        }]))


# 2. Revenue & Visitor Analysis Tab
with tabs[1]:
    # Define sub-tabs within this main tab
    tabs_bi = st.tabs([
        "Revenue Distribution",
        "Visitor Behavior",
        "Monthly & Visitor Insights",
        "Statistical Summaries"
    ])

    # 1. Revenue Distribution Tab
    with tabs_bi[0]:
        st.header("Overview of Purchase Behavior")
        st.columns(1)

       
        st.subheader("Revenue and Weekend Purchase Distribution")
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
        sns.countplot(x='Revenue', palette='pastel', data=df, ax=axs[0])
        axs[0].set_title('Buy or Not', fontsize=20)
        axs[0].set_xlabel('Revenue or Not', fontsize=15)
        axs[0].set_ylabel('Count', fontsize=15)

        sns.countplot(x='Weekend', palette='inferno', data=df, ax=axs[1])
        axs[1].set_title('Purchase on Weekends', fontsize=20)
        axs[1].set_xlabel('Weekend or Not', fontsize=15)
        axs[1].set_ylabel('Count', fontsize=15)
        st.pyplot(fig)




        st.header("Revenue Distribution")

        # Create subplots for pie charts
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))

        # Pie chart for Browsers
        slices, labels = top_values(df, "Browser")
        colors = ["orange", "yellow", "pink", "crimson", "lightgreen", "cyan", "blue"]
        axs[0, 0].pie(slices, colors=colors, labels=labels, shadow=True, startangle=90, rotatelabels=True)
        axs[0, 0].set_title("Different Browsers", fontsize=20)
        labels = ["{0} - {1:1.1f} %".format(i, j) for i, j in zip(labels, slices * 100 / slices.sum())]
        axs[0, 0].legend(labels, loc="best")

        # Pie chart for Operating Systems
        slices, labels = top_values(df, "OperatingSystems", top=3)
        colors = ["violet", "magenta", "pink", "blue"]
        axs[0, 1].pie(slices, colors=colors, labels=labels, shadow=True, autopct="%.1f%%", startangle=90)
        axs[0, 1].set_title("Different Operating Systems", fontsize=20)
        axs[0, 1].legend()

        # Pie chart for Monthly Distribution
        colors = ['yellow', 'pink', 'lightblue', 'crimson', 'lightgreen', 'orange', 'cyan', 'magenta', 'violet', 'pink', 'lightblue', 'red']
        slices, labels = top_values(df, 'Month', all_values=True)
        axs[1, 0].pie(slices, colors=colors, labels=labels, shadow=True, autopct='%.2f%%')
        axs[1, 0].set_title('Monthly Distribution', fontsize=30)
        axs[1, 0].legend(loc='upper left')

        # Pie chart for Visitor Types
        slices, labels = top_values(df, 'VisitorType', all_values=True)
        colors = ['lightGreen', 'green', 'pink']
        explode = [0, 0, 0.1]  # Slightly explode the third slice
        axs[1, 1].pie(slices, colors=colors, labels=labels, explode=explode, shadow=True, autopct='%.2f%%')
        axs[1, 1].set_title('Visitor Type Distribution', fontsize=30)
        axs[1, 1].legend()

        plt.tight_layout()
        st.pyplot(fig)

    # 2. Visitor Insights Tab
    with tabs_bi[1]:
        st.header("Visitor Behavior Insights")

        # Create subplots for boxen plots
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))

        # Boxen plot for Administrative Duration vs Revenue
        sns.boxenplot(x='Administrative_Duration', y='Revenue', palette='pastel', orient='h', data=df, ax=axs[0, 0])
        axs[0, 0].set_title('Administrative Duration vs Revenue', fontsize=20)
        axs[0, 0].set_xlabel('Administrative Duration', fontsize=15)
        axs[0, 0].set_ylabel('Revenue', fontsize=15)

        # Boxen plot for Informational Duration vs Revenue
        sns.boxenplot(x='Informational_Duration', y='Revenue', palette='rainbow', orient='h', data=df, ax=axs[0, 1])
        axs[0, 1].set_title('Informational Duration vs Revenue', fontsize=20)
        axs[0, 1].set_xlabel('Informational Duration', fontsize=15)
        axs[0, 1].set_ylabel('Revenue', fontsize=15)

        # Boxen plot for Product-Related Duration vs Revenue
        sns.boxenplot(x='ProductRelated_Duration', y='Revenue', palette='inferno', orient='h', data=df, ax=axs[1, 0])
        axs[1, 0].set_title('Product-Related Duration vs Revenue', fontsize=20)
        axs[1, 0].set_xlabel('Product-Related Duration', fontsize=15)
        axs[1, 0].set_ylabel('Revenue', fontsize=15)

        # Boxen plot for Exit Rates vs Revenue
        sns.boxenplot(x='ExitRates', y='Revenue', palette='dark', orient='h', data=df, ax=axs[1, 1])
        axs[1, 1].set_title('Exit Rates vs Revenue', fontsize=20)
        axs[1, 1].set_xlabel('Exit Rates', fontsize=15)
        axs[1, 1].set_ylabel('Revenue', fontsize=15)

        st.pyplot(fig)

    # 3. Traffic Analysis Tab
    with tabs_bi[2]:
        st.header("Traffic and Regional Analysis")

        # Create subplots for traffic and regional count plots
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Count plot for Traffic Type
        sns.countplot(x='TrafficType', palette='inferno', data=df, ax=axs[0])
        axs[0].set_title('Traffic Type Distribution', fontsize=20)
        axs[0].set_xlabel('Traffic Type', fontsize=15)
        axs[0].set_ylabel('Count', fontsize=15)

        # Count plot for Region
        sns.countplot(x='Region', palette='inferno', data=df, ax=axs[1])
        axs[1].set_title("User's Regional Distribution", fontsize=20)
        axs[1].set_xlabel('Region', fontsize=15)
        axs[1].set_ylabel('Count', fontsize=15)

        st.pyplot(fig)

        # Bar plot for Traffic Type vs Revenue
        st.subheader("Traffic Type vs Revenue")
        fig = plt.figure(figsize=(20, 10))
        sns.barplot(data=df.groupby('TrafficType')['Revenue'].value_counts(normalize=True).mul(100).rename('percent').reset_index(),
                    x='TrafficType', y='percent', hue='Revenue', palette='inferno')
        plt.title('Traffic Type Impact on Revenue', fontsize=20)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 4. Summary Stats Tab
    with tabs_bi[3]:
        st.header("Summary Statistics and Pair Plots")

        # Distribution plots
        st.subheader("Distribution Plots")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

        # Histogram for Bounce Rates
        sns.histplot(df['BounceRates'], bins=20, ax=ax1, kde=True)
        ax1.set_title('Bounce Rates', fontsize=15)

        # Histogram for Exit Rates
        sns.histplot(df['ExitRates'], bins=20, ax=ax2, kde=True)
        ax2.set_title('Exit Rates', fontsize=15)

        # Histogram for Page Values
        sns.histplot(df['PageValues'], bins=20, ax=ax3, kde=True)
        ax3.set_title('Page Values', fontsize=15)

        st.pyplot(fig)

        # Count plot for Monthly Revenue vs Visitor Type
        st.subheader("Monthly Revenue vs Visitor Type")
        fig, ax = plt.subplots(figsize=(18, 10))
        orderlist = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        sns.countplot(data=df, x='Month', hue='Revenue', ax=ax, order=orderlist)
        ax.set_title("Monthly Revenue vs Visitor Type", fontsize=20)
        st.pyplot(fig)

        # Pair plot of Bounce Rates and Exit Rates
        st.subheader("Pair Plot of Bounce and Exit Rates")
        pairplot_fig = sns.pairplot(df, x_vars=['BounceRates', 'ExitRates'], y_vars=['BounceRates', 'ExitRates'], hue='Revenue', diag_kind='kde')
        st.pyplot(pairplot_fig.fig)


# 6. Multivariate Analysis Tab
with tabs[2]:
    st.header("Multivariate Analysis")
    st.markdown("### Explore relationships between multiple variables.")

    # Create tabs within the Multivariate Analysis page
    tab1, tab2 = st.tabs(["Boxplots", "Correlation & Pairplot"])

    with tab1:
        st.subheader("Boxplots Analysis")
        st.markdown("#### Month vs Key Metrics with respect to Revenue")

        # Create subplots for boxplots
        fig, axs = plt.subplots(2, 2, figsize=(30, 20))

        # Boxplot for Month vs PageValues
        sns.boxplot(x='Month', y='PageValues', hue='Revenue', data=df, palette='inferno', ax=axs[0, 0])
        axs[0, 0].set_title('Month vs PageValues w.r.t. Revenue', fontsize=20)

        # Boxplot for Month vs ExitRates
        sns.boxplot(x='Month', y='ExitRates', hue='Revenue', data=df, palette='Reds', ax=axs[0, 1])
        axs[0, 1].set_title('Month vs ExitRates w.r.t. Revenue', fontsize=20)

        # Boxplot for Month vs BounceRates
        sns.boxplot(x='Month', y='BounceRates', hue='Revenue', data=df, palette='Oranges', ax=axs[1, 0])
        axs[1, 0].set_title('Month vs BounceRates w.r.t. Revenue', fontsize=20)
        axs[1, 0].legend(loc='upper left', fancybox=True, shadow=True)

        # Boxplot for Visitor Type vs BounceRates
        sns.boxplot(x='VisitorType', y='BounceRates', hue='Revenue', data=df, palette='Purples', ax=axs[1, 1])
        axs[1, 1].set_title('Visitor Type vs BounceRates w.r.t. Revenue', fontsize=20)

        plt.tight_layout()
        st.pyplot(plt.gcf())

        st.markdown("#### Visitor Type & Region vs Key Metrics with respect to Revenue")

        # Create subplots for barplots and boxplots
        fig, axs = plt.subplots(2, 2, figsize=(30, 20))

        # Barplot for Visitor Type vs ExitRates
        sns.barplot(x='VisitorType', y='ExitRates', hue='Revenue', data=df, palette='inferno', ax=axs[0, 0])
        axs[0, 0].set_title('Visitor Type vs ExitRates w.r.t. Revenue', fontsize=20)

        # Boxplot for Visitor Type vs PageValues
        sns.boxplot(x='VisitorType', y='PageValues', hue='Revenue', data=df, palette='Reds', ax=axs[0, 1])
        axs[0, 1].set_title('Visitor Type vs PageValues w.r.t. Revenue', fontsize=20)

        # Boxplot for Region vs PageValues
        sns.boxplot(x='Region', y='PageValues', hue='Revenue', data=df, palette='Oranges', ax=axs[1, 0])
        axs[1, 0].set_title('Region vs PageValues w.r.t. Revenue', fontsize=20)
        axs[1, 0].legend(loc='upper left', fancybox=True, shadow=True)

        # Boxplot for Region vs ExitRates
        sns.boxplot(x='Region', y='ExitRates', hue='Revenue', data=df, palette='Purples', ax=axs[1, 1])
        axs[1, 1].set_title('Region vs ExitRates w.r.t. Revenue', fontsize=20)

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Correlation Heatmap & Pairplot")

        st.markdown("#### Correlation Heatmap")
        # Use only numeric columns for correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        matrix = np.triu(numeric_df.corr())
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(numeric_df.corr(), annot=True, ax=ax, fmt='.1g', vmin=-1, vmax=1, center=0, mask=matrix, cmap='RdBu_r')
        st.pyplot(fig)

        st.markdown("#### Pairplot for Feature Relations")
        # Create pairplot for selected features
        g1 = sns.pairplot(df[['Administrative', 'Informational', 'ProductRelated', 'PageValues', 'Revenue']], hue='Revenue')
        st.pyplot(g1.fig)


# Clustering Analysis Tab
with tabs[3]:
    st.header("Clustering Analysis")
    st.markdown("### Use clustering techniques to find patterns in the data.")

    # Sidebar for Clustering options
    st.sidebar.subheader("Clustering Options")
    max_clusters = st.sidebar.slider("Max Clusters", 2, 15, 10)

    # Perform Elbow Method to determine optimal clusters
    st.subheader("Elbow Method for Optimal Clusters")

    def find_optimal_clusters(data, max_clusters):
        """
        Function to determine the optimal number of clusters using the Elbow Method.

        Parameters:
        - data: The input data for clustering.
        - max_clusters: The maximum number of clusters to test.

        Returns:
        - None
        """
        sse = []
        for k in range(1, max_clusters + 1):  # Test from 1 to max_clusters
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
            sse.append(kmeans.inertia_)  # Sum of squared distances to nearest cluster center
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), sse, 'bx-', c='r')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('SSE')
        plt.title('The Elbow Method showing the Optimal k')
        st.pyplot(plt)
        plt.clf()  # Clear the figure to avoid overlap

    # Selecting relevant columns for clustering
    X = df.iloc[:, [3, 6]].values  # Example columns (adjust as needed)
    find_optimal_clusters(X, max_clusters)

    st.subheader("Clustering Results")

    def draw_optimal_clusters(data, k, max_iter=300, n_init=10, xlabel=None, ylabel=None, title=None):
        """
        Function to draw clusters after applying KMeans clustering.

        Parameters:
        - data: The input data for clustering.
        - k: The number of clusters to form.
        - max_iter: Maximum number of iterations for the KMeans algorithm.
        - n_init: Number of time the KMeans algorithm will be run with different centroid seeds.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        - title: Title of the plot.

        Returns:
        - None
        """
        kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init, random_state=0)
        kmeans.fit(data)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='*', s=200, edgecolor='k')
        plt.title(title, fontsize=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(*scatter.legend_elements(), title="Clusters")
        st.pyplot(plt)
        plt.clf()  # Clear the figure to avoid overlap

    # Example clustering with 2 clusters
    draw_optimal_clusters(X, 2, xlabel="Informational Duration", ylabel="Bounce Rates", title="Informational Duration vs Bounce Rates")


# Caching the data loading functions
@st.cache_data
def load_comparison_data():
    """
    Load the model comparison data from CSV.

    Returns:
    - pd.DataFrame: DataFrame containing model comparison metrics.
    """
    return pd.read_csv('reports/comparison/model_comparison.csv')

@st.cache_resource
def load_comparison_plot():
    """
    Load the path to the ROC comparison plot image.

    Returns:
    - str: File path to the ROC comparison plot image.
    """
    return 'reports/comparison/roc_comparison.png'

@st.cache_resource
def load_model_report_image(model_name):
    """
    Load the path to the confusion matrix image for a specific model.

    Parameters:
    - model_name (str): The name of the model.

    Returns:
    - str: File path to the model's confusion matrix image.
    """
    return f"reports/model_reports/{model_name}_confusion_matrix.png"


# Load comparison data and plot
comparison_df = load_comparison_data()
comparison_plot_path = load_comparison_plot()

with tabs[4]:
    st.subheader("Modeling Dashboard")

    # Sidebar for Model Selection
    model_names = comparison_df['model_name'].unique()
    selected_model = st.selectbox("Select a model to view its report:", model_names)

    # Display model report
    st.header(f"Report for {selected_model}")

    # Load and display confusion matrix image for the selected model
    model_report_path = load_model_report_image(selected_model)
    st.image(model_report_path, caption=f"{selected_model} Confusion Matrix")

    # Extract and display performance metrics for the selected model
    model_report = comparison_df[comparison_df['model_name'] == selected_model].iloc[0]
    st.text(f"Accuracy: {model_report['accuracy']:.2f}")
    st.text(f"F1 Score: {model_report['f1']:.2f}")
    st.text(f"Precision: {model_report['precision']:.2f}")
    st.text(f"Recall: {model_report['recall']:.2f}")

    # Comparison Section
    st.header("Model Comparison")

    # Display ROC Curve Comparison Plot
    st.image(comparison_plot_path, caption="ROC Curve Comparison")

    # Display Comparison Table
    st.subheader("Comparison Table")
    st.dataframe(comparison_df)

# Footer for additional context or branding
st.markdown("""
<style>
    .reportview-container .main footer {visibility: hidden;}
    .reportview-container .main::after {
        content: 'Analysis by [Your Name]';
        display: block;
        text-align: center;
        color: grey;
        font-size: 14px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)
