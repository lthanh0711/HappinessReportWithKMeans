import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import cluster
import streamlit as st
from PIL import Image
import time
from os import path

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans, k_means
import plotly.express as px
import plotly.graph_objects as go
from chart_studio.plotly import plot, iplot
from plotly.offline import iplot

url = "https://www.kaggle.com/datasets/unsdsn/world-happiness"
columns = ['Score','GDP per capita', 'Social support', 'Healthy life expectancy', 
    'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']
train_columns = [item for item in columns if item not in ["Overall rank", "Country or region", "Score"]]

@st.cache(persist=True)
def loadData():
    return pd.read_csv("./happiness_report.csv")

# Interactive plot
def interactivePlot(happy_df, metric):
    fig = px.scatter(happy_df, x=metric, y="Score", size="Overall rank", color="Country or region")
    st.plotly_chart(fig)
    
# DATA VISUALIZATION FUNCTION
def pairPlot(happy_df):
    fig1 = sns.pairplot(happy_df[columns])
    fig1.savefig("./Save Figure/pairPlot.png")
    return fig1

def disPlot(happy_df):
    fig2 = plt.figure(figsize = (20, 50))
    for i in range(len(columns)):
        plt.subplot(8, 2, i+1)
        sns.distplot(happy_df[columns[i]], color = 'r');
        plt.title(columns[i])
    plt.tight_layout()
    fig2.savefig("./Save Figure/disPlot.png", bbox_inches="tight")
    return fig2

def corrMatrix(happy_df):
    cm = happy_df.corr()
    fig3, ax = plt.subplots()
    sns.heatmap(cm, ax=ax, annot=True)
    plt.tight_layout()
    fig3.savefig("./Save Figure/corrMatrix.png")
    return fig3

# PREPARE THE DATA
@st.cache(persist=True)
def prep(happy_df):
    df_seg = happy_df.drop(columns=["Overall rank", "Country or region", "Score"])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_seg)
    return scaler, scaled_data
 
# FIND OPTIMAL CLUSTERS 
@st.cache(persist=True)    
def optimalClusters(data):
    scores = []
    range_values = range(1, 20)
    
    for i in range_values:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        scores.append(kmeans.inertia_)
        
    return scores

def plotCluster(scores):
    fig = plt.figure()
    plt.plot(scores, "bx-")
    plt.title("Finding optimal number of clusters")
    plt.xlabel("Clusters")
    plt.ylabel("Score")
    fig.savefig("./Save Figure/clusterPlot.png")
    return fig

# TRAIN THE MODEL
#@st.cache(persist=True)
def train(scaled_data, n_k, scaler):
    kmeans = KMeans(n_k)
    kmeans.fit(scaled_data)
    
    cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=train_columns)
    cluster_centers = scaler.inverse_transform(cluster_centers)
    
    y_kmeans = kmeans.fit_transform(scaled_data)
    labels = kmeans.labels_
    
    return y_kmeans, labels

# VISUALIZE THE CLUSTERS
def visualCluster(happy_df_cluster):
    data = dict(type = 'choropleth', 
           locations = happy_df_cluster["Country or region"],
           locationmode = 'country names',
           colorscale='RdYlGn',
           z = happy_df_cluster['Cluster'], 
           text = happy_df_cluster["Country or region"],
           colorbar = {'title':'Clusters'})

    layout = dict(title = 'Geographical Visualization of Clusters', 
              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))

    fig = go.Figure(data = [data], layout=layout)
    return fig

def main():
    # OVERVIEW
    # Load data once
    happy_df = loadData()
    st.title("Analyzing World Happiness Report with K-Means Clustering")
    st.write("The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. This question, known as the Cantril ladder, asks respondents to think of a ladder with the best possible life for them being a 10 and the worst possible life being a 0 and to rate their own current lives on that scale.")
    # Center image
    background = Image.open("./earth.jpg")
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    col2.image(background, use_column_width=True)
    if (st.checkbox("Show raw data", False)):
        st.markdown("[Link to the dataset](%s)" % url)
        st.write(happy_df)
    
    # FIND OWN COUNTRY
    st.subheader("Find your country or region ranking")
    select_place = st.selectbox("Select below", happy_df["Country or region"].sort_values())
    st.write(happy_df[happy_df["Country or region"] == select_place])

    # DESCRIBE THE DATA
    st.subheader("Overview of the dataset")
    st.write(happy_df.describe())
    
    # Interactive plot
    feature_columns = [item for item in columns if item != "Score"]
    metric = st.selectbox("Select a metric", feature_columns)
    interactivePlot(happy_df, metric)
    
    
    # DATA VISUALIZATION
    st.subheader("Data Visualization")
    
    st.markdown("#### Pair plot")
    # If plot already created, just show it
    if path.exists("./Save Figure/pairPlot.png"):
        st.image("./Save Figure/pairPlot.png")
    # else create the plot
    else:
        # Loading screen
        _left, mid, _right = st.columns(3)
        with mid:
            gif_runner = st.image("./processing.gif")
            text = st.text("Loading data to display ...")
        st.pyplot(pairPlot(happy_df))
        gif_runner.empty()
        text.empty()
    
    st.markdown("#### Distribution plot")
    # If plot already created, just show it
    if path.exists("./Save Figure/disPlot.png"):
        st.image("./Save Figure/disPlot.png")
    # else create the plot
    else:
        # Loading screen
        _left, mid, _right = st.columns(3)
        with mid:
            gif_runner = st.image("./processing.gif")
            text = st.text("Loading data to display ...")
        st.pyplot(disPlot(happy_df))
        gif_runner.empty()
        text.empty()
    
    st.markdown("#### Correlation Matrix")
    # If plot already created, just show it
    if path.exists("./Save Figure/corrMatrix.png"):
        st.image("./Save Figure/corrMatrix.png")
    # else create the plot
    else:
        # Loading screen
        _left, mid, _right = st.columns(3)
        with mid:
            gif_runner = st.image("./processing.gif")
            text = st.text("Loading data to display ...")
        st.pyplot(corrMatrix(happy_df))
        gif_runner.empty()
        text.empty()
    
    # GET THE PREPARED DATA
    scaler, scaled_data = prep(happy_df)
    
    # TRAIN THE MODEL
    st.subheader("Train the model")
    
    # Find number of clusters
    st.markdown("#### Find the number of clusters")
    scores = optimalClusters(scaled_data)
    # If plot already created, just show it
    if path.exists("./Save Figure/clusterPlot.png"):
        st.image("./Save Figure/clusterPlot.png")
    else:
        st.pyplot(plotCluster(scores))
    
    # Apply K-Means
    st.markdown("#### Apply K-Means Clustering")
    n_k = st.selectbox("Select number of clusters", [x for x in range(1, 20)])
    y_kmeans, labels = train(scaled_data, n_k, scaler)
    
    # Data after cluster
    happy_df_cluster = pd.concat([happy_df, pd.DataFrame({"Cluster":labels})], axis=1)
    if (st.checkbox("Show data in cluster", False)):
        st.write(happy_df_cluster)
    
    # VISUALIZING THE CLUSTERS
    st.subheader("Visualize the clusters")
    st.plotly_chart(visualCluster(happy_df_cluster))
        
    
    
    
if __name__ == '__main__':
    main()