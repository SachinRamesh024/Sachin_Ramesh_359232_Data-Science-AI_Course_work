# ==============================
# Spotify ML Explorer - app.py
# ==============================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="🎧 Spotify ML Explorer",
    page_icon="🎵",
    layout="wide"
)

# ==============================
# Custom Spotify Theme CSS
# ==============================
st.markdown(
    """
    <style>
    /* Background & text */
    body {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Helvetica', sans-serif;
    }
    /* Sidebar */
    .css-1d391kg .css-1d391kg {
        background-color: #1A1A1A;
    }
    .css-1v3fvcr {
        background-color: #1DB954 !important;
        color: #FFFFFF !important;
    }
    /* Buttons */
    .stButton>button {
        background-color: #1DB954;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    /* Metrics box */
    .stMetric {
        background-color: #1DB954;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    /* Dataframe style */
    .stDataFrame table {
        background-color: #1A1A1A;
        color: #FFFFFF;
    }
    /* Headings */
    .css-10trblm h1, .css-10trblm h2, .css-10trblm h3 {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎧 Spotify ML Explorer Dashboard")
st.markdown("Interactive dashboard to explore Spotify tracks dataset and experiment with ML models")

# ==============================
# Sidebar - Controls
# ==============================
st.sidebar.header("🔧 Controls")

# Learning type
learning_type = st.sidebar.selectbox("Select Learning Type", ["Supervised", "Unsupervised"])

# Dataset upload
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
target_col = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")

    if learning_type == "Supervised":
        target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    # Algorithm selection and hyperparameters
    if learning_type == "Supervised":
        model_choice = st.sidebar.selectbox("Select Supervised Model", ["Decision Tree", "Random Forest", "SVM"])
        max_depth = st.sidebar.slider("Max Depth (Decision Tree / Random Forest)", 1, 20, 5)
        n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", 10, 200, 50)
    else:
        model_choice = st.sidebar.selectbox("Select Unsupervised Model", ["KMeans", "Agglomerative", "DBSCAN"])
        n_clusters = st.sidebar.slider("Number of Clusters (KMeans / Agglomerative)", 2, 10, 4)
        eps = st.sidebar.slider("Epsilon (DBSCAN)", 0.1, 5.0, 0.5)

# ==============================
# Dataset Preview
# ==============================
if uploaded_file:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))
    st.write("Dataset shape:", df.shape)

    # ==============================
    # Preprocessing
    # ==============================
    st.subheader("⚙️ Preprocessing")
    df_processed = df.copy()

    # Fill numeric missing values
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

    # Encode categorical columns
    cat_cols = df_processed.select_dtypes(include='object').columns.tolist()
    if learning_type == "Supervised" and target_col in cat_cols:
        cat_cols.remove(target_col)
    for col in cat_cols:
        df_processed[col] = LabelEncoder().fit_transform(df_processed[col].astype(str))

    # Feature scaling
    X = df_processed.drop(columns=[target_col]) if learning_type == "Supervised" else df_processed.copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if learning_type == "Supervised":
        y = df_processed[target_col]

    st.success("✅ Preprocessing Completed")

    # ==============================
    # Model Training & Evaluation
    # ==============================
    if st.sidebar.button("Train / Run Model"):
        st.subheader("🤖 Model Results")

        if learning_type == "Supervised":
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if model_choice == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            else:
                model = SVC()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc*100:.2f}%")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        else:
            # Use subset for large datasets
            if X_scaled.shape[0] > 5000:
                X_sample = X_scaled.sample(5000, random_state=42)
                st.info("⚡ Dataset too large, using a 5000-row sample for clustering")
            else:
                X_sample = X_scaled.copy()

            if model_choice == "KMeans":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(X_sample)
            elif model_choice == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X_sample)
            else:
                model = DBSCAN(eps=eps)
                labels = model.fit_predict(X_sample)

            X_sample['Cluster'] = labels

            if len(set(labels)) > 1 and -1 not in set(labels):
                score = silhouette_score(X_sample.drop(columns=['Cluster']), labels)
                st.metric("Silhouette Score", f"{score:.3f}")

            # PCA 2D visualization
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X_sample.drop(columns=['Cluster']))
            df_vis = pd.DataFrame(X_vis, columns=['PCA1', 'PCA2'])
            df_vis['Cluster'] = labels.astype(str)
            fig = px.scatter(df_vis, x='PCA1', y='PCA2', color='Cluster',
                             color_discrete_sequence=px.colors.qualitative.Vivid,
                             title="Cluster Visualization (2D PCA)")
            fig.update_layout(plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # Additional Insights
    # ==============================
    st.subheader("📊 Additional Spotify Insights")

    if 'popularity' in df.columns:
        fig = px.histogram(df, x='popularity', nbins=30, title="Popularity Distribution",
                           color_discrete_sequence=['#1DB954'])
        fig.update_layout(plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    if 'artists' in df.columns:
        top_artists = df['artists'].value_counts().head(10)
        fig = px.bar(x=top_artists.values, y=top_artists.index, orientation='h',
                     title="Top 10 Artists", color_discrete_sequence=['#1DB954'])
        fig.update_layout(plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    if 'genres' in df.columns:
        top_genres = df['genres'].value_counts().head(10)
        fig = px.bar(x=top_genres.values, y=top_genres.index, orientation='h',
                     title="Top 10 Genres", color_discrete_sequence=['#1DB954'])
        fig.update_layout(plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data Source: Kaggle Spotify Dataset")
