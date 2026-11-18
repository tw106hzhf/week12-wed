import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import Counter
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Cashing of model, done once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="CLOY Seri Comment Explorer", layout="wide")
st.title("CLOY 'Seri' Comments — NLP & Clustering Explorer")

# ---------------------------------------------------------
# Sidebar: Data Loading & Preprocessing
# ---------------------------------------------------------
st.sidebar.header("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload seri_comments.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("No CSV uploaded. Using sample comments from your thread.")
    df = pd.DataFrame({
        "timestamp": [1583764623, 1583885982],
        "submission title": [
            "ok, let’s get this started! What are your thoughts about the show?",
            "ok, let’s get this started! What are your thoughts about the show?",
        ],
        "comment_text": [
            (
                "i'm so emotionally drained and exhausted from being in awe of"
                " this show even though i finished watching 2 weeks ago. my"
                " brain doesn't work cuz i'm thinking about CLOY and my favorite"
                " part of it other than the couple is that the women have"
                " really 3D roles... like Seri can cry nonstop but you still"
                " know she a boss."
            ),
            (
                "I love this show. I stumbled across it on Netflix, gave it a"
                " try, and fell in love. I finished it this morning and"
                " already miss it. I love Yoon Se-Ri and Jeong Hyeok and nearly"
                " all of the other characters. I was so happy with the ending."
                " I’ve added two more kdramas to my Netflix list."
            ),
        ],
    })

if "comment_text" not in df.columns:
    st.error("The uploaded CSV must contain a `comment_text` column.")
    st.stop()

df = df.dropna(subset=["comment_text"]).reset_index(drop=True)
st.sidebar.metric("Loaded Documents (Comments)", len(df))


# Tokenization helper
def tokenize(text, remove_stopwords=False):
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", str(text).lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return tokens


# ---------------------------------------------------------
# Helper: c-TF-IDF Calculation
# ---------------------------------------------------------
def compute_ctfidf(docs, labels, top_n=10, stop_words=None):
    """Computes Class-based TF-IDF (c-TF-IDF) for each cluster."""
    df_temp = pd.DataFrame({"doc": docs, "label": labels})
    # Merge all documents in each cluster into a single meta-document
    docs_per_class = df_temp.groupby("label", as_index=False).agg(
        {"doc": " ".join}
    )

    count_vec = CountVectorizer(stop_words=stop_words)
    t_f = count_vec.fit_transform(docs_per_class["doc"]).toarray()
    words = count_vec.get_feature_names_out()

    # c-TF-IDF formula: tf_{x,c} * log(1 + A / f_x)
    w = t_f.sum(axis=1, keepdims=True)
    tf_norm = np.divide(t_f, w, out=np.zeros_like(t_f, dtype=float), where=w != 0)
    avg_nr_words_per_class = w.mean()
    f_x = t_f.sum(axis=0)
    idf = np.log(1 + (avg_nr_words_per_class / np.maximum(f_x, 1)))

    ctfidf = tf_norm * idf

    top_words_per_cluster = {}
    for idx, label in enumerate(docs_per_class["label"]):
        top_indices = np.argsort(ctfidf[idx])[::-1][:top_n]
        top_words_per_cluster[label] = [
            (words[i], round(ctfidf[idx][i], 4)) for i in top_indices
        ]
    return top_words_per_cluster


# ---------------------------------------------------------
# Tabs Setup
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
       "1. Zipf's Law & Vocabulary",
       "2. Word Length Distribution",
       "3. K-Means Comparison",
       "4. Cluster Explorer (c-TFIDF)",
       "5. BERTopic Modeling",
   ])

# =========================================================
# TAB 1: Zipf's Law & Vocabulary Size
# =========================================================
with tab1:
    st.header("Zipf's Law & Vocabulary Size")
    remove_sw_zipf = st.checkbox(
        "Remove English Stopwords", value=False, key="zipf_sw"
    )

    all_tokens = []
    for text in df["comment_text"]:
        all_tokens.extend(tokenize(text, remove_stopwords=remove_sw_zipf))

    word_counts = Counter(all_tokens)
    vocab_size = len(word_counts)

    col1, col2 = st.columns(2)
    col1.metric("Total Vocabulary Size (Unique Words)", f"{vocab_size:,}")
    col2.metric("Total Tokens", f"{len(all_tokens):,}")

    if vocab_size > 0:
        most_common = word_counts.most_common()
        zipf_df = pd.DataFrame(most_common, columns=["Word", "Frequency"])
        zipf_df["Rank"] = np.arange(1, len(zipf_df) + 1)
        # Theoretical Zipf curve: f(r) = f(1) / r
        zipf_df["Theoretical Zipf"] = zipf_df["Frequency"].iloc[0] / zipf_df["Rank"]

        fig_zipf = go.Figure()
        fig_zipf.add_trace(
            go.Scatter(
                x=zipf_df["Rank"],
                y=zipf_df["Frequency"],
                mode="lines+markers",
                name="Observed Frequency",
                text=zipf_df["Word"],
                hovertemplate=(
                    "Rank: %{x}<br>Word: %{text}<br>Count: %{y}<extra></extra>"
                ),
            )
        )
        fig_zipf.add_trace(
            go.Scatter(
                x=zipf_df["Rank"],
                y=zipf_df["Theoretical Zipf"],
                mode="lines",
                name="Ideal Zipf (1/rank)",
                line=dict(dash="dash", color="red"),
            )
        )
        fig_zipf.update_layout(
            title="Zipf's Law Distribution (Log-Log Scale)",
            xaxis_title="Word Rank (log)",
            yaxis_title="Frequency (log)",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
        )
        st.plotly_chart(fig_zipf, use_container_width=True)

        with st.expander("View Top 25 Most Frequent Words"):
            st.dataframe(zipf_df.head(25), use_container_width=True)

# =========================================================
# TAB 2: Word Length Distribution per Document
# =========================================================
with tab2:
    st.header("Document Word Count Distribution")
    remove_sw_dist = st.toggle(
        "Remove stopwords", value=False, key="dist_sw_toggle"
    )

    doc_word_counts = [
        len(tokenize(text, remove_stopwords=remove_sw_dist))
        for text in df["comment_text"]
    ]
    total_words_all_docs = sum(doc_word_counts)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Words Across All Documents", f"{total_words_all_docs:,}")
    c2.metric("Average Words per Document", f"{np.mean(doc_word_counts):.1f}")
    c3.metric("Median Words per Document", f"{np.median(doc_word_counts):.1f}")

    dist_df = pd.DataFrame({"Word Count": doc_word_counts})
    fig_dist = px.histogram(
        dist_df,
        x="Word Count",
        nbins=30,
        title=(
            "Distribution of Word Count per Document"
            f" ({'Stopwords Removed' if remove_sw_dist else 'With Stopwords'})"
        ),
        labels={"Word Count": "Number of Words in Comment"},
        color_discrete_sequence=["#2b5c8f"],
    )
    fig_dist.update_layout(yaxis_title="Number of Documents", bargap=0.05)
    st.plotly_chart(fig_dist, use_container_width=True)

# =========================================================
# TAB 3: K-Means Comparison (Count vs TF-IDF, Stopwords, min_df/max_df)
# =========================================================
with tab3:
    st.header("K-Means Clustering & Vectorizer Comparison")

    if len(df) < 10:
        st.warning(
            "You currently have fewer than 10 documents loaded. Upload your"
            " full CSV to test multiple K values."
        )
    else:
        with st.form("clustering_params"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                k_min, k_max = st.slider(
                    "Range of K values to evaluate",
                    min_value=2,
                    max_value=min(15, len(df) - 1),
                    value=(2, min(6, len(df) - 1)),
                )
            with col_b:
                min_df = st.slider(
                    "min_df (Ignore terms appearing in < X% of docs)",
                    min_value=0.0,
                    max_value=0.20,
                    value=0.01,
                    step=0.01,
                )
            with col_c:
                max_df = st.slider(
                    "max_df (Ignore terms appearing in > X% of docs)",
                    min_value=0.50,
                    max_value=1.00,
                    value=0.90,
                    step=0.05,
                )

            run_eval = st.form_submit_button("Run Clustering Comparison")

        if run_eval:
            k_values = list(range(k_min, k_max + 1))
            configs = [
                ("CountVectorizer (With Stopwords)", CountVectorizer, None),
                ("CountVectorizer (No Stopwords)", CountVectorizer, "english"),
                ("TF-IDF (With Stopwords)", TfidfVectorizer, None),
                ("TF-IDF (No Stopwords)", TfidfVectorizer, "english"),
            ]

            eval_records = []
            total_steps = len(configs) * len(k_values)
            current_step = 0

            progress_bar = st.progress(0, text="Initializing vectorizers...")
            with st.spinner("Running K-Means across configurations..."):
                for label, VecClass, sw in configs:
                    try:
                        vec = VecClass(
                            stop_words=sw, min_df=min_df, max_df=max_df
                        )
                        X = vec.fit_transform(df["comment_text"])

                        if X.shape[1] < 2:
                            st.warning(
                                f"Skipped {label}: too few features left with"
                                " current min_df/max_df."
                            )
                            current_step += len(k_values)
                            continue

                        for k in k_values:
                            km = KMeans(n_clusters=k, random_state=42, n_init=10)
                            preds = km.fit_predict(X)

                            # Goodness metrics (Higher is better for both)
                            sil = silhouette_score(X, preds)
                            ch = calinski_harabasz_score(X.toarray(), preds)

                            eval_records.append({
                                "Configuration": label,
                                "K": k,
                                "Silhouette Score": sil,
                                "Calinski-Harabasz Index": ch,
                                "Inertia": km.inertia_,
                            })
                            current_step += 1
                            progress_bar.progress(
                                current_step / total_steps,
                                text=f"Evaluating {label} at K={k}...",
                            )
                    except ValueError as e:
                        st.warning(f"Configuration '{label}' failed: {e}")

            progress_bar.empty()

            if eval_records:
                res_df = pd.DataFrame(eval_records)

                st.subheader("Clustering Goodness Criteria")
                st.markdown("""
                * **Silhouette Score (-1 to 1):** Measures how similar a comment is to its own cluster compared to other clusters. **Higher is better.**
                * **Calinski-Harabasz Index:** Ratio of between-cluster dispersion to within-cluster dispersion. **Higher indicates denser, better-separated clusters.**
                """)

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_sil = px.line(
                        res_df,
                        x="K",
                        y="Silhouette Score",
                        color="Configuration",
                        markers=True,
                        title="Silhouette Score by K & Vectorizer",
                    )
                    st.plotly_chart(fig_sil, use_container_width=True)

                with col_chart2:
                    fig_ch = px.line(
                        res_df,
                        x="K",
                        y="Calinski-Harabasz Index",
                        color="Configuration",
                        markers=True,
                        title="Calinski-Harabasz Index by K & Vectorizer",
                    )
                    st.plotly_chart(fig_ch, use_container_width=True)

                with st.expander("View Raw Evaluation Metrics Table"):
                    st.dataframe(res_df, use_container_width=True)

# =========================================================
# TAB 4: Cluster Content Explorer (c-TF-IDF & Centroid Distance)
# =========================================================
with tab4:
    st.header("Explore Cluster Contents")

    if len(df) < 3:
        st.warning("Upload your full dataset to explore clusters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            exp_k = st.number_input(
                "Number of Clusters (K)",
                min_value=2,
                max_value=min(20, len(df) - 1),
                value=min(5, len(df) - 1),
            )
        with c2:
            exp_vec_type = st.selectbox(
                "Vectorizer", ["TF-IDF", "CountVectorizer"]
            )
        with c3:
            exp_sw = st.checkbox("Remove Stopwords", value=True, key="exp_sw")
        with c4:
            exp_min_df = st.number_input(
                "min_df", min_value=0.0, max_value=0.2, value=0.01, step=0.01
            )

        if st.button("Cluster & Inspect"):
            with st.spinner(
                "Clustering documents and computing c-TF-IDF scores..."
            ):
                sw_param = "english" if exp_sw else None
                VecClass = (
                    TfidfVectorizer
                    if exp_vec_type == "TF-IDF"
                    else CountVectorizer
                )

                vec = VecClass(
                    stop_words=sw_param, min_df=exp_min_df, max_df=0.95
                )
                X = vec.fit_transform(df["comment_text"])

                km = KMeans(n_clusters=exp_k, random_state=42, n_init=10)
                labels = km.fit_predict(X)

                # Calculate Euclidean distance from each document to all cluster centroids
                distances = km.transform(X)
                dist_to_assigned = [
                    distances[i, labels[i]] for i in range(len(df))
                ]

                # Compute c-TF-IDF keywords
                ctfidf_terms = compute_ctfidf(
                    df["comment_text"], labels, top_n=10, stop_words=sw_param
                )

                # Store in session state so changing the cluster dropdown doesn't re-run KMeans
                explored_df = df.copy()
                explored_df["Cluster"] = labels
                explored_df["Distance_to_Centroid"] = dist_to_assigned

                st.session_state["explored_df"] = explored_df
                st.session_state["ctfidf_terms"] = ctfidf_terms

                # 2D PCA Projection for visual overview
                if X.shape[1] >= 2:
                    pca = PCA(n_components=2, random_state=42)
                    coords = pca.fit_transform(X.toarray())
                    explored_df["PCA_1"] = coords[:, 0]
                    explored_df["PCA_2"] = coords[:, 1]
                    st.session_state["explored_df"] = explored_df

        if "explored_df" in st.session_state:
            explored_df = st.session_state["explored_df"]
            ctfidf_terms = st.session_state["ctfidf_terms"]

            if "PCA_1" in explored_df.columns:
                fig_pca = px.scatter(
                    explored_df,
                    x="PCA_1",
                    y="PCA_2",
                    color=explored_df["Cluster"].astype(str),
                    hover_data=["submission title"],
                    title="2D PCA Projection of Clusters",
                    labels={"color": "Cluster"},
                )
                st.plotly_chart(fig_pca, use_container_width=True)

            st.divider()

            selected_cluster = st.selectbox(
                "Select a Cluster to Inspect",
                options=sorted(explored_df["Cluster"].unique()),
                format_func=lambda x: (
                    f"Cluster {x} ({sum(explored_df['Cluster'] == x)} comments)"
                ),
            )

            # Display c-TF-IDF Top Words
            st.subheader(
                f"Top c-TF-IDF Keywords for Cluster {selected_cluster}"
            )
            terms = ctfidf_terms.get(selected_cluster, [])
            if terms:
                kw_df = pd.DataFrame(terms, columns=["Term", "c-TF-IDF Score"])
                col_kw1, col_kw2 = st.columns([2, 1])
                with col_kw1:
                    fig_kw = px.bar(
                        kw_df[::-1],
                        x="c-TF-IDF Score",
                        y="Term",
                        orientation="h",
                        title="Most Representative Terms (c-TF-IDF)",
                    )
                    st.plotly_chart(fig_kw, use_container_width=True)
                with col_kw2:
                    st.dataframe(kw_df, use_container_width=True, height=350)

            # Display Comments Sorted by Distance to Centroid
            st.subheader(
                f"Comments in Cluster {selected_cluster} (Sorted Closest to"
                " Centroid First)"
            )
            cluster_docs = explored_df[
                explored_df["Cluster"] == selected_cluster
            ].sort_values(by="Distance_to_Centroid", ascending=True)

            display_cols = ["Distance_to_Centroid", "comment_text"]
            if "submission title" in cluster_docs.columns:
                display_cols.insert(1, "submission title")
            if "timestamp" in cluster_docs.columns:
                display_cols.insert(1, "timestamp")

            st.dataframe(
                cluster_docs[display_cols],
                use_container_width=True,
                column_config={
                    "Distance_to_Centroid": st.column_config.NumberColumn(
                        "Distance to Centroid", format="%.4f"
                    ),
                    "comment_text": st.column_config.TextColumn(
                        "Comment Text", width="large"
                    ),
                },
            )

# =========================================================
# TAB 5: BERTopic Modeling (On-the-Fly Semantic Topics)
# =========================================================
with tab5:
    st.header("BERTopic Semantic Topic Modeling")
    st.markdown(
        "Uses pretrained transformer embeddings (`all-MiniLM-L6-v2`) + UMAP"
        " + HDBSCAN + c-TF-IDF directly on your comments—no pre-training"
        " required."
    )

    if len(df) < 15:
        st.warning(
            "BERTopic needs at least ~15–20 comments to form density-based"
            " clusters. Please upload your full CSV."
        )
    else:
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            min_topic_size = st.number_input(
                "Minimum Topic Size",
                min_value=5,
                max_value=100,
                value=15,
                help="Minimum number of comments required to form a topic.",
            )
        with col_b2:
            nr_topics_option = st.selectbox(
                "Target Number of Topics",
                options=["Auto (HDBSCAN)", 5, 8, 10, 12, 15],
                help=(
                    "Auto lets HDBSCAN decide, or pick a number to merge"
                    " similar topics down."
                ),
            )
        with col_b3:
            bt_remove_sw = st.checkbox(
                "Remove Stopwords in Topic Labels", value=True
            )

        if st.button("Run BERTopic"):
            with st.spinner(
                "Generating embeddings and discovering semantic topics..."
                " (First run downloads ~80MB model)"
            ):
                docs = df["comment_text"].tolist()
                embedding_model = load_embedding_model()

                # Custom CountVectorizer ensures clean c-TF-IDF labels without filler words
                vec_model = CountVectorizer(
                    stop_words="english" if bt_remove_sw else None,
                    ngram_range=(1, 2),
                    min_df=2,
                )

                nr_topics = (
                    None
                    if nr_topics_option == "Auto (HDBSCAN)"
                    else int(nr_topics_option)
                )

                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    vectorizer_model=vec_model,
                    min_topic_size=min_topic_size,
                    nr_topics=nr_topics,
                    calculate_probabilities=True,
                    verbose=False,
                )

                topics, probs = topic_model.fit_transform(docs)

                bt_df = df.copy()
                bt_df["Topic"] = topics
                # Confidence score (probability of belonging to the assigned topic)
                bt_df["Topic_Confidence"] = [
                    round(float(np.max(p)), 4) if p is not None else 0.0
                    for p in probs
                ]

                st.session_state["bt_model"] = topic_model
                st.session_state["bt_df"] = bt_df

        if "bt_model" in st.session_state and "bt_df" in st.session_state:
            topic_model = st.session_state["bt_model"]
            bt_df = st.session_state["bt_df"]

            topic_info = topic_model.get_topic_info()
            num_real_topics = len(topic_info[topic_info["Topic"] != -1])
            outlier_count = int((bt_df["Topic"] == -1).sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Topics Discovered", num_real_topics)
            m2.metric("Assigned Comments", len(bt_df) - outlier_count)
            m3.metric("Outlier Comments (Topic -1)", outlier_count)

            # Built-in Plotly Bar Chart of Top c-TF-IDF Words per Topic
            st.subheader("Topic Keywords Overview (c-TF-IDF)")
            if num_real_topics > 0:
                fig_bt_bars = topic_model.visualize_barchart(
                    top_n_topics=min(8, num_real_topics), n_words=8
                )
                st.plotly_chart(fig_bt_bars, use_container_width=True)

            with st.expander("View All Topics Summary Table"):
                st.dataframe(topic_info, use_container_width=True)

            st.divider()

            # Inspect individual topic comments
            selected_topic = st.selectbox(
                "Select a Topic to Inspect Comments",
                options=topic_info["Topic"].tolist(),
                format_func=lambda t: (
                    f"Topic {t}:"
                    f" {topic_info.loc[topic_info['Topic'] == t, 'Name'].values[0]}"
                    f" ({topic_info.loc[topic_info['Topic'] == t, 'Count'].values[0]} comments)"
                ),
            )

            topic_docs = bt_df[bt_df["Topic"] == selected_topic].sort_values(
                by="Topic_Confidence", ascending=False
            )

            st.subheader(
                f"Comments in Topic {selected_topic} (Sorted by Topic"
                " Confidence)"
            )
            cols_to_show = ["Topic_Confidence", "comment_text"]
            if "submission title" in topic_docs.columns:
                cols_to_show.insert(1, "submission title")

            st.dataframe(
                topic_docs[cols_to_show],
                use_container_width=True,
                column_config={
                    "Topic_Confidence": st.column_config.NumberColumn(
                        "Confidence", format="%.4f"
                    ),
                    "comment_text": st.column_config.TextColumn(
                        "Comment Text", width="large"
                    ),
                },
            )