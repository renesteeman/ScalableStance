from bertopic import BERTopic
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import pandas as pd

def extract_topics(data):
    sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    embeddings = sentence_model.encode(data['title_topic'])

    cluster_model = AgglomerativeClustering(linkage='ward', distance_threshold=1.5, n_clusters=None)
    topic_model = BERTopic(hdbscan_model=cluster_model).fit(data['title_topic'], embeddings)
    topics, probs = topic_model.fit_transform(data['title_topic'])

    topic_labels = topic_model.generate_topic_labels(nr_words=3,
                                                     topic_prefix=False,
                                                     word_length=15,
                                                     separator=", ")
    topic_model.set_topic_labels(topic_labels)

    topic_labels_series = pd.Series(topic_labels)
    docs_topic = topic_labels_series[topics].tolist()
    data['predicted_topic'] = docs_topic

    return data, topic_model
