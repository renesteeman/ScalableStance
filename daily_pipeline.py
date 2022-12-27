import os
import modal

stub = modal.Stub()
getdailyarticles_image = modal.Image.debian_slim().pip_install(["hopsworks", "newsapi-python"])
preprocessing_image = modal.Image.debian_slim().pip_install(["hopsworks", "nltk"])
topicextraction_image = modal.Image.debian_slim().pip_install(["hopsworks", "bertopic", "sentence-transformers"])
stance_image = modal.Image.debian_slim().pip_install(["hopsworks", "keras"])


@stub.function(image=getdailyarticles_image, secrets=[
    modal.Secret.from_name("news-api-key"),
    modal.Secret.from_name("hopswork-api-key"),
])
def get_daily_articles_and_store():
    """Fetch top English headlines using newsapi and store in hopsworks."""
    import pandas as pd

    import hopsworks
    from newsapi import NewsApiClient

    # newsapi init
    newsapi = NewsApiClient(api_key=os.environ["NEWS-API-KEY"])

    # /v2/top-headlines
    top_headlines = newsapi.get_top_headlines(language='en', country='us')

    headlines = pd.DataFrame(top_headlines['articles'])
    headlines = headlines[['title', 'url', 'publishedAt']]

    project = hopsworks.login()
    feature_store = project.get_feature_store()
    article_feature_store = feature_store.get_or_create_feature_group(
        name="articles_daily",
        version=1,
        primary_key=["url"],
        description="Articles loaded in daily"
    )
    article_feature_store.insert(headlines, write_options={"wait_for_job" : False})

@stub.function(image=preprocessing_image, secret=modal.Secret.from_name("hopswork-api-key"))
def clean_and_store_daily_articles():
    """Fetch articles from Hopsworks, preprocess, and store in new cleaned articles feature group."""
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    import numpy as np
    import re
    import hopsworks

    custom_words_to_filter = []

    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    stop_words = nltk.corpus.stopwords.words('english')

    def clean_text(text, filter_stop_words=True):
        if filter_stop_words:
            words_to_filter = np.concatenate((stop_words, custom_words_to_filter))
        else:
            words_to_filter = custom_words_to_filter

        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text) # replace punctuation/special chars with a space (space instead of blank to prevent update 2-Hong Kong from becoming 2Hong)
        text = " ".join(text.split()) # remove extra whitespace
        text = lemmatizer.lemmatize(text)
        text = tokenizer.tokenize(text)
        text = [word for word in text if word not in words_to_filter]
        text = ' '.join(text)

        return text

    project = hopsworks.login()
    feature_store = project.get_feature_store()
    article_feature_group = feature_store.get_feature_group(name="articles_daily", version=1)
    data = article_feature_group.read()

    data['title_stance'] = data.apply(lambda row : clean_text(row['title'], filter_stop_words=False), axis=1)
    data['title_topic'] = data.apply(lambda row : clean_text(row['title'], filter_stop_words=True), axis=1)

    article_cleaned_feature_store = feature_store.get_or_create_feature_group(
        name="articles_daily_cleaned",
        version=1,
        primary_key=["url"],
        description="Articles loaded in daily that have gone through pre-processing"
    )
    article_cleaned_feature_store.insert(data, write_options={"wait_for_job" : False})

@stub.function(image=topicextraction_image, secret=modal.Secret.from_name("hopswork-api-key"))
def topic_extraction():
    from bertopic import BERTopic
    from sklearn.cluster import AgglomerativeClustering
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    import hopsworks

    project = hopsworks.login()
    feature_store = project.get_feature_store()
    article_feature_group = feature_store.get_feature_group(name="articles_daily_cleaned", version=1)
    data = article_feature_group.read()

    sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    embeddings = sentence_model.encode(data['title_topic'])

    cluster_model = AgglomerativeClustering(linkage='ward', distance_threshold=1.5, n_clusters=None)
    topic_model = BERTopic(hdbscan_model=cluster_model).fit(data['title_topic'], embeddings)
    topics, probs = topic_model.fit_transform(data['title_topic'])

    topic_labels = topic_model.generate_topic_labels(
        nr_words=3,
        topic_prefix=False,
        word_length=15,
        separator=", "
    )
    topic_model.set_topic_labels(topic_labels)

    topic_model.get_topic_info()

    topic_labels_series = pd.Series(topic_labels)
    docs_topic = topic_labels_series[topics].tolist()
    data['predicted_topic'] = docs_topic

    article_cleaned_feature_store = feature_store.get_or_create_feature_group(
        name="articles_topic",
        version=1,
        primary_key=["url"],
        description="Articles with predicted topic"
    )
    article_cleaned_feature_store.insert(data, write_options={"wait_for_job" : False})

@stub.function(image=stance_image, secret=modal.Secret.from_name("hopswork-api-key"))
def stance_predictions():
    import numpy as np
    import pandas as pd
    import keras

    #TODO(Model path from The Hub)
    model_path = ""
    model = keras.models.load_model(model_path)

    # Data from Hopsworks
    project = hopsworks.login()
    feature_store = project.get_feature_store()
    article_feature_group = feature_store.get_feature_group(name="articles_topic", version=1)
    data = article_feature_group.read()

    # inference
    probs = model.predict(
        x=[
            data['title_stance'],
            data['predicted_topic']
        ]
    )
    predicted_class = np.argmax(probs, axis=1)
    data['predicted_stance'] = predicted_class

    #TODO(Check datatype compatbility with other feature groups)
    article_stance_feature_store = feature_store.get_or_create_feature_group(
        name="articles_stance",
        version=1,
        primary_key=["url"],
        description="Articles with predicted stance"
    )
    article_stance_feature_store.insert(data, write_options={"wait_for_job" : False})

@stub.function(schedule=modal.Period(days=1))
def daily_pipeline():
    get_daily_articles_and_store()
    clean_and_store_daily_articles()
    topic_extraction()
    stance_predictions()

if __name__ == "__main__":
    # Programatic deployment of daily schedule
    stub.deploy("daily_pipeline")
