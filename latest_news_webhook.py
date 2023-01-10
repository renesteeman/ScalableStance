import modal

import hopsworks

stub = modal.Stub("latest_news_webhook")

image = modal.Image.debian_slim().pip_install("hopsworks")

@stub.webhook(method="GET", image=image, secret=modal.Secret.from_name("hopswork-api-key"))
def response_articles():
    project = hopsworks.login()
    feature_store = project.get_feature_store()
    article_feature_group = feature_store.get_feature_group(name="articles_stance", version=1)
    data = article_feature_group.read().drop(0)   # Pandas dataframe on AWS
    articles_show_on_ui = data.sort_values(by='publishedat', ascending=False).head(10)

    return articles_show_on_ui.to_dict(orient='records')

if __name__ == "__main__":
    stub.serve()
