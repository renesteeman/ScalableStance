# ScalableStance
## Goal
This project aims to use data from news articles to predict both the topic and the stance towards that topic. The data is collected from live news sources, 
of which the title is used to make the predictions (full article content is not available). 

## ML Architecture
The pipeline contains 2 major ML systems, the first predicts the topic without being given a list of possible topics nor the amount of (expected) topics. It then returns
3 keywords that describe the topic, which are shared amongst similar articles. The second system takes in the title and the predicted topic to determine if the article
is in favor/positive, neutral, or against/negative towards the given topic. The result of pipeline is a table consisting of the title, pre-processed titles with a different version for each ML system, topic, and stance. 

## Modal and Hugging Face specifications
There are 3 deployments on Modal:
1. Daily pipeline: Get new articles, preprocess, extract topic, predict stance based on best ML model version and store these results in their respective features groups on Hopsworks.
2. Biweekly pipeline: Train ML model on new training data and store the trained model on the Model Registry on Hopsworks.
3. Webhook: Endpoint that returns JSON with 10 most recent articles with predicted stance from Hopsworks.

Static user-interface on Hugging Face that uses the fetch JavaScript API to make a call to the Modal webhook, which displays the article title with link, publication date, predicted topics/categories, and the predicted stance.

For more information about the architecture, please have a look at "Scalable Stance Diagram" which also further explains how the algorithms function.
