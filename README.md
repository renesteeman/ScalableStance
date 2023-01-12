# ScalableStance
This project aims to use data from news articles to predict both the topic and the stance towards that topic. The data is collected from live news sources, 
of which the title is used to make the predictions (full article content is not available). 

The pipeline contains 2 major ML systems, the first predicts the topic without being given a list of possible topics nor the amount of (expected) topics. It then returns
3 keywords that describe the topic, which are shared amongst similar articles. The second system takes in the title and the predicted topic to determine if the article
is in favor/positive, neutral, or against/negative towards the given topic. 

The result of pipeline is a table consisting of the title, pre-processed titles with a different version for each ML system, topic, and stance. It automatically
gathers new news data daily, along with triggering the rest of the pipeline, including the re-training of the ML systems in case of updated training data. The results
are then displayed in a web UI that shows the article's title, predicted topic, and stance.

For more information about the architecture, please have a look at "Scalable Stance Diagram" which also further explains how the algorithms function.
