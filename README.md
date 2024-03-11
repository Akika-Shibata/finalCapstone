Sentiment Analysis of Amazon product reviews

Introduction.

Sentiment analysis is an important and powerful tool which provides valuable insights by detecting positive, negative, or neutral emotions based on a piece of text using natural language processing (NLP). Ths is highly relevant in our daily lives, such as in businesses (analyzing customer feedback of product, brand, services, or campaigns), analyzing sentiment on social media and news topics, and so on.


This programme.

In this programme, it analyses the sentiment of Consumer Reviews of Amazon Products obtained from Kaggle 
(source: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products). 

The dataset used for this is: ‘Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv’ which contains 5000 records of Amazon reviews of various Amazon devices. 

The dataset goes through data quality checks and it is cleansed, ready for analysis. Stop words and punctuations are removed from the reviews. The reviews were then processed with spaCy and then the tokens were lemmatized. 

Analyze_polarity function was created which analyses the sentiment of the text using TextBlob. The polarity score was calculated, and the sentiment was carried based on the returned polarity score. The polarity score ranges from -1 to 1 whereby if the score was greater than 0, then the sentiment was positive; if the polarity score was less than 0, the sentiment was negative. Score of 0.0 indicates it is neutral. Stronger sentiment will be further away from 0.0.

Semantic similarities of the reviews were calculated against each other to gain an understanding of how similar the reviews were. For this, the medium English model was used as the small model has no word vector loaded. The reviews were processed with spaCy and the tokens were used for similarity analysis. The value ranges from 1 to 0, whereby 1 means it is exactly the same and 0 means it is not siimlar at all.



Installations needed to run this project (Windows users).

py -m pip install --user spacy textblob

py -m pip install --user matplotlib

py -m pip install --user pandas

py -m pip install --user wordcloud

py -m spacy download en_core_web_sm

py -m spacy download en_core_web_md
