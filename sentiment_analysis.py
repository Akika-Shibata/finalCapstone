# Load pandas, SpaCy, textblob, wordcloud, pyplot, and defaultdict 

import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict

# Load spaCy's small-sized English model
nlp = spacy.load('en_core_web_sm')

# Load Amazon dataset (source: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
df = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv',low_memory=False)


###### Data Quality Checks ######

# Checking column names
print(f"This is a list of column names: {list(df.columns)}")

# Checking the brands listed in the data file (checking all files are labelled Amazon)
print(f"\n\nThe list of brands in this file is: {df['brand'].unique()}")

# Checking all items are similar categories i.e. electronics
print(f"\n\nThe primary categories of the items are: {df['primaryCategories'].unique()}")

# Data Quality check: visual check to make sure product names are aligned with the categories (data entry issue)
print(f"\n\nThese are the product names in the file: {df['name'].unique()}")



###### Data Preprocessing ######

#Selecting the reviews and the titles of the products
reviews_data = df[['reviews.text','reviews.title']]
print(f"\n\nThe count of the number of reviews are: \n{reviews_data.count()}")

# Drop data that has missing reviews (reviews.text)
cleaned_data = reviews_data.dropna()

# Selecting just the reviews
text = cleaned_data['reviews.text']



###### Processing the data with NLP ######


# This function convert the letters to lowercase and stripping leading or trailing white spaces.
# This text gets passed through spaCy's NLP pipeline for processing.
# lemmatizing the tokens, and removing stop words and punctuation from the reviews

def preprocess(text):
    doc = nlp(text.lower().strip())
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])


# Append a column called "processed.text" which "reviews.text" has gone under the "preprocess" function
cleaned_data['processed.text'] = text.apply(preprocess)

# New variable name "processed" which has the new processed text
processed = cleaned_data['processed.text']


###### Analyzing the sentiment of a review using the polarity score ######

# Choosing one sample review (index value). 
text = processed[1]

# Function "analyze_polarity" which takes a review ('text') and analyzes sentiment with TextBlob 
def analyze_polarity(text):

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    return polarity

# Running the sample review through the 'analyze_polarity' function to calculate the polarity score
polarity_score = analyze_polarity(text)

# Calculates the sentiment based on the polarity score 
if polarity_score > 0:
    sentiment = 'positive'
elif polarity_score < 0:
    sentiment = 'negative'
else:
    sentiment = 'neutral'

# Printing out the title of the review, the original review, lemmatized text which was used for analysis, polarity score, and sentiment of the review.
print(f"Title of the review: " + str(cleaned_data['reviews.title'].iloc[1]) )
print(f"Original review: " + str(cleaned_data['reviews.text'].iloc[1]) )
print(f"Lemmatized text used for analysis: {text}\nPolarity score: {polarity_score}\nSentiment: {sentiment}")


###### Testing the model at different indices ######

# Amend Index value on lines 70, 92, and 93

# Index[100]
# Title of the review: Awesome having Show!
# Original review: I have one Alexa and three Echo dots and having Echo Show now is awesome!
# Lemmatized text used for analysis: alexa echo dot have echo awesome
# Polarity score: 1.0
# Sentiment: positive


# Index[1]
# Title of the review: Great light reader. Easy to use at the beach
# Original review: This kindle is light and easy to use especially at the beach!!!
# Lemmatized text used for analysis: kindle light easy use especially beach
# Polarity score: 0.2777777777777778
# Sentiment: positive


###### Semantic Similarity between a selection of Reviews ######

# This requires the medium English model (the small model has no word vector loaded, 
# so the result of the Doc.similarity method will be based on the tagger, parser and NER
# which may not give useful similarity jodgements.)

# Load spaCy's small-sized English model
nlp = spacy.load('en_core_web_md')

#review_to_compare = processed[5]
selection_of_reviews = processed[300:305]

# Comparing the semantic similarity of reviews with each other. 
print("\nSemantic Similarity of the reviews:")
for token in selection_of_reviews:
    token = nlp(token)
    for token_ in selection_of_reviews:
        token_ = nlp(token_)
        print(token.similarity(token_))

# Returning the tokenized sentences used for the semantic similarity above
print("/nTokenized sentences used to calculate the semantic similarity scores: ")
for sentences in processed[300:305]:
    print(sentences)

# Output to be discussed in the 'sentiment_analysis_report' in a separate document.
# Next steps: add word cloud of positive and negative reviews