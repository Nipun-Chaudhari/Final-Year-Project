import pandas as pd
import matplotlib.pyplot as plt
import warnings, string

warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

df = pd.read_csv('data.csv')
print("FIRST FEW RECORDS IN DATA\n", df.head())

cols = df.columns
print("\nFIRST COLUMN\n", cols[0])

# Dropping unnecessary columns from data
df = df.drop(cols[0], axis=1)
print("\nDATA AFTER REMOVING UNNECESSARY COLUMN\n", df.head())

# Adding new column 'length' to data which shows the length of the review
df['length'] = df['text_'].apply(len)
print('\nDATA INFORMATION\n', df.info())
print('\nFIRST FEW RECORDS IN DATA\n', df.head())

# Plotting the 'length' feature
fig = plt.hist(df['length'], bins=10)
plt.xlabel('Length')
plt.ylabel('Reviews')
plt.show()

# Data Description
print('\nDATA DESCRIPTION(grouped by label)\n', df.groupby('label').describe())

# Removing punctuations

# def text_process(review):
#     tokens = nltk.word_tokenize(review)
#     punct_removed = [token for token in tokens if token not in string.punctuation]
#     return ' '.join(punct_removed)


# Creating Bag of Words
bag_of_words = CountVectorizer()
bag_of_words.fit(df['text_'])
print("Vocabulary:", len(bag_of_words.vocabulary_))
review = df['text_'][10]
print("\nRANDOM REVIEW\n", review)

# Testing bag of words
bow_msg = bag_of_words.transform([review])
print(bow_msg)
print(bow_msg.shape)
# print(bag_of_words.get_feature_names()[2108])
# print(bag_of_words.get_feature_names()[15942])

# Applying bag of words
bow_transformed_reviews = bag_of_words.transform(df['text_'])
print("Bag of words transformed review corpus shape: ", bow_transformed_reviews.shape)
print("Non-zero values in the bag of words model:", bow_transformed_reviews.nnz)

# TFIDF
tfidf_transformer = TfidfTransformer().fit(bow_transformed_reviews)
tfidf_rev4 = tfidf_transformer.transform(bow_msg)
print(bow_msg)
tfidf_reviews = tfidf_transformer.transform(bow_transformed_reviews)
print("Shape:", tfidf_reviews.shape)
print("No. of Dimensions:", tfidf_reviews.ndim)
