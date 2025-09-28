import pandas as pd
import re

file_path = "/Users/kk/Downloads/Hashtag Tesla Tweets.csv"
df = pd.read_csv(file_path)

# Keep only tweet text
df = df[['Tweet Text']]
df.rename(columns={'Tweet Text': 'review'}, inplace=True)

# Add empty Sentiment column (to be predicted later)
df['Sentiment'] = None

# Convert to lowercase
df['review'] = df['review'].str.lower()

# Function to remove HTML tags
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub('', text)

# Example test
sample_text = "<html><body><p> Movie 1</p><p> Actor – Aamir Khan</p><p>Click <a href='http://google.com'>here</a></p></body></html>"
print("Before:", sample_text)
print("After :", remove_html_tags(sample_text))

# Apply to your dataset (if needed)
df['review'] = df['review'].apply(remove_html_tags)

print(df.head())
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

text1 = 'Check out my notebook https://www.kaggle.com/campusx/notebook8223fc1abb'
text2 = 'Check out my notebook http://www.kaggle.com/campusx/notebook8223fc1abb'
text3 = 'Google search here www.google.com'
text4 = 'For notebook click https://www.kaggle.com/campusx/notebook8223fc1abb to search check www.google.com'

print(remove_url(text1))
print(remove_url(text2))
print(remove_url(text3))
print(remove_url(text4))

import string
string.punctuation

punc = string.punctuation

def remove_punc(text):
    return text.translate(str.maketrans('', '', punc))

text = "The quick brown fox jumps over the lazy dog. However, the dog doesn't seem impressed! Oh no, it just yawned. How disappointing! Maybe a squirrel would elicit a reaction. Alas, the fox is out of luck."
text
print(remove_punc(text))

print(df['review'][9])

print(df['review'][9])

# Remove Punctuation
print(remove_punc(df['review'][9]))

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}
def chat_conversion(text):
    new_text = []
    for i in text.split():
        if i.upper() in chat_words:
            new_text.append(chat_words[i.upper()])
        else:
            new_text.append(i)
    return " ".join(new_text)

# Text
text = 'IMHO he is the best'
text1 = 'FYI Islamabad is the capital of Pakistan'
# Calling function
print(chat_conversion(text))
print(chat_conversion(text1))

from textblob import TextBlob

incorrect_text = 'ceertain conditionas duriing seveal ggenerations aree moodified in the saame maner.'
print(incorrect_text)
# Text 2 
incorrect_text2 = 'The cat sat on the cuchion. while plyaiing'
# Calling function
textBlb = TextBlob(incorrect_text)
textBlb1 = TextBlob(incorrect_text2)
# Corrected Text
print(textBlb.correct().string)
print(incorrect_text2)

from nltk.corpus import stopwords

# Get the English stopword list once
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word not in stop_words:   # ✅ use stop_words, not stopwords
            new_text.append(word)
    return " ".join(new_text)

text = "probably my all-time favorite movie, a story of selflessness, sacrifice and dedication"
print("Text With Stop Words:", text)

cleaned = remove_stopwords(text)
print("After Removing Stop Words:", cleaned)

df['review'].apply(remove_stopwords)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

text = "Loved the movie. It was 😘"
text1 = 'Python is 🔥'
print(text ,'\n', text1)

# Remove Emojies using Fucntion
print(remove_emoji(text))
remove_emoji(text1)

import emoji

print(emoji.demojize(text))
print(emoji.demojize(text1))

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize,sent_tokenize

sentence = 'I am going to visit delhi!'
# Calling tool
print(word_tokenize(sentence))

import spacy
nlp = spacy.load("en_core_web_sm")

text = "I love Python. Do you also love NLP? Let's build something cool!"
doc = nlp(text)

# Sentence tokenization
sentences = [sent.text for sent in doc.sents]
print("Sentences:", sentences)

# Word tokenization
words = [token.text for token in doc]
print("Words:", words)

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')   # optional: WordNet multilingual wordnets

from nltk.stem import WordNetLemmatizer
# Intilize Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Sentence 
sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."

# Intilize Punctuation
punctuations="?:!.,;"

# Tokenize Word
sentence_words = nltk.word_tokenize(sentence)

# Using a Loop to Remove Punctuations.
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)
# Printing Word and Lemmatized Word
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word,pos='v')))

    # Apply your preprocessing pipeline step by step on the 'review' column

# 1. Remove HTML tags
df['review'] = df['review'].apply(remove_html_tags)

# 2. Remove URLs
df['review'] = df['review'].apply(remove_url)

# 3. Remove punctuation
df['review'] = df['review'].apply(remove_punc)

# 4. Expand chat abbreviations
df['review'] = df['review'].apply(chat_conversion)

# 5. Remove stopwords
df['review'] = df['review'].apply(remove_stopwords)

# 6. Remove emojis
df['review'] = df['review'].apply(remove_emoji)

# 7. Lemmatization (word by word)
def lemmatize_text(text):
    words = nltk.word_tokenize(text)
    lemmatized = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]
    return " ".join(lemmatized)

df['review'] = df['review'].apply(lemmatize_text)

# Now save the cleaned dataset to CSV
output_path = "/Users/kk/Downloads/Hashtag_Tesla_Tweets_Cleaned.csv"
df.to_csv(output_path, index=False)

print("✅ Preprocessed data saved to:", output_path)
print(df.head())

import re

# Function to clean tweets for FinBERT
def clean_for_finbert(text):
    if pd.isnull(text):
        return ""
    
    # Remove RT (retweets)
    text = re.sub(r'\brt\b', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keep word, drop #)
    text = re.sub(r'#', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters (keep letters and spaces)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply to dataset
df['clean_review'] = df['review'].apply(clean_for_finbert)

# Save cleaned dataset for FinBERT
output_path = "/Users/kk/Downloads/Hashtag_Tesla_Tweets_FinBERT.csv"
df[['clean_review']].to_csv(output_path, index=False)

print("✅ cleaned file saved:", output_path)
df.head(10)

import re
from langdetect import detect, DetectorFactory
from textblob import TextBlob

DetectorFactory.seed = 0  # ensures consistent language detection

# Load your file
file_path = "/Users/kk/Downloads/Hashtag_Tesla_Tweets_FinBERT.csv"
df = pd.read_csv(file_path)

# --- 1. Keep only English tweets ---
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

df = df[df['clean_review'].apply(is_english)]

# --- 2. Clean usernames/handles and gibberish tokens ---
def remove_noise(text):
    if pd.isnull(text):
        return ""
    # Remove usernames/handles (long alphanumeric tokens or fake names)
    text = re.sub(r'\b\w{12,}\b', '', text)  # removes very long words (like carinasteslatray)
    text = re.sub(r'@\w+', '', text)         # remove @handles if any left
    text = re.sub(r'#', '', text)            # remove hashtags (just in case)
    text = re.sub(r'\s+', ' ', text).strip() # normalize spaces
    return text

df['clean_review'] = df['clean_review'].apply(remove_noise)

# --- 3. (Optional) Spell correction ---
def correct_spelling(text):
    return str(TextBlob(text).correct())

# Uncomment below if you want spelling correction (slower for large datasets!)
# df['clean_review'] = df['clean_review'].apply(correct_spelling)

# Save cleaned file for FinBERT
output_path = "/Users/kk/Downloads/Hashtag_Tesla_Tweets_FinBERT_Final.csv"
df.to_csv(output_path, index=False)

print("✅ Final FinBERT-ready dataset saved at:", output_path)
print(df.head(10))



