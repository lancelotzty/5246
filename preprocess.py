from bs4 import BeautifulSoup
import contractions
import emoji
from flashtext import KeywordProcessor
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import spacy
import string
from wordsegment import load, segment
from mapping import contraction_mapping, punct, punct_mapping, mispell_dict

load()
wnl = WordNetLemmatizer()
nlp0 = spacy.load("en_core_sci_sm")
nlp1 = spacy.load("en_ner_bc5cdr_md")

######################################################
# MAIN FUNCTIONS USED BY 04 Data Preprocessing.ipynb #
######################################################

def preprocess(df, text_column='post'):

    df["num_char"] = df[text_column].apply(get_num_char)
    df["text_processed"] = df[text_column] \
        .apply(replace_trailing_space) \
        .apply(lower)
    df["mentioned_usernames"] = df['text_processed'].apply(get_username)
    df["num_mentioned_usernames"] = df["mentioned_usernames"].apply(get_len)
    df["mentioned_urls"] = df[text_column].apply(get_url)
    df["num_mentioned_urls"] = df['mentioned_urls'].apply(get_len)
    df = df.drop(columns=["mentioned_urls"])
    df["mentioned_hashtags"] = df['text_processed'].apply(get_hashtag)
    df["num_mentioned_hashtags"] = df['mentioned_hashtags'].apply(get_len)
    # df["mentioned_smileys"] = df["text_processed"].apply(get_smiley)
    # df["num_mentioned_smileys"] = df["mentioned_smileys"].apply(get_len)
    df["mentioned_haha"] = df["text_processed"].apply(get_haha)
    df["num_mentioned_haha"] = df["mentioned_haha"].apply(get_len)
    df = df.drop(columns=["mentioned_haha"])
    df["mentioned_lol"] = df["text_processed"].apply(get_lol)
    df["num_mentioned_lol"] = df["mentioned_lol"].apply(get_len)
    df = df.drop(columns=["mentioned_lol"])

    df["text_processed"] = df["text_processed"] \
        .apply(replace_html) \
        .apply(replace_username) \
        .apply(replace_url) \
        .apply(replace_hashtag) \
        .apply(replace_emoji) \
        .apply(replace_duplicated_word_groups) \
        .apply(replace_new_line) \
        .apply(replace_ampersand) \
        .apply(replace_zero_width_space) \
        .apply(replace_haha) \
        .apply(replace_lol) \
        .apply(replace_contractions) \
        .apply(replace_only_numbers_in_words) \
        .apply(replace_special_chars) \
        .apply(replace_keywords) \
        .apply(replace_punctuations) \
        .apply(replace_trailing_space) \
        .apply(replace_consecutive_spaces) \
        .apply(lemmatize) \
        .apply(trim)

    df["processed_num_char"] = df["text_processed"].apply(get_num_char)
    df["mentioned_hashtags_cleaned"] = df['mentioned_hashtags'].apply(replace_hash)
    df['num_word'] = df[text_column].apply(get_len)
    df['processed_num_word'] = df['text_processed'].apply(get_len)
    # df['num_mentioned_expressions'] = df['num_mentioned_smileys'] + df['num_mentioned_haha'] + df['num_mentioned_lol']
    df['num_mentioned_expressions'] = df['num_mentioned_haha'] + df['num_mentioned_lol']
    df['date_processed'] = pd.to_datetime(df['date'])
    df['date_day_of_week'] = df['date_processed'].apply(lambda x: x.weekday())
    df['date_month'] = df['date_processed'].apply(lambda x: x.month)
    df['date_year'] = df['date_processed'].apply(lambda x: x.year)
    df['date_hour'] = df['date_processed'].apply(lambda x: x.hour)
    df['date_date'] = df['date_processed'].apply(lambda x: x.date())
    
    df['text_processed'] = df['text_processed'].fillna('')
    
    return df

def get_disease_keyterms(text):
    doc = nlp0(text)
    lem_text_lst = [wd.lemma_ for wd in doc]
    lem_text = ' '.join(lem_text_lst)
    doc = nlp1(lem_text)
    return ' '.join([e.text for e in doc.ents])

##############################################################
# REPLACE HELPER FUNCTIONS USED BY <preprocess> FUNCTION     #
##############################################################

def replace_trailing_space(text):
    text = text.strip()
    text = text.split()
    return " ".join(text)

def lower(text):
    return text.lower()

def replace_html(text):
    text = BeautifulSoup(text, 'lxml').get_text()
    return text

def replace_username(text):
    replacement_text = ' thisisusername '
    text = re.sub('@[^\s]+', replacement_text, text, flags=re.I)
    text = re.sub("@([a-z0-9_]+)", replacement_text, text, flags=re.I)
    return text

def replace_url(text):
    replacement_text = ' thisisurl '
    text = re.sub('http[s]?[^\s]+', replacement_text, text)
    text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', replacement_text, text, flags=re.I)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', replacement_text, text)
    return text

def replace_hashtag(text):
    lst = re.findall('#+[a-zA-Z0-9(_)]{1,}', text, flags=re.I)
    lst = [' '.join(segment(item.replace('#', ''))).lower() for item in lst]
    return re.sub('#+[a-zA-Z0-9(_)]{1,}', '', text) + ' ' + ' '.join(lst)

def replace_emoji(text):
    text = emoji.demojize(text)
    return text

def replace_duplicated_word_groups(text):
    return re.sub(r"\b(\w+)\s+\1\b", "", text, flags=re.I)

def replace_new_line(text):
    text = re.sub(r'\n|\t', '', text)
    return text

def replace_ampersand(text):
    text = re.sub('&amp', '', text)
    return text

def replace_zero_width_space(text):
    text = re.sub('#x200B', '', text)
    return text

def replace_haha(text):
    return re.sub(r"\b(?:a*(?:ha)+h?)\b", "haha", text, flags=re.I)

def replace_lol(text):
    return re.sub(r'\b(?:l+o+)+l+\b', "lol", text, flags=re.I)

def replace_only_numbers_in_words(text):
    return re.sub(r'\b[0-9]+\b\s*', '', text)

def replace_punctuations(text):
    return re.sub(r'[^\w\s]', ' ', text)

def replace_consecutive_spaces(text):
    text = re.sub(' +', ' ', text)
    return text

def lemmatize(text):
    # cleaned_tokens = [wnl.lemmatize(ps.stem(x)) for x in word_tokenize(text)]
    cleaned_tokens = [wnl.lemmatize(str(x)) for x in word_tokenize(str(text))]
    text = " ".join(cleaned_tokens)
    return text

def trim(text):
    return " ".join(str(text).split(" ")[:400])

#####################################################
# CONTRACTION HANDLING                              #
#####################################################

def clean_contractions(text, mapping):
    '''Clean contraction using contraction mapping'''    
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if ""+word+"" in text:
            text = text.replace(""+word+"", ""+mapping[word]+"")
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def replace_contractions(text):
    text = contractions.fix(text)
    text = clean_contractions(text=text, mapping=contraction_mapping)
    return text

#####################################################
# SPECIAL CHARACTER HANDLING                        #
#####################################################

def clean_special_chars(text, punct, mapping):
    '''Cleans special characters present(if any)'''   
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def replace_special_chars(text):
    text = clean_special_chars(text=text, punct=punct, mapping=punct_mapping)
    return text

#####################################################
# KEYWORD HANDLING                                  #
#####################################################

kp = KeywordProcessor(case_sensitive=True)
                
mix_mispell_dict = {}
for k, v in mispell_dict.items():
    mix_mispell_dict[k] = v
    mix_mispell_dict[k.lower()] = v.lower()
    mix_mispell_dict[k.upper()] = v.upper()
    mix_mispell_dict[k.capitalize()] = v.capitalize()
    mix_mispell_dict[k.title()] = v.title()
    
for k, v in mix_mispell_dict.items():
    kp.add_keyword(k, v)
    
def replace_keywords(text):
    text = kp.replace_keywords(text)
    return text

##############################################################
# GET HELPER FUNCTIONS USED BY <preprocess> FUNCTION         #
##############################################################


def get_num_char(text):
    return len(text)

def get_username(text):
    return ' '.join(re.findall("@([a-z0-9_]+)", text, flags=re.I)).lower()

def get_len(text):
    if text == '':
        return 0
    else:
        return len(text.split(' '))

def get_url(text):
    return ' '.join(re.findall('http[s]?[^\s]+', text, flags=re.I)).lower()

def get_hashtag(text):
    return ' '.join(re.findall('#+[a-zA-Z0-9(_)]{1,}', text, flags=re.I)).lower()

def get_haha(text):
    return " ".join(re.findall(r"\b(?:a*(?:ha)+h?)\b", text, flags=re.I))

def get_lol(text):
    return " ".join(re.findall(r'\b(?:l+o+)+l+\b', text, flags=re.I))

def replace_hash(text):
    return re.sub('#', '', text)