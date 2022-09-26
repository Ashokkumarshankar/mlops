from cProfile import label
import json
import re
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from configparser import ConfigParser
import nltk
from sklearn.preprocessing import LabelEncoder


def replace_oos_labels(df : pd.DataFrame,label_columns: str, labels : list, other_labels:str = "other"):
    """
    Replace lables which are not in our scope

    Args:
        df  -  pandas dataframe
        labels  - list of labels
        other_labels - labels which aren't in our scope
        label_columns -  the columns name in the dataset to check of labels
    """

    _tags=[label for label in df[label_columns].unique() if label not in labels]
    df[label_columns]=df[label_columns].apply(lambda x : other_labels if  x in _tags else x)
    return df

def replace_minority_labels(df : pd.DataFrame, label_columns: str, frequency: int, new_label:str="others"):
    """
    Replace minority lables with other labels

    Args:
        df  -  pandas dataframe
        frequency  - minimum number of labels the dataset should have in order to take the label
        new_label - the value of labels which are minimum in number
        label_columns -  the columns name in the dataset to check of labels
    """

    labels=Counter(df[label_columns].values)
    labels_frequency=Counter(label for label in labels.elements() if (labels[label] >= frequency))
    df[label_columns]=df[label_columns].apply(lambda lab : lab if lab in labels_frequency else None)
    df[label_columns]=df[label_columns].fillna(new_label)


def clean_text(text: str):

    """
    Cleaning the raw text

    Args:
    text - the text to clean
    optional:
    stem -  stemmer
    lower -  boolean (lower case or uppercase)

    """

    lst_stopwords = nltk.corpus.stopwords.words("english")
    text = text.lower()

    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)

    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  
    text = re.sub("[^A-Za-z0-9]+", " ", text) 
    text = re.sub(" +", " ", text) 
    text = text.strip()

    text = re.sub(r"http\S+", "", text)
    stemmer = PorterStemmer()
    text = " ".join([stemmer.stem(word) for word in text.split(" ")])
    return text



def preprocess(df: pd.DataFrame, min_freq: int):
    """ Preprocess the data.
    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        lower (bool): whether to lowercase the text.
        stem (bool): whether to stem the text.
        min_freq (int): minimum # of data points a label must have.
    Returns:
        pd.DataFrame: Dataframe with preprocessed data.
    """

    df["text"] = df['Short description'] + " " + df.description
    df.text = df.text.apply(clean_text)
    df = replace_oos_labels(
        df=df, label_col="Code", oos_label="other"
    ) 
    df = replace_minority_labels(
        df=df, label_col="tag", min_freq=min_freq, new_label="other"
    )  

    return df


def get_data_split(X: pd.Series, y: np.ndarray, train_size: float = 0.7):
    """Generate balanced data splits.
    Args:
        X : input features.
        y : encoded labels.
        train_size (float, optional): proportion of data to use for training. 
    Returns:
        Tuple: data splits as Numpy arrays.
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test


encoder = LabelEncoder()
