# import unidecode
# import contractions
# from nltk.tokenize import TreebankWordTokenizer
# from nltk.stem import SnowballStemmer
# import pandas as pd

# tbt = TreebankWordTokenizer()
# ss = SnowballStemmer('english')

# # remove non-ascii chars - we explicitly have str.isspace() even though it is an ASCII char
# def ascii_filter(text: str) -> str:
#     return ''.join([c for c in text if c.isascii() or c.isspace()])

# def clean_series(strings: pd.Series) -> pd.Series:
#     strings = strings.copy()
#     strings = strings.apply(ascii_filter)
#     # fix hashtags
#     strings = strings.str.replace(r"# ([A-Za-z]+[A-Za-z0-9]*)", r' #\1', regex=True)
#     # replace hashtags with token
#     strings = strings.str.replace(r"(#[A-Za-z]+[A-Za-z0-9]*)", 'hashtag', regex=True)
#     # replace usernames
#     strings = strings.str.replace(r"(@[^\s:]+)", 'user', regex=True)
#     # replace all urls with link
#     strings = strings.str.replace(r"(http[^\s]+)", 'link', regex=True)
#     # fix unicode anomalies
#     strings = strings.apply(unidecode.unidecode)
#     # fix contractions
#     strings = strings.apply(lambda msg: contractions.fix(msg, slang=False))
#     # remove any non alphabetic chars and replace with spaces
#     strings = strings.str.replace(r"([^A-Za-z0-9 ])+", ' ', regex=True)
#     # remove any instance of more than 1 whitespace, replace with single space
#     return strings.str.replace(r"\s{2,}", ' ', regex=True).str.strip()