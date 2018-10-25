import re

def preprocess(filename):
    """ 
    Returns the list of sentences (one concatenated string).
    It also removes non alphabetic characters and make every word lowercase.
    <Design Consideration> 1. Should I make every word lowercase?
    """
    list_of_sentences = []
    with open(filename) as f:
        for line in f:
            list_of_sentences.append(re.sub("[^a-zA-Z | ^.]+", "", line).lower() + ' <STOP>')
    return list_of_sentences
