import nltk
import sys
import os
import string
import numpy as np
import functools

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    result = dict()
    filenames = os.listdir(directory)
    for filename in filenames:
        path = os.path.join(directory, filename)
        sentence = ""
        with open(path, "r", encoding="UTF-8") as file:
            for f in file.readlines():
                sentence += f.strip() + " "
            result[filename] = sentence

    return result


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    temp = nltk.word_tokenize(document)
    words = [t.lower() for t in temp]
    result = []
    for word in words:
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            result.append(word)
            
    return result


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    result = dict()
    n = len(documents)
    temp_docs = dict()
    all_words = set()
    for filename in documents:
        temp_docs[filename] = set(documents[filename])
        all_words.update(temp_docs[filename])
        
    for word in all_words:
        count = 0
        for filename in temp_docs:
            if word in temp_docs[filename]:
                count += 1
        result[word] = np.log(n / count)
        
    return result


def compute_tf(word, words):
    count = 0
    for w in words:
        if w == word:
            count += 1
            
    return count
    
    
def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    temp = []
    for filename in files:
        score = 0
        for word in query:
            if word in idfs:
                score += compute_tf(word, files[filename]) * idfs[word]
            temp.append((filename, score))
            
    # Sorting with the a variable key(lambda), criteria x, highest score desc)
    temp.sort(key=lambda x: -x[1])
    result = []
    for i in range(n):
        result.append(temp[i][0])

    return result
    
    
def compute_count(word, sentence):
    score = 0
    for s in sentence:
        if s == word:
            score += 1
            
    return score / len(sentence)
    
    
def cmp(a, b):
    if a[1] != b[1]:
        return b[1] - a[1]
    else:
        return b[2] - a[2]
        
        
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    temp = []
    for sentence in sentences:
        s1 = 0
        s2 = 0
        for word in query:
            if word in sentences[sentence]:
                s1 += idfs[word]
            s2 += compute_count(word, sentences[sentence])
        temp.append((sentence, s1, s2))
        
    temp = sorted(temp, key=functools.cmp_to_key(cmp))
    
    result = []
    for i in range(n):
        result.append(temp[i][0])
        
    return result
    
    
if __name__ == "__main__":
    main()
