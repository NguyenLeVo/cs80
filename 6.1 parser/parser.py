import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | NP VP Conj VP
AP -> Adj | Adj AP
NP -> N | Det NP | AP NP | NP PP
PP -> P NP | P S
VP -> V | V NP | V NP PP | V PP | VP Adv | Adv VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))

def judge(word):
    flag = False
    for character in word:
        if character.isalpha():
            flag = True
            break
    
    return flag
    
    
def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = nltk.wordpunct_tokenize(sentence)
    result = []
    for word in words:
        lower_word = word.lower()
        if judge(lower_word):
            result.append(lower_word)
            
    return result
    
    
def np_helper(tree, List):
    if tree == None:
        return
    count = 0
    temp =[]
    for branch in tree.subtrees():
        if branch.label() == "NP":
            temp.append(branch)
            count += 1
    if count == 1:
        branch = temp[0]
        if branch not in List:
            List.append(branch)
        return
    elif count > 1:
        for branch in tree.subtrees():
            if branch != tree:
                np_helper(branch, List)
                
        return
    
    return


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    List = []
    np_helper(tree, List)


if __name__ == "__main__":
    main()
