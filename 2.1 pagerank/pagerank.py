import os
import random
import re
import sys
import numpy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Number of pages
    n = len(corpus)

    # Result dictionary for dist
    res = dict()
    
    # Number of links in the particular page
    m = len(corpus[page])

    for i in corpus:
        
        # If the page is not empty or the page has links to other pages
        if m != 0:
            
            '''
            With probability damping_factor, the random surfer should randomly choose one of the links 
            from page with equal probability.
            With probability 1 - damping_factor, the random surfer should randomly choose one of all pages 
            in the corpus with equal probability.
            '''
            
            # If i is in the page. we have damping_factor prob + random prob
            if i in corpus[page]:
                res[i] = (1 - damping_factor) / n + damping_factor / m
            
            # Else just have random prob 
            else:
                res[i] = (1 - damping_factor) / n
        
        # If page is empty, equally divide by all the pages
        else:
            res[i] = 1 / n
    
    # Return the dictionary that contains the prob dist for all the pages. 
    return res


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    res = dict()
    for i in corpus:
        res[i] = 0
        
    # Randomly choose the first page to cursor
    # Use .keys() because we want the page itself, not the values in the page a.k.a .values()
    page = random.choice(list(corpus.keys()))
    
    # Populate n dictionaries and choose the next page for after each new page has been chosen
    # This i represents the order of the number of pages. i = 5 means res[page_number] for the 5th time.
    # The result is n pages that could either be any pages from the corpus, i.e (4) 1.html & (3) 2.html pages 
    for i in range(n):
        res[page] += 1
        prob = transition_model(corpus, page, damping_factor)
        key = list(prob.keys())
        value = list(prob.values())
        
        # Non-uniform random sample with the probability gained from the values in the page, not the corpus.
        page = numpy.random.choice(key, p=value)

    # Calculate the percentage of apperance for each page (corpus.keys()) in the corpus
    # The result is like 0.4 for 1.html for each iteration. 
    # i = 1 as in page 1.html. res[i] = res[1] is the percentage of page 1.html. 
    for i in corpus:
        res[i] /= n

    # res is the dictionary of all pages. Return res means returning a dictionary of percentages for each page.
    return res
    

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    numlinks = dict()
    corpus_reverse = dict()
    for i in corpus:
        corpus_reverse[i] = set()
        
        # If page is empty, set the links in page to be all the pages of the corpus since random prob dist
        # If len(corpus[i]) == 0 then len(corpus[i]) == len(corpus)
        if len(corpus[i]) == 0:
            corpus[i] = set(corpus.keys())
    
    # If number_of_links(i) == 0 then number_of_links(i) == len(corpus). This won't happen because above
    # Else number_of_links(i) == len(corpus[i])
    # Page link traversal
    for i in corpus:
        for j in corpus[i]:
            corpus_reverse[j].add(i)
        numlinks[i] = len(corpus[i])

    # n is the number of pages in the corpus
    n = len(corpus)
    page = dict()
    for i in corpus:
        page[i] = 1/n
        
    # PR(p) = (1 - damping_factor) / len(corpus) + damping_factor * sigma(PR(i) / number_of_links(i))
    while True:
        newpage = dict()
        for i in corpus:
            newpage[i] = (1 - damping_factor) / n
            for j in corpus_reverse[i]:
                newpage[i] += damping_factor * page[j] / numlinks[j]

        s = 0
        flag = True
        for i in corpus:
            diff = abs(newpage[i] - page[i])
            if diff > 0.001:
                flag = False
            page[i] = newpage[i]
            s += page[i]

        if flag:
            break

    return page


if __name__ == "__main__":
    main()
