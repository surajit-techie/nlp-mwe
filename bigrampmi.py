import math
import nltk
from collections import defaultdict

def generateUnigramsInMovie(Tokens,freqThreshold):          
    unigrams_in_movie=defaultdict(int)          
    fdistUnigrams = nltk.FreqDist(Tokens)
    for unigram, freq in sorted(fdistUnigrams.iteritems(), key=lambda (k,v): (v,k)):
        if freq > freqThreshold:
            unigrams_in_movie[unigram] = freq
    return unigrams_in_movie

def generateBigramsInMovie(Tokens,freqThreshold):
        bigrams_in_movie=defaultdict(int)

        b = nltk.collocations.BigramCollocationFinder.from_words(Tokens)
        b.apply_freq_filter(freqThreshold)
        for bigram, freq in b.ngram_fd.items():

                bigram=" ".join([bigram[0], bigram[1]])
                bigrams_in_movie[bigram] = freq
        return bigrams_in_movie


#This method is copied from the code given by "alvas"
#Taken from this project: Multi-Word Expression (MWE) extractor from the "Terminator" project   
#Liling Tan. 2013. Terminator - Terminology Extraction to Improve 
#Machine Translation [Software]. Available from
#https://github.com/alvations/Terminator.

def pmi(word1, word2, unigram_freq, bigram_freq):

    prob_word1 = unigram_freq[word1] / float(sum(unigram_freq.values()))
    prob_word2 = unigram_freq[word2] / float(sum(unigram_freq.values()))
    prob_word1_word2 = bigram_freq[" ".join([word1, word2])] / float(sum(bigram_freq.values()))

    try:

        return math.log(prob_word1_word2/float(prob_word1*prob_word2),2)

    except: # Occurs when calculating PMI for Out-of-Vocab words.

        return 0



with open('paulgraham.txt') as wordfile:
    text = wordfile.read()
Tokens = nltk.word_tokenize(text)

unigrams_in_movie= generateUnigramsInMovie(Tokens,1)
bigrams_in_movie=  generateBigramsInMovie(Tokens,1)

b = nltk.collocations.BigramCollocationFinder.from_words(Tokens)
b.apply_freq_filter(1)
bigram_measures = nltk.collocations.BigramAssocMeasures()
bestBigrams=b.nbest(bigram_measures.pmi, 50) 
#I guess that this is what you are looking for it prints the bigram along with its frequency
for bigram in bestBigrams:
    bigram=" ".join([bigram[0], bigram[1]])

    bigrmaFreq=bigrams_in_movie[bigram]
    print str(bigram) +" "+str(bigrmaFreq)

# Then if you want the pmi score for a certain bigram use this :
#As stated before this method is copied from the code given by "alvas"
#print pmi(word1, word2, unigrams_in_movie, bigrams_in_movie)
