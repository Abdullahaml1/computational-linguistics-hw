# Author: YOUR NAME HERE
# Date: DATE SUBMITTED

# Use word_tokenize to split raw text into words
from string import punctuation

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


class LimerickDetector:

    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.
        (the number of syllables is the number of the vowle sounds in the 
        word)
        """
        if not (word in self._pronunciations):
            return 1
        else:
            phonemes_list = self._pronunciations[word]
            small_count = None # counting vowle sounds
            for word_phonemes in phonemes_list:
                count =0
                for p in word_phonemes:
                    if p[-1].isdigit(): # is a vowle  sound
                        count +=1
                small_count = count if small_count==None else min(count, small_count)


            return small_count


    def is_list_subsetof_list(self, longer_list, shorter_list):
        # if a word start wiht a vowle we'll consider all of it as a rhymes
        # NOte: no two consnent sound can follow each other
        if not longer_list[0][-1].isdigit():
                longer_list = longer_list[1:]
        if not shorter_list[0][-1].isdigit():
                shorter_list = shorter_list[1:]

        # matching phonemes of the two words startng with the first 
        # vowle sound
        for i in range(-1, -len(shorter_list)-1, -1): # lloping backwords
            if shorter_list[i] != longer_list[i]:
                return False

        return True

    
    def is_word_has_vowel(self, word_phonemes):
        """
        word_phonemes: a list of string of CMU pronunciation dictionary
        """
        for sound in word_phonemes:
            if sound[-1].isdigit():
                return True

        return False




    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.
        """
        longer_word = max(a.lower(), b.lower(), key=len)
        shorter_word = min(b.lower(), a.lower(), key=len)

        # the word in not CMPU dictionary
        if (longer_word not in self._pronunciations) or (
                shorter_word not in self._pronunciations):
            return False

        longer_phonemes_list = self._pronunciations[longer_word]
        shorter_phonemes_list = self._pronunciations[shorter_word]

        for longer_phonemes in longer_phonemes_list:
            for shorter_phonemes in shorter_phonemes_list:
                # if one the two words has no vwel shond return false
                if (not self.is_word_has_vowel(longer_phonemes)) or (
                        not self.is_word_has_vowel(shorter_phonemes)):
                    return False
                elif self.is_list_subsetof_list(longer_phonemes,
                        shorter_phonemes):
                    return True

        return False

    def split_words_and_remove_punctuation(self, sent):
        """
        sent: a string
        """
        words_list = word_tokenize(sent)

        #removing punctuatoin
        for char in punctuation:
            if char in words_list:
                words_list.remove(char)

        return words_list





    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other (and not the A
        lines).

        (English professors may disagree with this definition, but that's what
        we're using here.)
        """
        sent_list = sent_tokenize(text)

        # limerick poerm consits of 5 sentences
        if len(sent_list) != 5:
            return False

        rhymes_words_list =[]
        for sent in sent_list:
            # get the last word of the snetence
            rhymes_words_list.append(
                    self.split_words_and_remove_punctuation(sent)[-1])

        return (self.rhymes(rhymes_words_list[0], rhymes_words_list[1]) and
                self.rhymes(rhymes_words_list[0], rhymes_words_list[4]) and
                self.rhymes(rhymes_words_list[1], rhymes_words_list[4]) and
                self.rhymes(rhymes_words_list[2], rhymes_words_list[3])
                )



if __name__ == "__main__":

    # buffer = ""
    # inline = " "
    # while inline != "":
    #     buffer += "%s\n" % inline
    #     inline = input()

    # ld = LimerickDetector()
    # print("%s\n-----------\n%s" % (buffer.strip(), ld.is_limerick(buffer)))


    # """
    # Test num_syllables
    # """
    # print(ld.num_syllables('fire'))


    # """
    # Test rhymes
    # """
    # ld = LimerickDetector()
    # print(ld.rhymes('beard', 'feared'))
    # print(ld.rhymes('nantucket', 'bucket'))
    # print(ld.rhymes('dog', 'cat'))
    # print(ld.rhymes('bagel', 'sail'))
    # # print(ld.is_list_subsetof_list([1,2,3,4], [2, 3, 4]))
    # # print(ld.is_list_subsetof_list([1,2,3,4], [1, 3, 4]))


    # """
    # split words test
    # """
    # ld = LimerickDetector()
    # print(ld.split_words_and_remove_punctuation('This was a nice day.!'))


    """
    lemrick test
    """
    ld = LimerickDetector()
    sent1 = "There once was a man from Nantucket.\
            Who kept all his cash in a bucket.\
            But his daughter, named Nan.\
            Ran away with a man.\
            And as for the bucket, Nantucket."
    print(ld.is_limerick(sent1))
