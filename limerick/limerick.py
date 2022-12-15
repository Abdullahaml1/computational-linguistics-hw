# Author: YOUR NAME HERE
# Date: DATE SUBMITTED

# Use word_tokenize to split raw text into words
from string import punctuation

import nltk
from nltk.tokenize import word_tokenize


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
        i=0
        while i<len(words_list):
            if words_list[i] in punctuation:
                del words_list[i]
            else:
                i+=1

        # handling words with apostphe "'" ie: can't
        # the tokenizer will return ["ca", "n't"] we will ->>> ["can't"]
        refined_words_list = []
        for i in range(len(words_list)):
            if "'" in words_list[i]:
                del refined_words_list[-1] # remove last word
                refined_words_list.append("".join(words_list[i-1:i+1]))
            else:
                refined_words_list.append(words_list[i])

        return refined_words_list





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
        words_list = self.split_words_and_remove_punctuation(text)
        A_rhyme = words_list[-1]
        count_A_rhyme =0
        last_A_rhyme_pos=0
        sent_length =4

        print(words_list)
        # Counting A rhyme
        i=sent_length #min sentence length
        while i< len(words_list)-1:
            if self.rhymes(words_list[i], A_rhyme):
                count_A_rhyme +=1
                last_A_rhyme_pos = i
                i +=sent_length # min sentence length
            else:
                i+=1

            if count_A_rhyme ==2:
                break

        # print (count_A_rhyme)
        # print(words_list[last_A_rhyme_pos], ", ", i)
        if count_A_rhyme !=2:
            return False

        # getting B rhyme
        for i in range(last_A_rhyme_pos+1+sent_length, len(words_list)-1, 1):
            for j in range(i+sent_length, len(words_list)-1, 1):
                if self.rhymes(words_list[i], words_list[j]):
                    return True


        return False






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
    # print(ld.split_words_and_remove_punctuation("I'm not can't do.!"))


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
