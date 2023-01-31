Language Models
=

# Resources that helps me to understand Kneser-Ney Smoothing
* [speech and languge processing book](https://web.stanford.edu/~jurafsky/slp3/)
* [this lecture from the course](https://www.youtube.com/watch?v=4wa2WyDrgMA)
* [lecture form India](https://www.youtube.com/watch?v=NiKGlBb3NTE&list=PLJJzI13YAXCHxbVgiFaSI88hj-mRSoMtI&index=14)
* [this paper review of smoothing techniques in Ngram language Models](http://www.cs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf)


Languge Models
=
As always, check out the Github repository with the course homework templates:

http://github.com/ezubaric/cl1-hw.git

The code for this homework is in the _lm_ directory.  This assignment is worth 40 points.

Preparing Data (10 points)
---
 
We will use the Brown corpus (nltk.corpus.brown) as our training set and the Treebank (nltk.corpus.treebank) as our test set.  Eventually, we'll want to build a language model from the Brown corpus and apply it on the Treebank corpus.  First, however, we need to prepare our corpus.
* First, we need to collect word counts so that we have a vocabulary.  This is done by the _train\_seen_ function.  Modify this function so that it will keep track of all of the tokens in the training corpus and their counts.
* After that is done, you can complete the _vocab\_lookup_ function.  This should return a unique identifier for a word, or a common "unknown" identifier for words that do not meet the _unk\_cutoff_ threshold.  You can use strings as your identifier (e.g., leaving inputs unchanged if they pass the threshold) or you can replace strings with integers (this will lead to a more efficient implementation).  The unit tests are engineered to accept both options.
* After you do this, then the finalize and censor functions should work (but you don't need to do anything).  But check that the appropriate unit tests are working correctly.

Estimation (20 points)
------

After you've finalized the vocabulary, then you need to add training
data to the model.  This is the most important step!  Modify the
_add\_train_ function so that given a sentence it keeps track of the
necessary counts you'll need for the probability functions later.  You
will probably want to use default dictionaries or probability
distributions.  Finally, given the counts that you've stored in
_add\_train_, you'll need to implement probability estimates for
contexts.  These are the required probability estimates you'll need to
implement:
* _mle_: Simple division of counts for that observation by total counts for the context
* _laplace_: Add one to all counts
* _dirichlet_: Add a specified parameter greater than zero to all counts
* _jelinek_mercer_: Interpolate between probability distributions with parameter lambda

Now if you run the main section of the _language\_model_ file, you'll
get per-sentence reports of perplexity.  Take a look at what sentences
are particularly hard or easy (you don't need to turn anything in
here, however).

Exploration (10 points)
----------

Try finding sentences from the test dataset that get really low perplexities for each of the estimation schemes (you may want to write some code to do this).  Can you find any patterns?  Turn in your findings and discussion as \texttt{discussion.txt}.

Extra Credit
------

Extra Credit (make sure they don't screw up required code / functions that will be run by the autograder):
* _kneser\_ney_: Use discounting and prefixes with discount parameter $\delta$ and concentration parameter alpha to implement interpolated Kneser-Ney.
* Implement a function to produce English-looking output (return an iterator or list) from your language model (function called _sample_)
* Make the code really efficient for reading in sequences of characters

FAQ
--------
*Q: Why are there two passes over the data?*

A: The first pass establishes the vocabulary, the second pass accumulates the counts.  You could in theory do it in one pass, but it gets much more complicated.

*Q: What if the counts of \<s\> and \<\/s\> fall below the threshold?*

A: They should always be included in the vocabulary.

*Q: And what about words that fall below the threshold?*

A: They must also be represented, so the vocab size will be the number of tokens at or above the UNK threshold plus three (one for UNK, one for START, and one for END).  

*Q: What happens when I try to take the log of zero?*

A: Return kNEG\_INF instead.

*Q: Do I tune the hyperparameters for interpolation, discount, etc.?*

A: No, that's not part of this assignment.



Top 10 lowwer preplexityies -higer probabilities- with laplace smoothing: (most of them of unkown token)
----
```
(1) 3.6683662567610402   -> IRAs .
<s>#UNK#.#</s>
-----
(2) 4.325582590645588   -> It was outrageous .
<s>#it#was#UNK#.#</s>
-----
(3) 6.620080041752021   -> The '82 Salon is $ 115 *U* .
<s>#the#UNK#UNK#is#$#UNK#UNK#.#</s>
-----
(4) 8.783914718624546   -> They mature 1992-1999 , 2009 and 2017 .
<s>#they#mature#UNK#,#UNK#and#UNK#.#</s>
-----
(5) 9.038148201313476   -> They mature in 2005 , 2009 and 2029 .
<s>#they#mature#in#UNK#,#UNK#and#UNK#.#</s>
-----
(6) 9.732751233562453   -> Test-preparation booklets , software and worksheets are a booming publishing subindustry .
<s>#UNK#UNK#,#UNK#and#UNK#are#a#UNK#publishing#UNK#.#</s>
-----
(7) 10.014733529523305   -> Marie-Louise , a small-time abortionist , was their woman .
<s>#UNK#,#a#UNK#UNK#,#was#their#woman#.#</s>
-----
(8) 10.298798507579201   -> I believe in the system .
<s>#i#believe#in#the#system#.#</s>
-----
(9) 10.630311815783879   -> That was the law .
<s>#that#was#the#law#.#</s>
-----
(10) 11.519723041766555   -> He is his own man .
<s>#he#is#his#own#man#.#</s>
```
