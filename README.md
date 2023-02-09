# tasks4you

This is a little write-up of the solution to the assignments.  I chose to start with the MT task since it looked very straightforward, then I moved over to the cleaning task. 


## MT Task Comments

The task is to find a method of comparing a hypothesis translation to the reference.  For this, we do not need the source, so we will only be working with the target corpora.

At first, I was going to use perplexity between the reference and the hypothesis, but this is not really what MT normally works with, so I chose to use the BLEU score.  The beautiful part is that it is implemented in the NLTK.  I also remembered that people use a METEOR score, but I didn't find a quick way of making it run for Romainan (I believe I would need some kind of stemming and synonymization tools)

It took me about half an hour to polish the code and report the BLEU, but then out of curiosity I spent another half an hour playing with different smoothing functions

The solution is in the run_mt.py script.  Please run it directly with python.  The only dependency here is the MLTK toolkit.

### Result

I had been playing with some of the parameters and tried many different combinations, but the results always correlated.  The following lines are the dump of the script:

    Average BLEU score for system 1 is 0.2985305600899736
    Average BLEU score for system 2 is 0.2581138890699899

These numbers suggest that system 1 is supperior to system 2


## Data Filtration Task

I found this task to be a little more labor intensive.  I know from preparing the ASR corpora that tons of stuff needs to be done.  I taclked the problem from two point of views - the classical common sense (knowledge based) filtering = junk (too much punctuation and numbers), max length, predefined vocabulary, automatic LID decision.

The second approach was ML based.  I wanted to find some interesting heuristics that could be extracted for each pair and that would serve as a feature for some classifier.  I thought that perhaps I could use a BLEU score (or preferably a set of BLEU scores) obtained using some MT system (e.g. run MT on the DE part of the eval pair, and compute BLEU between the EN portion and the DE-EN MT output).  I also chose to use the sentence word count difference between the pairs as another feature.  One could come up with tons of other heuristics that would be suitable for classification.

From the WMT18 evaluation plan it was clear that I could use a limited set of news corpora so I chose to use a subset of the DE-EN Europarl.  Originally, I wanted to train only the classifier on the Europarl, but in the end I used it also for vocabulary filtration and also for very simple MT system training.

I spent some time getting a suitable DE-EN model, but I kept on running into technical issues.  In the end I found a very simple seq2seq MT recipe, so I decided to build my own model.  This turned out to be very simple but rather distracting and adictive - I spent quite some time exploring it and playing around with the model.  

There are tons of things that came to my mind that we could do, e.g. clustering of similar features, using the seq2seq embeddings as feats, use multiple MT systems (both DE-EN and EN-DE), use translation scores as features, etc...  If only there was more time.

The solution is in the data_cleaning_task.ipynb (jupyter notebook), and in mymt.pt which implements the MT model training.  Also note that before running the whole thing, run_lid.py needs to be ran in order to augment the input json file with the LID info.  This was much easier for debugging as LID takes a signifficant amount of time.

### Result

By running the first part, out of the original ~1M pairs, we are left with roughly ~5k pairs.  Please see the following dump for detailed analysis.

    Loading corpus noisy-corpus.with_lid.json
    Loaded 1039251 pairs
    Filtering source vocab
    Left with 12424 pairs
    Filtering LID
    Left with 5032 pairs
    Filtering nonsense
    Left with 5032 pairs
    Filtering very long sentences
    Left with 4954 pairs

After running the ML part, we were left with 4405 pairs.  

    Filtering using classifier
    Left with 4405 pairs

Please note that the training ran only for couple of iterations and for demo purposes, the model was not really trained.


## Time analysis

I am ignoring the time needed to setup the conda environment - I used one of my predefined ones

- 10 minutes reading and understanding the assignment
- 20 minutes to write the basic MT script that loads the data and massages them to the format that NLTK likes for BLEU computation
- 10 minutes adding punctuation filtering
- 30 minutes playing around with NLTK, trying to compute the METEOR, playing with smoothing functions, 
- 20 minutes reading the WMT18 Shared Task: Parallel Corpus Filtering evaluation plan.
- 30 minutes implementing loading of the data and basic filtering
- 30 minutes running LID
- 20 minutes downloading Europarl corpus and loading it to suitable python structure
- 30 minutes looking for various MT models
- 30 minutes running seq2seq example 
- 90 minutes playing with the data and training MT
- 20 minutes implementing the feature extraction 
- 10 minutes implementing implementing Logistic Regression classification and filtering

- 5 minutes setting up GIT

- 30 minutes checking and writing this README

## Running the package

Please note that the assignment directory needs to be coppied/linked to the project root directory.  I used one of my anaconda python environments.  All was run on MacOSX with the M1 processor. Training was run on the machine's CPU as there was an issue with the python M1 implementation (it did run, but the training is slower than on CPU).

### Conda packages

Here's a dump of my conda env export:
name: torch-gpu
channels:
  - defaults
dependencies:
  - python=3.8
  - ipython
  - pytorch
  - torchvision
  - torchaudio
  - jupyterlab
  - jupyter
  - matplotlib
  - scikit-learn

### PIP packages
  - nltk
  -langid

