from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import string

ref_file      = 'assignment/2-mt-evaluation/reference.txt'
system1_file  = 'assignment/2-mt-evaluation/system1.txt'
system2_file  = 'assignment/2-mt-evaluation/system2.txt'


n_sent              = 0;
cum_bleu            = 0.0;

# n-gram smoothing function
# Default is method0 = no smoothing
smoothing_func  = SmoothingFunction().method1

no_punctuation  = False

# Loop over eval systems and copute the macro-average BLEU.
# We could as well use the corpus_bleu function to eval the whole thing.
for hyp_file_idx, hyp_file in enumerate([ system1_file, system2_file ], start=1):
    with open(ref_file, 'rt') as rf:
        with open(hyp_file, 'rt') as hf:
            # read line by line from each file and copute the sentence BLEU score
            while True:
                ref_line = rf.readline()
                hyp_line = hf.readline()

                if (not ref_line or not hyp_line):
                    if (ref_line != hyp_line):
                        print("Warning: it seems like line-count for reference file does not match hypothesis file")
                    break
                    
                n_sent += 1

                if not no_punctuation:
                    # For sanity, let's get rid of the punctuation
                    # We could as well perhaps do other fancy normalizing stuff, like case normalization, etc...
                    ref_line = ref_line.translate(str.maketrans('', '', string.punctuation))
                    hyp_line = hyp_line.translate(str.maketrans('', '', string.punctuation))

                # of course, we could compute other metric here, such as METEOR
                sent_bleu   = sentence_bleu([ ref_line.split() ], hyp_line.split(), smoothing_function = smoothing_func)
                cum_bleu   += sent_bleu

    # report
    print("Average BLEU score for system {} is {}".format(hyp_file_idx, cum_bleu/float(n_sent)))


            
        
