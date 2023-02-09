import json

# We are going to use the langid package since this is the easiest to use
# right off-the-shelf
import langid


noisy_corpus_file  = 'assignment/1-data-cleaning/noisy-corpus.json'
out_file           = 'noisy-corpus.with_lid.json'


print("Parsing corpus {}".format(noisy_corpus_file))
corpus = list()

with open(noisy_corpus_file, 'rt') as nf:
    with open(out_file, 'wt') as of:
        for line_idx, line in enumerate(nf):
            data = json.loads(line)
            src_text, tgt_text = data['source'], data['target']

            src_lang_id, src_lang_lh = langid.classify(src_text)
            tgt_lang_id, tgt_lang_lh = langid.classify(tgt_text)

            of.write('{{"source": {}, "target": {}, "source_lid": "{}", "source_lid_llh": {}, "target_lid": "{}", "target_lid_llh": {} }}\n'.format(json.dumps(src_text), json.dumps(tgt_text), src_lang_id, src_lang_lh, tgt_lang_id, tgt_lang_lh))

# this should leave us with 260023 sentences
print("Parsed {} pairs".format(len(corpus)))
print("Written output to {}".format(out_file))

