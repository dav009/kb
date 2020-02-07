
from kb.knowbert import BertPretrainedMaskedLM, KnowBert
from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator
from kb.bert_pretraining_reader import BertPreTrainingReader
from kb.include_all import ModelArchiveFromParams
from kb.include_all import TokenizerAndCandidateGenerator

from allennlp.data import DatasetReader, Vocabulary, DataIterator

from allennlp.models.archival import load_archive
from allennlp.common import Params

import torch

def print_result(vocab, linking_scores, candidate_spans, candidate_entities):
    max_candidate_score, max_candidate_indices = linking_scores.max(dim=-1)
    above_threshold_mask = max_candidate_score > 0.0
    extracted_candidates = candidate_spans[above_threshold_mask]
    candidate_entities_for_extracted_spans = candidate_entities[above_threshold_mask]
    extracted_indices = max_candidate_indices[above_threshold_mask]
    batch_size, num_spans, _ = linking_scores.shape
    batch_indices = torch.arange(batch_size).unsqueeze(-1).repeat([1, num_spans])[above_threshold_mask.cpu()]
    extracted_entity_ids = []

    for k, ind in enumerate(extracted_indices):
        extracted_entity_ids.append(candidate_entities_for_extracted_spans[k, ind])

    # make tuples [(span start, span end), id], ignoring the null entity
    ret = []
    for start_end, eid, batch_index in zip(
                extracted_candidates.tolist(),
                extracted_entity_ids,
                batch_indices.tolist()
    ):
        entity_id = eid.item()
        resolved_entity = vocab.get_token_from_index(entity_id, 'entity')
        ret.append((batch_index, tuple(start_end), entity_id, resolved_entity))
    return ret


model_archive_file= "/home/dav009/code/idio/kb/knowbert_wiki_model.tar.gz"
archive = load_archive(model_archive_file)
params = archive.config
vocab = Vocabulary.from_params(params.pop('vocabulary'))
model = archive.model
reader_params = Params({
    "type": "aida_wiki_linking",
    "entity_disambiguation_only": False,
    "token_indexers": {
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased",
            "do_lowercase": True,
            "use_starting_offsets": True,
            "max_pieces": 512,
        },
    },
    "entity_indexer": {
       "type": "characters_tokenizer",
       "tokenizer": {
           "type": "word",
           "word_splitter": {"type": "just_spaces"},
       },
       "namespace": "entity",
    },
    "should_remap_span_indices": True,
})
reader = DatasetReader.from_params(Params(reader_params))
iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 16}))
iterator.index_with(vocab)


from flask import Flask
from flask import request, jsonify
app = Flask(__name__)


def annotate(text):
    instances = reader.read(text)
    for batch_no, b in enumerate(iterator(instances, shuffle=False, num_epochs=1)):
        b['candidates'] = {'wiki': {
                'candidate_entities': b.pop('candidate_entities'),
                'candidate_entity_priors': b.pop('candidate_entity_prior'),
                'candidate_segment_ids': b.pop('candidate_segment_ids'),
                'candidate_spans': b.pop('candidate_spans')}}
        print("runnning against model")
        result = model(**b)
        linking_score  = result['wiki']['linking_scores']
        candidate_spans = b['candidates']['wiki']['candidate_spans']
        candidate_entites =  b['candidates']['wiki']['candidate_entities']['ids']
        print("decoding")
        # model.soldered_kgs['wiki'].entity_linker._decode(linking_score, candidate_spans,candidate_entites) 
        results = print_result(vocab, linking_score, candidate_spans, candidate_entites)
        candidates = candidate_entites.tolist()
        resolved_candidates = []
        for c in candidates:
            for e in c:
                candidates_for_span = []
                for ent in e:
                    resolved_entity = vocab.get_token_from_index(ent, 'entity')
                    if resolved_entity != "@@PADDING@@":
                        candidates_for_span.append(resolved_entity)
                resolved_candidates.append(candidates_for_span)
            
        return results, resolved_candidates


@app.route('/')
def annotate_endpoint():
    text  = request.args.get('text')
    results, candidates = annotate(text)
    print(text)
    for r in results:
        print(r)
    print("======")
    return jsonify({'annotation': results, 'candidates': candidates})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4444)

