import os
import torch
from tqdm import tqdm
from pylab import *
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Dict, Any
from collections import Counter
import scipy.stats


class Tokenizor():
    def __init__(self, text_data: Any, maximum_vocab_size: int, min_freq: int = 5):
        self.token_to_id, self.id_to_token = self.get_word_freq_count(
            text_data, maximum_vocab_size, min_freq)
        self.vocab_size = len(self.token_to_id)

    def get_word_freq_count(self, text_data: Any, maximum_vocab_size: int, min_freq: int) -> Tuple[Dict, Dict]:
        count = Counter()
        for line in text_data:
            count.update(line.strip().split())
        # print(len(count.most_common()))
        most_freq_counts = count.most_common()[:maximum_vocab_size-2]
        token_to_id = {token: i + 2 for i,
                       (token, freq) in enumerate(most_freq_counts) if freq > min_freq}
        token_to_id['pad'] = 0
        token_to_id['unk'] = 1
        id_to_token = {val: key for key, val in token_to_id.items()}
        return token_to_id, id_to_token

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = 'False') -> Union[str, List[str]]:
        if ids is None:
            return None

        if isinstance(ids, int):
            return self.id_to_token[ids] if ids in self.id_to_token else 'unk'

        return [self.id_to_token[id] if id in self.id_to_token else 'unk' for id in ids]

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self.token_to_id[tokens] if tokens in self.token_to_id else self.token_to_id['unk']

        return [self.token_to_id[token] if token in self.token_to_id else self.token_to_id['unk'] for token in tokens]


class MetricTracker(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, summed_val, n=1):
        self.val = summed_val / n
        self.sum += summed_val
        self.count += n
        self.avg = self.sum / self.count


# NOTE MODIFICATION (EMBEDDING)
def get_pretrained_weights(glove_path: str, vocab: List, embed_dim, device):
    """
    Returns 50002 words' pretrained weights in tensor
    :param glove_path: path of the glove txt file
    :param corpus_vocab: vocabulary from dataset
    :return: tensor (len(vocab), embed_dim)
    """
    save_dir = 'pretrained_embedding.pt'
    if os.path.exists(save_dir):
        return torch.load(save_dir, map_location=device)

    word_pretrained = {}
    with open(glove_path, 'r') as f:
        for l in f:
            line = l.split()
            word_pretrained[line[0]] = np.array(line[1:], dtype=np.float64)

    word_embedding_weights = []
    for token in vocab:
        if token in word_pretrained:
            word_embedding_weights.append(word_pretrained[token])
        else:
            tmp_val = np.zeros(embed_dim, dtype=np.float64)
            token_count = 0
            for character in token:
                if character in word_pretrained:
                    tmp_val += word_pretrained[character]
                    token_count += 1
                else:
                    tmp_val += np.random.randn(embed_dim)
                    token_count += 1
            word_embedding_weights.append(tmp_val/token_count)
    embeddings = torch.from_numpy(
        np.array(word_embedding_weights, dtype=np.float32))
    torch.save(embeddings, save_dir)
    return embeddings


# NOTE MODIFICATION (FEATURE)
# referenced to https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768
def map_sentence_to_color(words, scores, sent_score):
    """
    :param words: array of words
    :param scores: array of attention scores each corresponding to a word
    :param sent_score: sentence attention score
    :return: html formatted string
    """

    sentencemap = matplotlib.cm.get_cmap('binary')
    wordmap = matplotlib.cm.get_cmap('OrRd')
    result = '<p><span style="margin:5px; padding:5px; background-color: {}">'\
        .format(matplotlib.colors.rgb2hex(sentencemap(sent_score)[:3]))
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    for word, score in zip(words, scores):
        color = matplotlib.colors.rgb2hex(wordmap(score)[:3])
        result += template.format(color, '&nbsp' + word + '&nbsp')
    result += '</span><p>'
    return result


# NOTE MODIFICATION (FEATURE)
def bar_chart(categories, scores, graph_title='Prediction', output_name='prediction_bar_chart.png'):
    y_pos = arange(len(categories))

    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, categories)
    plt.ylabel('Attention Score')
    plt.title(graph_title)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig(output_name)


# # NOTE MODIFICATION (FEATURE)
# def visualize(model, dataset, doc):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     """
#     # Predicts, and visualizes one document with html file
#     :param model: pretrained model
#     :param dataset: news20 dataset
#     :param doc: document to feed in
#     :return: html formatted string for whole document
#     """
#
#     orig_doc = [word_tokenize(sent) for sent in sent_tokenize(doc)]
#     doc, num_sents, num_words = dataset.transform(doc)
#     label = 0  # dummy label for transformation
#
#     doc, label, doc_length, sent_length = collate_fn([(doc, label, num_sents, num_words)])
#     score, word_att_weight, sentence_att_weight \
#         = model(doc.to(device), doc_length.to(device), sent_length.to(device))
#
#     # predicted = int(torch.max(score, dim=1)[1])
#     classes = ['Cryptography', 'Electronics', 'Medical', 'Space']
#     result = "<h2>Attention Visualization</h2>"
#
#     bar_chart(classes, torch.softmax(score.detach(), dim=1).flatten().cpu(), 'Prediction')
#     result += '<br><img src="prediction_bar_chart.png"><br>'
#     for orig_sent, att_weight, sent_weight in zip(orig_doc, word_att_weight[0].tolist(), sentence_att_weight[0].tolist()):
#         result += map_sentence_to_color(orig_sent, att_weight, sent_weight)
#
#     return result
