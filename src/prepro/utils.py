""" Utility functions."""
import re


def str2bool(bool_arg):
    """ Enable flexible value of arguments.

    Args:
        bool_arg (string): user input string to be parse
    """
    if bool_arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif bool_arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clean(x):
    """ Replaces special tokens. """

    REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
             "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}

    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))

    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0
    words = sum(sentences, [])

    return _get_ngrams(n, words)
