import string

PUNCTUATION_LIST = string.punctuation


def detokenize(tokens):
    """
    Function to de-tokenize stream of tokens
    :param tokens:
    :return: detokenized text
    """

    process_tokens = lambda token_list, puncts: [" "+token if
                                                 not token.startswith("'") and token not in
                                                 puncts else token for token in token_list]

    processed_tokens = process_tokens(tokens, PUNCTUATION_LIST)
    detokenized_sentence = "".join(processed_tokens).strip()
    return detokenized_sentence

def del_some_symbol_from_mose(sentence):
    sentence = sentence.replace("&bar;", "\|").replace("#124;", "\|").replace("&lt;", "<")
    sentence = sentence.replace("&gt;", ">").replace("&bra;", "[").replace("&ket;", "]").replace("&quot;", "\"")
    sentence = sentence.replace("&#91;", "[").replace("&#93;", "]").replace("&amp;", "\&").replace(" &apos;", "\'")
    return sentence


def my_detokenizer(my_sentences):
    res = []
    for sentence in my_sentences:
        sentence = sentence.split()
        dek_sentence = detokenize(sentence)
        dek_sentence = del_some_symbol_from_mose(dek_sentence)
        res.append(dek_sentence)
    return res


if __name__ == '__main__':
    my_sentences = ["the ethnosphere is humanity &apos;s great legacy .", "the ethnosphere is humanity &apos;s great legacy ."]
    print(my_detokenizer(my_sentences))