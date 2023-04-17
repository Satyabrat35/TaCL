import unicodedata
import random

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""
    def __init__(self, lower_case=True):
        self.lower_case = lower_case

    def strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            if unicodedata.category(char) == "Mn":
                continue # Nonspacing mark - http://www.unicode.org/reports/tr44/#GC_Values_Table
            output.append(char)

        return "".join(output)

    def split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
            else:
                if start_new_word:
                    output.append([])
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def clean_text(self, text):
        output = []
        for char in text:
            c = ord(char) #ascii value
            if c == 0 or c == 0xfffd or is_control(c):
                continue
            elif is_whitespace(c):
                output.append(" ")
            else:
                output.append(char)

        return "".join(output)

    def tokenize(self, text):
        text = self.clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.lower_case:
                token = token.lower()
                token = self.strip_accents(token)
            split_tokens.extend(self.split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

def whitespace_tokenize(text):
        text = text.strip()
        if not text:
            return []

        return text.split()

def is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True #Space Separator
    return False

def is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True # Control space characters
    return False

def is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def truncate_seq_pairs(token_a, token_b, max_tokens):
    while True:
        total_len = len(token_a) + len(token_b)
        if total_len <= max_tokens:
            break

        trunc_tokens = token_a if len(token_a) > token_b else token_b
        assert len(trunc_tokens) >= 1
        # add randomness
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def create_instance_from_document(all_documents, document_index, max_seq_len, short_seq_prob=0.1):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_len - 3
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    pass