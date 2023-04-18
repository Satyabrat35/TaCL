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

# revisit
def create_instance_from_document(all_documents, document_index, max_seq_len, short_seq_prob=0.1):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]
    # Accounts for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_len - 3
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or random.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # check for random doc index
                    for _ in range(10):
                        random_document_index = random.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments

                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pairs(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1
                instance = (tokens_a, tokens_b, is_random_next)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances
