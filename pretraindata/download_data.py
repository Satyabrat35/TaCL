from preproc import *
from datasets import load_dataset

if __name__ == '__main__':
    dataset = load_dataset('wikipedia', "20220301.en", split='train')
    stop_prefix_list = ['References', 'External links', 'Category:', 'See also']
    all_doc_list = process_corpus(dataset, stop_prefix_list)
    out_f = './english_wiki.txt'
    with open(out_f, 'w', encoding = 'utf8') as o:
        for doc in all_doc_list:
            for sen in doc:
                o.writelines(sen + '\n')
            o.writelines('\n')
