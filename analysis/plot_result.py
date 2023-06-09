import json


def process_result(in_f):
    with open(in_f) as f:
        res_dict = json.load(f)
    print(len(res_dict))

    layer_res = {}
    for idx in range(1, 13):
        layer_res[idx] = {'cosine_sum': 0., 'token_sum': 0}

    key_list = list(res_dict.keys())
    for key in key_list:
        instance = res_dict[key]
        for idx in range(1, 13):
            key = str(idx)
            one_cosine_sum = instance[key]['cosine_sum']
            one_token_sum = instance[key]['token_sum']
            layer_res[idx]['cosine_sum'] += one_cosine_sum
            layer_res[idx]['token_sum'] += one_token_sum

    res_list = []
    for idx in range(1, 13):
        one_cross_similarity = layer_res[idx]['cosine_sum'] / layer_res[idx]['token_sum']
        res_list.append(round(one_cross_similarity, 3))
    return res_list


if __name__ == '__main__':
    in_f = r'bert_result.json'
    bert_res_list = process_result(in_f)
    # print(bert_res_list)

    in_f = r'tacl_result.json'
    tacl_res_list = process_result(in_f)


    import matplotlib.pyplot as plt

    plt.rc('text',)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('Layer', fontsize=16)
    plt.ylabel('Self-similarity', fontsize=16)

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    y3 = bert_res_list
    y4 = tacl_res_list

    my_xticks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

    color_1, color_2 = 'orangered', 'royalblue'

    plt.xticks(x, my_xticks, fontsize=14)
    plt.plot(x, y4, marker='s', markerfacecolor='none', label='TaCL', linestyle='--', color=color_1)
    plt.plot(x, y3, marker='d', markerfacecolor='none', linestyle='--', label='BERT', color=color_2)

    plt.ylim(ymin=0.1, ymax=0.77)  # this line
    # plt.legend(loc='upper left', fontsize=12)
    plt.legend(fontsize=12, ncol=2, handleheight=2, labelspacing=0.005, columnspacing=0.5, loc='upper left')
    # plt.show()
    plt.savefig('self-similarity.png', format='png', dpi=500, bbox_inches='tight')