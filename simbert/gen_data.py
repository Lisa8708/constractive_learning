#-*- coding: utf-8 -*-
# 原数据：{"sentence1": "x1", "sentence2": "x2", "gold_label": "entailment"}
# 生成训练simbert的数据，格式为{'text': 'x', 'synonyms': ['x1', 'x2', 'x3']}
import json

def gen_data(input_file, output_file):
    with open(input_file, 'r') as fi, open(output_file, 'w', encoding = 'utf-8') as fo:
        lines = fi.readlines()
        middle_results = {}
        for line in lines:
            line = json.loads(line)
            s1 = line['sentence1'] # .encode('utf-8').decode('unicode-escape')
            s2 = line['sentence2']
            type = line['gold_label']
            if type == 'entailment':
                middle_results[s1] = middle_results.get(s1, []) + [s2]
        results = []
        for text, values in middle_results.items():
            if len(values) > 0:
                res = {}
                res['text'] = text
                res['synonyms'] = values
                #print(res)
                results.append(res)
        json.dump(results, fo)
if __name__ == '__main__':
    gen_data('../../data/cnsd-snli/cnsd_snli_v1.0.train.jsonl', '../../data/cnsd-snli/train.json')
    gen_data('../../data/cnsd-snli/cnsd_snli_v1.0.dev.jsonl', '../../data/cnsd-snli/dev.json')
    gen_data('../../data/cnsd-snli/cnsd_snli_v1.0.test.jsonl', '../../data/cnsd-snli/test.json')