from context_manager import *
import numpy as np
from typing import List, Dict
from qa_manager import *
import os
import pandas as pd
import math
import sys

def checkout_prob(text, file_path = 'prob.tsv'):
    tokens, self_info = get_self_information(text)
    with open(file_path, 'w') as f:
        for token, info in zip(tokens, self_info):
            print(token, info)
            f.write(token + '\t' + str(info) + '\n')
    print('Finished writing to file: ', file_path)

def read_lexical_units(article: ArxivArticle, mask_level = 'phrase'):
    if mask_level == 'sent':
        lexical_units = article.units[0]
        assert lexical_units.unit_type == 'sent'
    elif mask_level == 'phrase':
        lexical_units = article.units[1]
        assert lexical_units.unit_type == 'phrase'
    elif mask_level == 'token':
        lexical_units = article.units[2]
        assert lexical_units.unit_type == 'token'

    tokens = lexical_units.text[:50] + lexical_units.text[360:421]
    self_info = lexical_units.self_info[:50] + lexical_units.self_info[360:421]
    self_info = [x**1.2 for x in self_info]

    max_score = max(self_info)
    min_score = min(self_info)

    mid = np.percentile(self_info, 50)

    lines = []
    highlighted = []
    buffer = []
    for token, score in zip(tokens, self_info):
        normalized_score = ((score - min_score) / (max_score - min_score)) * 100
        line = f"\\colorize{{{normalized_score}}}{{{token}}}"
        if score > mid:
            if len(buffer) > 0:
                str_ = '\n'.join(buffer)
                lines.append(f"\\underline{{{str_}}}")
                buffer = []
            highlighted.append(line)
            lines.append(line)
        else:
            # token = f"\\sdelete{{{token}}}"
            # line = f"\\colorize{{{normalized_score}}}{{{token}}}"
            buffer.append(line)

    return '\n'.join(lines) + '\n\n\n' + '\n'.join(highlighted)

def datasets_statistics(manager: ArxivContextManager, tokenizer):
    def num_tokens(text):
        return len(tokenizer(text)['input_ids'])

    articles = manager.articles
    num_sents = [len(article.units[0].text) for article in articles]
    num_phrases = [len(article.units[1].text) for article in articles]
    num_tokens = [len(article.units[2].text) for article in articles]

    print('Number of articles: ', len(articles))
    print('Average number of sentences: ', np.mean(num_sents))
    print('Average number of phrases: ', np.mean(num_phrases))
    print('Average number of tokens: ', np.mean(num_tokens))

def merge_answer(tasks: List[str], data_sources: List[str], mask_ratios: List[str], context_type,):
    # read all answers objects, and merge them based on the given demands
    
    all_ans_paths = []
    for task in tasks:
        for data_source in data_sources:
            for mask_ratio in mask_ratios:
                if data_source == 'news':
                    ans_path = f'/vol/research/lyc/llm_memorize/answer_{task}_{data_source}_{mask_ratio}.pkl'
                elif data_source == 'arxiv':
                    ans_path = f'/vol/research/lyc/llm_memorize/{"arxiv_buggy/" if context_type == "Random-phrase" else ""}answer_{task}_{data_source}_{mask_ratio}.pkl'
                all_ans_paths.append(ans_path)

    answer_of_contexts = {}
    for ans_path in all_ans_paths:
        for context, answer in answer_of_contexts.items():
            print(context, len(answer))

        print(ans_path, '-------')

        with open(ans_path, 'rb') as f:
            ans = pickle.load(f)
        
        if ans.reference_context not in answer_of_contexts:
            answer_of_contexts[ans.reference_context] = []
        if context_type not in answer_of_contexts:
            answer_of_contexts[context_type] = []

        if ans.task_name == 'qa':
            refs = ans.answer_of_contexts[ans.reference_context]
            answer = ans.answer_of_contexts[context_type]
            for ref, ans_ in zip(refs, answer):
                if ref is None or ans_ is None:
                    continue
                assert len(ref) == len(ans_)
                for r, a in zip(ref, ans_):
                    if isinstance(r, float) or isinstance(a, float):
                        continue
                    answer_of_contexts[context_type].append(a)
                    answer_of_contexts[ans.reference_context].append(r)
            continue

        answer = ans.answer_of_contexts[context_type]
        reference = ans.answer_of_contexts[ans.reference_context]
        slice_ = min(len(reference), len(answer))

        answer_of_contexts[context_type].extend(answer[:slice_])
        answer_of_contexts[ans.reference_context].extend(reference[:slice_])

    return answer_of_contexts

def prepare_results_1(context_type):

    save_path = '/vol/research/lyc/llm_memorize/results'
    evaluator = Evaluator(metrics=['bleu', 'meteor', 'rouge', 'bertscore']) #  'bertscore'

    # all_mask_ratios = [0.8]
    # all_mask_ratios = [0.2, 0.35, 0.5, 0.65, 0.8]
    all_mask_ratios = [0.35, ]
    
    # results 1
    answers = {}
    # for mask_ratio in all_mask_ratios:
    ans = merge_answer(tasks = ['qa', 'reconstruction', 'summarisation'], data_sources = ['news', 'arxiv'], mask_ratios = all_mask_ratios, context_type = context_type)
    sip_ans = ans[context_type]
    references = ans['no']
    answers['all'] = evaluator.evaluate(sip_ans, references)

        # random_ans = merge_answer(tasks = ['qa', 'reconstruction', 'summarisation'], data_sources = ['news', 'arxiv'], mask_ratios = [mask_ratio], context_type = 'Random-phrase')
        # random_ans_ = random_ans['Random-phrase']
        # references = random_ans['no']
        # random_phrase[mask_ratio] = evaluator.evaluate(random_ans_, references)
    
    sip_df = pd.DataFrame.from_dict(answers, orient='index', columns=['bleu', 'meteor', 'rouge1', 'bertscore_precision', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore_recall', 'bertscore_f1'])

    sip_df.to_csv(os.path.join(save_path, f'{context_type}_0.5_all.csv'))

def prepare_results_2():

    save_path = '/vol/research/lyc/llm_memorize/results'
    evaluator = Evaluator(metrics=['bleu', 'meteor', 'rouge', 'bertscore']) #  'bertscore'

    # all_mask_ratios = [0.8]
    all_mask_ratios = [0.2, 0.35, 0.5, 0.65, 0.8]
    
    # results 2
    self_info_phrase = {
        'qa': {},
        'summarisation': {},
        'reconstruction': {}
    }
    for task in ['qa', 'reconstruction', 'summarisation']:
        for mask_ratio in all_mask_ratios:
            ans = merge_answer(tasks = [task], data_sources = ['news', 'arxiv'], mask_ratios = [mask_ratio], context_type = 'self-info-phrase')
            sip_ans = ans['self-info-phrase']
            references = ans['no']
            self_info_phrase[task][mask_ratio] = evaluator.evaluate(sip_ans, references)

        sip_df = pd.DataFrame.from_dict(self_info_phrase[task], orient='index', columns=['bleu', 'meteor', 'rouge1', 'bertscore_precision', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore_recall', 'bertscore_f1'])
        sip_df.to_csv(os.path.join(save_path, f'{task}_all.csv'))

def prepare_results_3(context_type):

    save_path = '/vol/research/lyc/llm_memorize/results'
    evaluator = Evaluator(metrics=['bleu', 'meteor', 'rouge', 'bertscore']) #  'bertscore'

    # all_mask_ratios = [0.8]
    # all_mask_ratios = [0.2, 0.35, 0.5, 0.65, 0.8]
    all_mask_ratios = [0.35,]
    all_tasks = ['qa', 'reconstruction', 'summarisation'] if context_type != 'no2-phrase' else ['qa', 'summarisation']
    
    # results 1
    answers = {}
    # for mask_ratio in all_mask_ratios:
    for task in all_tasks:
        ans = merge_answer(tasks = [task], data_sources = ['news', 'arxiv'], mask_ratios = all_mask_ratios, context_type = context_type)
        sip_ans = ans[context_type]
        references = ans['no']
        answers[task] = evaluator.evaluate(sip_ans, references)

        # random_ans = merge_answer(tasks = ['qa', 'reconstruction', 'summarisation'], data_sources = ['news', 'arxiv'], mask_ratios = [mask_ratio], context_type = 'Random-phrase')
        # random_ans_ = random_ans['Random-phrase']
        # references = random_ans['no']
        # random_phrase[mask_ratio] = evaluator.evaluate(random_ans_, references)
    
    sip_df = pd.DataFrame.from_dict(answers, orient='index', columns=['bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'])
    sip_df.to_csv(os.path.join(save_path, f'{context_type}_task_wise.csv'))

def visualisation(dfs: Dict[str, pd.DataFrame]):
    import matplotlib.pyplot as plt

    random_df = pd.read_csv('results/random_phrase_all.csv', index_col=0)
    sip_df = pd.read_csv('results/self_info_phrase_all.csv', index_col=0)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), dpi=120)
    sip_df.plot(y='bleu', ax=axes[0], marker='^', label = 'SC')
    random_df.plot(y='bleu', ax=axes[0], marker='+', label = 'Random')
    axes[0].set_title('BLEU')
    axes[0].set_xticks([0.2, 0.35, 0.5, 0.65, 0.8])
    # axes[0].set_xlabel('Filtered Ratio')

    sip_df.plot(y='rouge1', ax=axes[1], marker='^', label = 'SC')
    random_df.plot(y='rouge1', ax=axes[1], marker='+', label = 'Random')
    axes[1].set_title('ROUGE1')
    axes[1].set_xticks([0.2, 0.35, 0.5, 0.65, 0.8])
    # axes[1].set_ylim(0.2, 0.7)
    # axes[1].set_xlabel('Filtered Ratio')

    sip_df.plot(y='bertscore_f1', ax=axes[2], marker='^', label = 'SC')
    random_df.plot(y='bertscore_f1', ax=axes[2], marker='+', label = 'Random')
    axes[2].set_title('BERTScore')
    axes[2].set_xticks([0.2, 0.35, 0.5, 0.65, 0.8])
    # axes[-1].set_xlabel('Filtered Ratio')
    # axes[2].set_ylim(0.5, 1)
    fig.text(0.55, -0.03, 'Filter ratio', ha='center', fontsize=13)
    plt.tight_layout()

def visualisation_2():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), dpi=120)
    no_df.plot(y='bleu', ax=axes[0], marker='2', label = 'Original')
    sip_df.plot(y='bleu', ax=axes[0], marker='^', label = 'Selective Context', color = 'salmon')
    random_df.plot(y='bleu', ax=axes[0], marker='+', label = 'Random', color = 'grey')
    axes[0].set_title('BLEU')
    axes[0].set_xticks([0.2, 0.35, 0.5, 0.65, 0.8])
    axes[0].set_ylim(0.02, 0.47)

    no_df.plot(y='rouge1', ax=axes[1], marker='2', label = 'Original')
    sip_df.plot(y='rouge1', ax=axes[1], marker='^', label = 'Selective Context', color = 'salmon')
    random_df.plot(y='rouge1', ax=axes[1], marker='+', label = 'Random', color = 'grey')
    axes[1].set_title('ROUGE1')
    axes[1].set_xticks([0.2, 0.35, 0.5, 0.65, 0.8])
    axes[1].set_ylim(0.3, 0.72)
    # axes[1].set_ylim(0.2, 0.7)
    # axes[1].set_xlabel('Filtered Ratio')

    no_df.plot(y='bertscore_f1', ax=axes[2], marker='2', label = 'Original')
    sip_df.plot(y='bertscore_f1', ax=axes[2], marker='^', label = 'Selective Context', color = 'salmon')
    random_df.plot(y='bertscore_f1', ax=axes[2], marker='+', label = 'Random', color = 'grey')
    axes[2].set_title('BERTScore')
    axes[2].set_xticks([0.2, 0.35, 0.5, 0.65, 0.8])
    axes[2].set_ylim(0.87, 0.945)
    # axes[-1].set_xlabel('Filtered Ratio')
    # axes[2].set_ylim(0.5, 1)
    fig.text(0.52, -0.03, 'Context reduction ratio', ha='center', fontsize=14)
    plt.tight_layout()


def ramdom_baseline():
    from glob import glob
    import pickle

    for random_ans_file in glob('/vol/research/lyc/llm_memorize/arxiv_buggy/*.pkl'):
        base = os.path.basename(random_ans_file)

        if not os.path.exists(os.path.join('/vol/research/lyc/llm_memorize/', base)):
            continue

        with open(random_ans_file, 'rb') as f:
            ans = pickle.load(f)
        
        random_ans = ans.answer_of_contexts['Random-phrase']

        with open(os.path.join('/vol/research/lyc/llm_memorize/', base), 'rb') as f:
            ans = pickle.load(f)
        
        ans.answer_of_contexts['Random-phrase'] = random_ans

        with open(os.path.join('/vol/research/lyc/llm_memorize/', base), 'wb') as f:
            pickle.dump(ans, f)
        
        print(f"Done {base}")

if __name__ == '__main__':
    context_type, = sys.argv[1:]
    
    # prepare_results_1(context_type)
    # ramdom_baseline()
    prepare_results_3(context_type) # 'no2-phrase', 'self-info-phrase', 'Random-phrase'
    # prepare_results_2()