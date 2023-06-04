from qa_manager import *
from utils import *
from context_manager import *
import sys
import pickle
import os
import logging
import copy

def save_as_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f'Saved to {file_path}')

def display_performance(context: ContextAndAnswer):
    assert context.metrics is not None, 'Not evaluted yet!'
    metric_result = '\n'.join([f'{k}: {v}' for k, v in context.metrics.items()])
    logging.info(f'Performance summary:\ntask type: {context.task_name}\ndataset type: {context.dataset_type}\nMask_ratio: {context.mask_ratio}\nMetrics: {metric_result}\n')
    print(f'\nPerformance summary:\ntask type: {context.task_name}\ndataset type: {context.dataset_type}\nMask_ratio: {context.mask_ratio}\nMetrics: {metric_result}\n')

def main():
    arxiv_path, news_path, conversation_path, save_to_path, num_articles, model_name = sys.argv[1:]
    logging.basicConfig(level=logging.INFO, filename=os.path.join(save_to_path, f'log_{model_name}.txt'), filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(f'arxiv_path: {arxiv_path}, news_path: {news_path}, conversation_path: {conversation_path}, save_to_path: {save_to_path}, num_articles: {num_articles}, model_name: {model_name}, task_name: {task_name}')
    num_articles = int(num_articles)
    
    # task_types = ['summarisation', 'masked-targeting-qa', 'qa']
    # mask_types = ['self-info-sentence', 'Ramdom', 'no']

    mask_types = ['no', 'self-info', 'Random']
    # mask_types = ['no', 'self-info', 'Random', 'no2']
    mask_levels = ['phrase', ]

    # task_types = ['continue_conversation']
    # task_types = [task_name]
    task_types = ['reconstruction', 'summarisation', 'qa', ]

    # dataset_types = [dataset_type]
    dataset_types = ['news', 'arxiv']
    # mask_ratios = [float(mask_ratio)]
    mask_ratios = [0.2, 0.35, 0.5, 0.65, 0.8]
    # models = ['gpt-3.5-turbo']
    models = [model_name]
    do_eval = False

    dataset_managers = {
        'arxiv': ArxivContextManager,
        'conversation': ConversationContextManager,
        'news': NewsContextManager,
    }

    task_managers = {
        'summarisation': Summarisation,
        'masked-targeting-qa': MaskedTargetingQA,
        'qa': QA,
        'reconstruction': OriginalContextReconsutrction,
        'continue_conversation': ContinueConversation,
    }

    data_paths = {
        'arxiv': arxiv_path,
        'news': news_path,
        'conversation': conversation_path
    }

    if do_eval:
        eavluator = Evaluator(metrics = ['bleu', 'meteor', 'rouge',])
        # eavluator = Evaluator(metrics = ['bleu', 'meteor', 'rouge', 'bertscore', 'bleurt'])
    managers = [ task_managers[task_type](task_type, model, save_to_path) for task_type in task_types for model in models]

    for dataset_type in dataset_types:
        data_path =  data_paths[dataset_type]

        # check if checkpoint exists
        checkpoint_path = os.path.join(data_path, f"{dataset_managers[dataset_type].__name__}_sent.pkl")
        if os.path.exists(checkpoint_path):
            context_manager = dataset_managers[dataset_type].from_checkpoint(checkpoint_path, phrase_mask_token = '', max_token_len = 1200, path = data_path)
        else:
            context_manager = dataset_managers[dataset_type](data_path, num_articles=100)
        for mask_ratio in mask_ratios:
            # first, we need to get all the contexts: origin contexts and masked contexts
            context_dict = {}
            for mask_type in mask_types:
                if mask_type == 'no':
                    contexts = context_manager.generate_context('no', num_articles=num_articles)
                    context_dict[mask_type] = contexts
                    continue
                for mask_level in mask_levels:
                    context_manager.mask_ratio = mask_ratio
                    contexts = context_manager.generate_context(mask_type, mask_level=mask_level, num_articles=num_articles)
                    context_dict[f'{mask_type}-{mask_level}'] = contexts
            ans = ContextAndAnswer(reference_context = 'no', contexts_dict=context_dict, dataset_type=dataset_type, mask_ratio=mask_ratio)
            
            for manager in managers:
                task_ans = copy.deepcopy(ans)
                task_ans.task_name = manager.task_name

                # second, we need to generate the answer for the given contexts.
                # we may also generate questions for some tasks, inside the setup() function
                manager.setup(task_ans)
                manager.get_answer()

                # third, we need to evaluate the performance of the task
                if do_eval:
                    manager.evaluate(eavluator)
                    display_performance(manager.ans)

                # save the answer and performance
                manager.save_as_pickle()


if __name__ == '__main__':
    main()