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
    arxiv_path, save_to_path, = sys.argv[1:]
    
    # task_types = ['summarisation', 'masked-targeting-qa', 'qa']
    # mask_types = ['self-info-sentence', 'Ramdom', 'no']

    mask_types = ['no', 'self-info-sent', ]
    mask_levels = ['phrase']
    task_types = ['summarisation', 'qa']
    dataset_types = ['arxiv']
    mask_ratios = [0.2, 0.4]
    models = ['gpt-3.5-turbo']

    dataset_managers = {
        'arxiv': ArxivContextManager,
        'conversations': ConversationContextManager,
    }

    task_managers = {
        'summarisation': Summarisation,
        'masked-targeting-qa': MaskedTargetingQA,
        'qa': QA,
    }

    data_path = {
        'arxiv': arxiv_path
    }

    managers = [ task_managers[task_type](task_type, model, save_to_path) for task_type in task_types for model in models]

    for dataset_type in dataset_types:
        data_path =  data_path[dataset_type]

        # check if checkpoint exists
        checkpoint_path = os.path.join(data_path, f"{ArxivContextManager.__name__}_sent.pkl")
        if os.path.exists(checkpoint_path):
            context_manager = ArxivContextManager.from_checkpoint(checkpoint_path, phrase_mask_token = '')
        else:
            context_manager = dataset_managers[dataset_type](data_path)
        for mask_ratio in mask_ratios:
            # first, we need to get all the contexts: origin contexts and masked contexts
            context_dict = {}
            for mask_type in mask_types:
                if mask_type == 'no':
                    contexts = context_manager.generate_context('no')
                    context_dict[mask_type] = contexts
                    continue
                for mask_level in mask_levels:
                    context_manager.mask_ratio = mask_ratio
                    contexts = context_manager.generate_context(mask_type, mask_level=mask_level)
                    context_dict[f'{mask_type}_{mask_level}'] = contexts
            ans = ContextAndAnswer(reference_context = 'no', contexts_dict=context_dict, dataset_type=dataset_type, mask_ratio=mask_ratio)
            
            for manager in managers:
                task_ans = copy.deepcopy(ans)
                task_ans.task_name = manager.task_name

                # second, we need to generate the answer for the given contexts.
                # we may also generate questions for some tasks, inside the setup() function
                manager.setup(task_ans)
                manager.get_answer()

                # third, we need to evaluate the performance of the task
                manager.evaluate()
                display_performance(manager.ans)

                # save the answer and performance
                manager.save_as_pickle()
    
    # for mask_ratio in mask_ratios:
    #     for dataset_type in dataset_types:
    #         for task_type in task_types:
    #             # first, we need to get all the contexts: origin contexts and masked contexts
    #             context_dict = {}
    #             for mask_type in mask_types:
    #                 context_manager = dataset_managers[dataset_type](path=arxiv_path, random_mask_ratio=mask_ratio)
    #                 contexts = context_manager.generate_context(mask_type)
    #                 context_dict[mask_type] = contexts
    #             # save_as_pickle(context_dict, f'contexts_{task_type}.pkl')
    #             ans = ContextAndAnswer(reference_context = 'no', contexts_dict=context_dict, task_name=task_type, dataset_type=dataset_type, mask_ratio=mask_ratio)

    #             # second, we need to generate the answer for the given contexts
    #             task_manager = task_managers[task_type](task_type, 'gpt-3.5-turbo')
    #             ans = task_manager.get_answer(ans)

    #             # third, we need to evaluate the performance of the task
    #             performance = task_manager.evaluate(ans)
    #             ans.metrics = performance

    #             # save the answer and performance
    #             save_path = os.path.join(save_to_path, f'answer_{task_type}_{dataset_type}_{mask_ratio}.pkl')
    #             display_performance(ans)
    #             save_as_pickle(ans, save_path)

if __name__ == '__main__':
    main()