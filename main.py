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
    logging.info(f'Performance summary:\ntask type: {context.task_name}\ndataset type: {context.dataset_type}\nMetrics: {context.metrics}\n')
    print(f'\nPerformance summary:\ntask type: {context.task_name}\ndataset type: {context.dataset_type}\nMask_ratio: {context.mask_ratio}\nMetrics: {context.metrics}\n')

def main():
    arxiv_path, save_to_path, = sys.argv[1:]
    
    # task_types = ['summarisation', 'masked-targeting-qa', 'qa']
    # mask_types = ['self-info-sentence', 'Ramdom', 'no']

    mask_types = ['no', 'self-info-sent', ]
    task_types = ['summarisation']
    dataset_types = ['arxiv']
    mask_ratios = [0.2, 0.4, 0.6]

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

    for dataset_type in dataset_types:
        context_manager = dataset_managers[dataset_type](data_path[dataset_type])
        for mask_ratio in mask_ratios:
            # first, we need to get all the contexts: origin contexts and masked contexts
            context_dict = {}
            for mask_type in mask_types:
                context_manager.mask_ratio = mask_ratio
                contexts = context_manager.generate_context(mask_type)
                print(contexts)
                context_dict[mask_type] = contexts
            ans = ContextAndAnswer(reference_context = 'no', contexts_dict=context_dict, dataset_type=dataset_type, mask_ratio=mask_ratio)
            
            for task_type in task_types:

                # second, we need to generate the answer for the given contexts
                task_manager = task_managers[task_type](task_type, 'gpt-3.5-turbo')
                task_ans = copy.deepcopy(ans)
                task_ans.task_name = task_type
                ans = task_manager.get_answer(copy.deepcopy(ans))

                # third, we need to evaluate the performance of the task
                performance = task_manager.evaluate(ans)
                ans.metrics = performance

                # save the answer and performance
                save_path = os.path.join(save_to_path, f'answer_{task_type}_{dataset_type}_{mask_ratio}.pkl')
                display_performance(ans)
                save_as_pickle(ans, save_path)
    
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