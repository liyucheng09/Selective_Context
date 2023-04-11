# Use OpenAI's GPT-3.5-turbo to generate questions and answer from a given document
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, asdict
from context_manager import ArxivContext
import sys
import json
import logging
import evaluate
import os
import openai
import time

@dataclass
class ContextAndAnswer:
    """
        This class stores contexts and its list of masked_contexts.
        It also take care of questions and reference answer based on original context.
        As well as answers based on masked_context.

        should add a function in this class to evaluate/compare answers against reference answer.
    """
    reference_context: str
    contexts_dict: Dict[str, List[ArxivContext]]
    mask_ratio: float
    reduced_ratio: Dict[str, float] = None
    task_name: str = None
    questions: Union[List[str], List[List[str]], Dict[str, List[str]], Dict[str, List[List[str]]]] = None
    answer_of_contexts: Dict[str, List[str]] = None
    dataset_type : str = None
    metrics: Dict[str, float] = None

    def __post_init__(self):
        reference_contexts = self.contexts_dict[self.reference_context]
        self.reduced_ratio = {}
        for context_type in self.contexts_dict:
            if context_type == self.reference_context:
                continue
            self.reduced_ratio[context_type] = []
            for ref, cont in zip(reference_contexts, self.contexts_dict[context_type]):
                sub_len = len(ref.context) - len(cont.context)
                if sub_len < 0:
                    sub_len = 0
                self.reduced_ratio[context_type].append(sub_len / len(ref.context))
            
class TaskManager:

    def __init__(self, task_name, model_type):
        self.task_name = task_name
        self.model_type = model_type

        self._prepare_model()
    
    def _prepare_model(self):
        # prepare model and generate function
        # should support GPT-3.5-turbo, llama-7B,13B,30B, and Flan family?
        if self.model_type == "gpt-3.5-turbo":
            self._generate_answer = self._gpt_3_5_turbo_generate
        elif self.model_type == "llama-7B":
            pass
    
    def _gpt_3_5_turbo_generate(self, prompt, num_retry = 5):
        # generate answer by gpt-3.5-turbo
        openai_key = os.environ.get("OPENAI_KEY")
        for _ in range(num_retry):
            try:
                r = openai.ChatCompletion.create(
                    model = 'gpt-3.5-turbo',
                    messages = [
                        {"role": "user", "content": prompt},
                    ],
                )
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        
        return r.choices[0]['message']['content']
    
    def prompt_for_the_task(self):
        raise NotImplementedError
    
    def _generate_answer(self, prompt):
        raise NotImplementedError
    
    def generate_by_openai(self, prompt):
        # generate answer by openai
        pass
    
    def get_answer(self, contexts: List[ContextAndAnswer]):
        raise NotImplementedError

class Summarisation(TaskManager):
    """
        This task is summarisation on the given context.
    """

    def __init__(self, task_name, model_type):
        super().__init__(task_name, model_type)
    
    def prompt_for_the_task(self, context: ArxivContext):
        prompt = f"Summarise the following content:\n\n----\n\n{context.context}"
        return prompt

    def get_answer(self, ans: ContextAndAnswer):
        answer_of_contexts = {}
        for context_type, contexts in ans.contexts_dict.items():
            for context in contexts:
                prompt = self.prompt_for_the_task(context)
                summary = self._generate_answer(prompt)
                if context_type not in answer_of_contexts:
                    answer_of_contexts[context_type] = []
                answer_of_contexts[context_type].append(summary)
        ans.answer_of_contexts = answer_of_contexts
        logging.info(f"Summarisation task is done.")
        return ans
    
    @staticmethod
    def evaluate(contexts: ContextAndAnswer):
        # evaluate the summarisation task
        # try to use BLEU, ROUGE, METEOR, and BERTScore
        # bleu, bertscore, meteor, rouge all implemented by huggingface.metrics

        reference_context = contexts.reference_context
        reference_answer = contexts.answer_of_contexts[reference_context]
        performance = {}
        for context_type in contexts.answer_of_contexts:
            if context_type == reference_context:
                continue
            performance[context_type] = {}
            answer = contexts.answer_of_contexts[context_type]
            for metric in ['bleu', 'meteor', 'rouge']:
                metric = evaluate.load(metric)
                if metric == 'bertscore':
                    score = metric.compute(predictions=answer, references=reference_answer, lang='en')
                else:
                    score =  metric.compute(predictions=answer, references=reference_answer)
                logging.info(f"Score for {metric} is {score}")
                performance[context_type].update(score)
        return performance
        

class MaskedTargetingQA(TaskManager):
    """
        This task is questions targeting on the masked sentences.
    """
    def __init__(self, task_name, model_type):
        super().__init__(task_name, model_type)
    
    def prompt_for_the_task(self):
        # prepare the prompt for the masked targeting QA task
        pass

    def get_answer(self, prompt):
        # generate answer for the given prompt
        pass

class QA(TaskManager):
    """
        This task conducts general QA on the given context.

        It first generate questions based on the given context.
        Then it generate answers for the questions given list of contexts.

        Note that the questions generated are shared across all contexts.
    """
    def __init__(self, task_name, model_type):
        super().__init__(task_name, model_type)
    
    def prompt_for_the_task(self, context: ArxivContext, questions: List[str]):
        # prepare the prompt for the QA task
        pass

    def get_answer(self, contexts: List[ContextAndAnswer]):
        for context in contexts:
            answer_of_contexts = {}
            if context.questions is None:
                # generate questions
                questions = []
            else:
                questions = context.questions
                
            for context_type, context in context.contexts_dict.items():
                prompt = self.prompt_for_the_task(context)
                answer = self._generate_answer(prompt)
                answer_of_contexts[context_type] = answer
            context.answer_of_contexts = answer_of_contexts