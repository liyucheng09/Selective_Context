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
import pickle
import pandas as pd

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
    
    def __repr__(self):
        contexts = '\n'.join(self.contexts_dict.keys())
        return f"ContextAndAnswer:\n{contexts}"
            
class TaskManager:

    def __init__(self, task_name, model_type, save_path):
        self.task_name = task_name
        self.model_type = model_type
        self.save_path = save_path

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
    
    def setup(self, ans: ContextAndAnswer):
        self.ans = ans
        self.dataset_type = ans.dataset_type
        self.mask_ratio = ans.mask_ratio

        # see if checkpoint exists
        file_path = os.path.join(self.save_path, f'answer_{self.task_name}_{self.dataset_type}_{self.mask_ratio}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                pickled_ans = pickle.load(f)
            logging.info(f'Loaded from {file_path}')
            print(f'Loaded from {file_path}')

            # update saved answers and questions to the latest
            self.ans.answer_of_contexts = pickled_ans.answer_of_contexts
            self.ans.questions = pickled_ans.questions

    def save_as_pickle(self):
        file_path = os.path.join(self.save_path, f'answer_{self.task_name}_{self.dataset_type}_{self.mask_ratio}.pkl')
        # save the ContextAndAnswer object as pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.ans, f)
        logging.info(f'Saved to {file_path}')
        print(f'Saved to {file_path}')

class Summarisation(TaskManager):
    """
        This task is summarisation on the given context.
    """

    def __init__(self, task_name, model_type, save_path):
        super().__init__(task_name, model_type, save_path)
    
    def prompt_for_the_task(self, context: ArxivContext):
        prompt = f"Summarise the following content:\n\n----\n\n{context.context}"
        return prompt

    def get_answer(self):
        ans = self.ans
        answer_of_contexts = ans.answer_of_contexts
        for context_type, contexts in ans.contexts_dict.items():
            if context_type not in answer_of_contexts:
                answer_of_contexts[context_type] = []
            else:
                continue
            for context in contexts:
                prompt = self.prompt_for_the_task(context)
                summary = self._generate_answer(prompt)
                answer_of_contexts[context_type].append(summary)
        ans.answer_of_contexts = answer_of_contexts
        self.ans = ans
        logging.info(f"Summarisation task is done.")
        return ans
    
    def evaluate(self):
        # evaluate the summarisation task
        # try to use BLEU, ROUGE, METEOR, and BERTScore
        # bleu, bertscore, meteor, rouge all implemented by huggingface.metrics

        contexts = self.ans
        reference_context = contexts.reference_context
        reference_answer = contexts.answer_of_contexts[reference_context]
        performance = {}
        for context_type in contexts.answer_of_contexts:
            if context_type == reference_context:
                continue
            performance[context_type] = {}
            answer = contexts.answer_of_contexts[context_type]
            reference_answer_ = reference_answer[:len(answer)]
            for metric in ['bleu', 'meteor', 'rouge']:
                metric = evaluate.load(metric)
                if metric == 'bertscore':
                    score = metric.compute(predictions=answer, references=reference_answer_, lang='en')
                else:
                    score =  metric.compute(predictions=answer, references=reference_answer_)
                logging.info(f"Score for {metric} is {score}")
                performance[context_type].update(score)
        self.ans.metrics = performance
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
    def __init__(self, task_name, model_type, save_path):
        super().__init__(task_name, model_type, save_path)

        self.question_saved_path = os.path.join(self.save_path, task_name,)
        if not os.path.exists(self.question_saved_path):
            os.makedirs(self.question_saved_path)
    
    def generate_questions(self, ans: ContextAndAnswer):
        # see if the questions are already generated
        if ans.questions is not None:
            return ans

        # generate questions based on the origin context
        origin_contexts = ans.contexts_dict[ans.reference_context]
        all_questions = []
        reference_answers = []
        for cont in origin_contexts:
            question_save_file = os.path.join(self.question_saved_path, f"{ans.dataset_type}_{cont.id}.tsv")
            if os.path.exists(question_save_file):
                pass
            else:
                # generate questions
                prompt = self.prompt_for_the_task(cont, task = "question_generation")
                questions = self._generate_answer(prompt)

                # save the questions
                with open(question_save_file, "w") as f:
                    f.write(questions)

            # load the questions
            try:
                questions = pd.read_csv(question_save_file, sep = "\t")
                questions_ = questions['Question'].tolist()
                answers = questions['Answer'].tolist()
            except Exception as e:
                print(e)
                questions_ = None
                answers = None

            all_questions.append(questions_)
            reference_answers.append(answers)
        
        ans.questions = all_questions
        ans.answer_of_contexts = {ans.reference_context: reference_answers}
        return ans

    def prompt_for_the_task(self, context: ArxivContext, task : str, questions: List[str] = None):
        assert task in ["question_generation", "answer_generation"], "task should be either question_generation or answer_generation"

        # prepare the prompt for question generation
        if task == "question_generation":
            prompt = f"Please generate a tsv file containing a list of questions and answers based on the following given context. Remember, generate only the table and nothing else. The two column names should be Question and Answer.\n\n---\n{context.context}"
        elif task == "answer_generation":
            questions = "\n".join([f"{idx+1}. {qus}" for idx, qus in enumerate(questions)])
            prompt = f"Please generate a tsv file to answer the given questions based on the following given paragraph. Remember, generate only two columns for the question number and answers and nothing else. The column names should be Num and Answer.\n\n---Paragraph\n{context.context}\n\n---Questions\n{questions}"

        return prompt

    def get_answer(self):
        ans = self.ans
        answer_of_contexts = ans.answer_of_contexts
        for context_type, contexts in ans.contexts_dict.items():
            if context_type not in answer_of_contexts:
                answer_of_contexts[context_type] = []
            else:
                continue
            for index, context in enumerate(contexts):
                if ans.questions[index] is None:
                    answer_of_contexts[context_type].append(None)
                    continue
                answer_save_file = os.path.join(self.question_saved_path, f"{ans.dataset_type}_{cont.id}_{context_type}_{self.mask_ratio}.tsv")

                if os.path.exists(answer_save_file):
                    pass
                else:
                    # generate questions
                    prompt = self.prompt_for_the_task(context, task = "answer_generation", questions = ans.questions[index])
                    answers = self._generate_answer(prompt)

                    # save the questions
                    with open(answer_save_file, "w") as f:
                        f.write(answers)
                
                # load the answers
                with open(answer_save_file, "r") as f:
                    answers = pd.read_csv(f, sep = "\t")
                answers = answers['Answer'].tolist()
                try:
                    assert len(answers) == len(ans.questions[index]), f"the number of answers {len(answers)} should be equal to the number of questions {len(ans.questions[index])}"
                except AssertionError as e:
                    print(e)
                    answers = None
                answer_of_contexts[context_type].append(answers)
        ans.answer_of_contexts = answer_of_contexts
        self.ans = ans
        logging.info(f"Summarisation task is done.")
        return ans
    
    def evaluate(self):
        # evaluate the summarisation task
        # try to use BLEU, ROUGE, METEOR, and BERTScore
        # bleu, bertscore, meteor, rouge all implemented by huggingface.metrics

        contexts = self.ans
        reference_context = contexts.reference_context

        reference_answer = contexts.answer_of_contexts[reference_context]

        performance = {}
        for context_type in contexts.answer_of_contexts:
            if context_type == reference_context:
                continue
            performance[context_type] = {}

            # the answers here is a list of list of answers, should be flatten into a 1-D list
            # also remember to remove the None answers
            answers = contexts.answer_of_contexts[context_type]
            flatten_answer = []
            flatten_reference_answer = []
            for p_a, r_a in zip(answers, reference_answer):
                if p_a is None or r_a is None:
                    continue
                flatten_answer.extend(p_a)
                flatten_reference_answer.extend(r_a)

            for metric in ['bleu', 'meteor', 'rouge', 'bertscore']:
                metric = evaluate.load(metric)
                if metric == 'bertscore':
                    score = metric.compute(predictions=flatten_answer, references=flatten_reference_answer, lang='en')
                else:
                    score = metric.compute(predictions=flatten_answer, references=flatten_reference_answer)
                logging.info(f"Score for {metric} is {score}")
                performance[context_type].update(score)
        
        self.ans.metrics = performance
        return performance
    
    def setup(self, ans):
        super().setup(ans)
        self.ans = self.generate_questions(ans)