from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, asdict
from glob import glob
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
import json
import re
import random
import os
import logging
import openai
import numpy as np
import pickle
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import GPT2Tokenizer
import time

@dataclass
class ArxivArticle:
    text: str
    entry_id: str
    title: str
    sections: Dict[str, str]

    def __repr__(self):
        return f"ArxivArticle: {self.title}\n\n"

@dataclass
class Conversation:
    id: str
    context: List[Tuple[str, str]]

@dataclass
class ArxivContext:
    text: str
    entry_id: str
    context: str
    context_masked: bool
    masked_sents: List[str] = None

    def __repr__(self):
        return f"ArxivContext:\n --{self.context}\n\n"

class ArxivContextManager:
    """
        Loading arxiv articles, and process the article to sections.
        Obtaining the context of interest and do the partially masking (optional).

        Args:
            - mask_method: "Random", "self-info-sent" or "no". Randomly mask the context or mask the context based on the perplexity.
    """

    def __init__(
        self,
        path : str,
        random_mask_ratio = 0.2, 
        keep_leading_word = True,
        num_lead_words = 3,
        ppl_threshold = None,
        tokenizer = None,
        ppl_func = None,
    ):
        self.path = path
        self.load_articles(path)

        self.random_mask_ratio = random_mask_ratio
        self.keep_leading_word = keep_leading_word
        self.num_lead_words = num_lead_words
        self.ppl_threshold = ppl_threshold
        self.tokenizer = tokenizer
        self.ppl_func = ppl_func
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.max_token_len = 2000
        self.resulting_contexts = []

        self.mask_token = "<...some content omitted.>"
        # self.sent_tokenize_pattern = r"((?<!e\.g)(?<!i\.e)(?<!w\.r\.t)(?<=\.)\s)|(?<=\?\s)|(?<=!\s)"
        # self.sent_tokenize_pattern = r"(?<!e\.g)(?<!i\.e)(?<=\.\s)|(?<=\?\s)|(?<=!\s)"
        self.sent_tokenize_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    
    def load_articles(self, path: str) -> List[ArxivArticle]:
        self.articles = []
        for file_path in tqdm(glob(os.path.join(path, "*.json")), desc="Loading Arxiv articles"):
            with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                article = json.load(f)
                entry_id = article["entry_id"]
                title = article["title"]
                text = article["text"]

                # remove anything before introduction
                text = re.sub(r"^.*?(§)", r"\1", text, flags=re.DOTALL)

                # split article into sections
                sections = re.split(r"(?<!§\.)§\s", text)
                sections = [section for section in sections if section.strip()]
            
            if len(sections) == 0:
                continue
            
            self.articles.append(ArxivArticle(text=text, entry_id=entry_id, title=title, sections=sections))
        
        logging.info(f"Finish preprocessing Arxiv articles. Loaded {len(self.articles)} documents.")
    
    def beautify_context(self, context: str) -> str:
        context = context.replace("<cit.>", '').replace('<ref>', '')
        context = re.sub(r"\s+", " ", context)
        return context
    
    def varify_context_length(self, context: str) -> bool:
        if context is None:
            return False
        num_tokens = len(self.tokenizer(context)['input_ids'])
        if num_tokens > self.max_token_len:
            return False
        return True

    def generate_context(self, mask_method: str) -> List[ArxivContext]:
        assert mask_method in ["Random", "self-info-sent", "no"]

        for article_idx, article in tqdm(enumerate(self.articles), desc="Generating contexts"):
            intro = article.sections[0]
            intro = self.beautify_context(intro)
            if not self.varify_context_length(intro):
                continue

            if mask_method == "Random":
                context, masked_sents = self.random_mask_context(intro)
            elif mask_method == "self-info-sent":
                # sents = sent_tokenize(intro)
                sents = re.split(self.sent_tokenize_pattern, intro)
                sents = [sent for sent in sents if sent.strip()]
                try:
                    context, masked_sents = self.self_info_sent_mask(sents, title=article.title, sent_level=True)
                except Exception as e:
                    self._check_point()
                    sys.exit(f'Error in article {article_idx}: {e}')
            elif mask_method == "no":
                context = intro
                masked_sents = None

            self.resulting_contexts.append(
                ArxivContext(text=article.text, entry_id=article.entry_id, context=context, context_masked=(mask_method != "no"), masked_sents=masked_sents)
            )

        logging.info(f"Finish generating {len(resulting_contexts)} contexts.")
        self._check_point('Ending checkpointing.')
        return self.resulting_contexts
    
    def _check_point(self, message = '') -> bool:
        pickle_file = os.path.join(self.path, f"ArxivContextManager_{self.path}_{self.random_mask_ratio}.pkl")
        logging.info(f"saved to {pickle_file}. {message}")
        with open(pickle_file, "wb") as f:
            pickle.dump(self, f)
    
    def self_info_sent_mask(self, context: List[str], title: str = None, sent_level = False):
        sents_after_mask = []
        masked_sents = []

        if sent_level:
            sent_self_info = []
            for sent in context:
                tokens, self_info = get_self_information(sent)
                sent_self_info.append(np.mean(self_info))
            sents = context
        else:
            sents = sent_tokenize(context)
            context = ' '.join(sents)
            tokens, self_info = get_self_information(context)
            sent_self_info = self.calculate_sent_self_info(context, tokens, self_info)
                
        self.ppl_threshold = np.percentile(sent_self_info, self.random_mask_ratio * 100)

        if title is not None:
            with open(os.path.join(self.path, title+'_prob_token.tsv'), 'w', encoding='utf-8') as f:
                for token, info in zip(tokens, self_info):
                    f.write(f"{token}\t{info}\n")
            with open(os.path.join(self.path, title+'_prob_sent.tsv'), 'w', encoding='utf-8') as f:
                for sent, info in zip(sents, sent_self_info):
                    f.write(f"{sent}\n{info}\n\n")

        for sent, info in zip(sents, sent_self_info):
            if info < self.ppl_threshold:
                if self.keep_leading_word:
                    leading_few_words = " ".join(word_tokenize(sent)[:self.num_lead_words]) + " "
                else:
                    leading_few_words = ""
                masked_sents.append(sent)
                sents_after_mask.append(leading_few_words + self.mask_token)
            else:
                sents_after_mask.append(sent)
        masked_context = " ".join(sents_after_mask)
        
        return masked_context, masked_sents

    def calculate_sent_self_info(self, context, tokens, self_info) -> List[Tuple[str, float]]:
        sents = re.split(self.sent_tokenize_pattern, ''.join(tokens))
        sents = [sents[0][:-1]] + [' ' + sent[:-1] for sent in sents[1:-1]] + [' ' + sents[-1]]
        current_sent_idx = 0
        current_position = 0
        sent_self_info = [[] for _ in range(len(sents))]
        start = 0
        for idx, (token, info) in enumerate(zip(tokens, self_info)):
            current_position += len(token)

            if current_position >= len(sents[current_sent_idx]):
                end = idx
                # print(tokens[start:end+1], '^^', sents[current_sent_idx])
                # print(current_position, len(sents[current_sent_idx]), current_sent_idx)
                start = end + 1
                sent_self_info[current_sent_idx].append(info)
                current_position = current_position - len(sents[current_sent_idx])
                current_sent_idx += 1
            else:
                # print(current_position)
                if token == ' ':
                    continue
                sent_self_info[current_sent_idx].append(info)
        
        sent_self_info = [np.mean(info) for info in sent_self_info]
        return sent_self_info
    
    def random_mask_context(self, context: str,) -> str:
        sents = sent_tokenize(context)
        sents_after_mask = []
        masked_sents = []
        for sent in sents:
            if random.random() < self.random_mask_ratio:
                if self.keep_leading_word:
                    leading_few_words = " ".join(word_tokenize(sent)[:self.num_lead_words]) + " "
                else:
                    leading_few_words = ""
                masked_sents.append(sent)
                sents_after_mask.append(leading_few_words + self.mask_token)
            else:
                sents_after_mask.append(sent)
        masked_context = " ".join(sents_after_mask)
        return masked_context, masked_sents

class ConversationContextManager(ArxivContextManager):

    def __init__(
        self,
        path : str,
        random_mask_ratio = 0.2, 
        keep_leading_word = True,
        num_lead_words = 3,
        ppl_threshold = None,
        tokenizer = None,
        ppl_func = None,
    ):
        super().__init__(path, random_mask_ratio, keep_leading_word, num_lead_words, ppl_threshold, tokenizer, ppl_func)
    
    def load_articles(self, path):
        self.conversations = []
        count = 0
        with open(os.path.join(path, 'conversation_2k.json'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                count += 1
                if count > 10:
                    break
                conversation = json.loads(line)
                conversation = self._parse_conversation(conversation)
                self.conversations.append(conversation)
    
    def _parse_conversation(self, conversation):
        id = conversation['id']
        convs = []
        for sent in conversation['conversations']:
            role = sent['from']
            if role != 'human':
                bsobj = BeautifulSoup(sent['value'])
                for tag_name in ['p', 'br', 'div', 'li', 'h1', 'h2', 'h3',]:
                    for tag in bsobj.find_all(tag_name):
                        if tag.string is not None:
                            tag.string.replace_with(tag.string + ' ')
                value = bsobj.get_text()
            else:
                value = sent['value']
            convs.append((role, value))
        return Conversation(id, convs)
    
    def generate_context(self, mask_method):
        # self.conversations = self.conversations[:2]
        resulting_contexts = []
        for conversation in self.conversations:

            if mask_method == 'self-info-sent':
                masked_context, masked_sents = self.self_info_sent_mask(conversation)
            elif mask_method == 'Random':
                masked_context, masked_sents = self.random_mask_context(conversation.context)

            if not self.varify_context_length(masked_context):
                continue

            resulting_contexts.append(
                ArxivContext(text='', entry_id=conversation.id, context=masked_context, context_masked=True, masked_sents=masked_sents)
            )

        return resulting_contexts

    def random_mask_context(self, context):
        masked_context = ''
        masked_sents = []

        for sent in conversation.context:
            role = sent[0]
            value = sent[1]
            sents = re.split(self.sent_tokenize_pattern, value)
            masked_context += role + ": "
            for sent in sents:
                if random.random() < self.random_mask_ratio:
                    if self.keep_leading_word:
                        leading_few_words = " ".join(word_tokenize(sent)[:self.num_lead_words]) + " "
                    else:
                        leading_few_words = ""
                    masked_sents.append(sent)
                    masked_context += leading_few_words + self.mask_token
                else:
                    masked_context += sent
        return masked_context, masked_sents
        
    def self_info_sent_mask(self, conversation: Conversation, output = False):
        convs = []
        sent_self_info = []
        masked_context = ''
        masked_sents = []

        f = open(os.path.join(self.path, f'{conversation.id}.txt'), 'w', encoding='utf-8')

        for sent in conversation.context:
            role = sent[0]
            value = sent[1]
            sents = re.split(self.sent_tokenize_pattern, value)
            sents = [sent.strip() for sent in sents if sent.strip()]
            utterences = []
            for sent in sents:
                if not self.varify_context_length(sent):
                    return None, None
                tokens, self_info = get_self_information(sent)
                info = np.mean(self_info)
                utterences.append((sent, info))
                sent_self_info.append(info)
                f.write(f'{sent}\n{info}\n\n')
            convs.append((role, utterences))
        self.ppl_threshold = np.percentile(sent_self_info, self.random_mask_ratio * 100)
        for role, utterences in convs:
            masked_context += role + ': '
            for sent, info in utterences:
                if info < self.ppl_threshold:
                    if self.keep_leading_word:
                        leading_few_words = " ".join(word_tokenize(sent)[:self.num_lead_words]) + " "
                    else:
                        leading_few_words = ""
                    masked_sents.append(sent)
                    masked_context += leading_few_words + self.mask_token
                else:
                    masked_context += sent
        f.close()
        return masked_context, masked_sents

def get_self_information(text, num_retry = 3):
    # text = text[:1000]
    openai_key = os.environ["OPENAI_API_KEY"]

    for _ in range(num_retry):
        try:
            r = openai.Completion.create(
                model="curie",
                prompt=f"<|endoftext|>{text}",
                max_tokens=0,
                temperature=0,
                echo=True,
                logprobs=0,
            )
            break
        except Exception as e:
            print(e)
            time.sleep(1)

    result = r['choices'][0]
    tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

    assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

    self_info = [ -logprob for logprob in logprobs]
    # TODO: deal with the first delimiter
    return tokens, self_info


if __name__ == "__main__":
    context_path, = sys.argv[1:]
    mask_type = "self-info-sent"
    dataset_type = "conversations"
    dataset_manager = {
        'arxiv': ArxivContextManager,
        'conversations': ConversationContextManager,
    }
    context_manager = dataset_manager[dataset_type](context_path)
    contexts = context_manager.generate_context(mask_type)
    print(contexts)
    with open(f'{dataset_type}_contexts_{mask_type}.pkl', 'wb') as f:
        pickle.dump(contexts, f)
