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
import spacy


@dataclass
class LexicalUnits:
    unit_type: str
    text: List[str]
    self_info: List[float] = None

@dataclass
class ArxivArticle:
    text: str
    entry_id: str
    title: str
    sections: Dict[str, str]
    context_type: str = None
    units: List[LexicalUnits] = None

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
        mask_ratio = 0.2, 
        keep_leading_word = True,
        num_lead_words = 3,
        ppl_threshold = None,
        tokenizer = None,
        compute_self_info = True,
    ):
        self.path = path
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.nlp.add_pipe('merge_noun_chunks')
        self.load_articles(path)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if tokenizer is None else tokenizer

        self.keep_leading_word = keep_leading_word
        self.num_lead_words = num_lead_words
        self.ppl_threshold = ppl_threshold
        self.max_token_len = 1800
        self.sent_level_self_info = True
        self.mask_ratio = mask_ratio

        self.mask_token = "<...some content omitted.>"
        self.phrase_mask_token = "..."

        # self.sent_tokenize_pattern = r"((?<!e\.g)(?<!i\.e)(?<!w\.r\.t)(?<=\.)\s)|(?<=\?\s)|(?<=!\s)"
        # self.sent_tokenize_pattern = r"(?<!e\.g)(?<!i\.e)(?<=\.\s)|(?<=\?\s)|(?<=!\s)"
        self.sent_tokenize_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        self._prepare_self_info()
    
    def _prepare_self_info(self):
        logging.info("Preparing self information...")
        articles = []
        for article_idx, article in tqdm(enumerate(self.articles), desc="Preparing self information"):
            if article.units is not None:
                # means the LexicalUnits has been calculated
                articles.append(article)
                continue
            intro = self.beautify_context(article.sections[0])

            if not self.varify_context_length(intro):
                continue
            sents = re.split(self.sent_tokenize_pattern, intro)
            sents = [sent for sent in sents if sent.strip()]

            if len(sents) == 0:
                continue

            # try:
            article.units = self._lexical_unit(sents)
            # except Exception as e:
            #     logging.error(f"Error in article {article_idx}: {e}")
            #     articles = articles + self.articles[article_idx:]
            #     self.articles = articles
            #     self._check_point('Error _preparing_self_info {article_idx}: {e}')
            #     exit(1)
            
            articles.append(article)

        self.articles = articles
        self._check_point('Finished _preparing_self_info')
    
    def _lexical_unit(self, sents):

        if self.sent_level_self_info:
            sent_self_info = []
            all_noun_phrases = []
            all_noun_phrases_info = []
            all_tokens = []
            all_token_self_info = []

            for sent in sents:
                tokens, self_info = get_self_information(sent)
                sent_self_info.append(np.mean(self_info))

                all_tokens.extend(tokens)
                all_token_self_info.extend(self_info)

                noun_phrases, noun_phrases_info = self._calculate_lexical_unit(tokens, self_info)
                all_noun_phrases.extend(noun_phrases)
                all_noun_phrases_info.extend(noun_phrases_info)
            
            return [
                LexicalUnits('sent', text=sents, self_info=sent_self_info),
                LexicalUnits('phrase', text=all_noun_phrases, self_info=all_noun_phrases_info),
                LexicalUnits('token', text=all_tokens, self_info=all_token_self_info)
            ]
            
        else:
            sents = sent_tokenize(context)
            context = ' '.join(sents)
            tokens, self_info = get_self_information(context)
            sent_lexical_units, phrase_lexical_units =  self._calculate_lexical_unit(tokens, self_info)

            return [
                sent_lexical_units,
                phrase_lexical_units,
                LexicalUnits('token', text=tokens, self_info=self_info)
            ]

    def _calculate_lexical_unit(self, tokens, self_info):
        def _unit_info(tokens, self_info, units):
            current_unit_idx = 0
            current_position = 0
            unit_self_info = [[] for _ in range(len(units))]

            for idx, (token, info) in enumerate(zip(tokens, self_info)):
                current_position += len(token)
                if current_position >= len(units[current_unit_idx]):
                    unit_self_info[current_unit_idx].append(info)
                    current_position = current_position - len(units[current_unit_idx])
                    current_unit_idx += 1
                else:
                    if token == " ":
                        continue
                    unit_self_info[current_unit_idx].append(info)
            
            unit_self_info = [np.mean(info) for info in unit_self_info]
            return unit_self_info

        if self.sent_level_self_info:
            # in this case, the self_info is for each sentence
            # we only need to calculate the self_info for each phrase

            sent = ''.join(tokens)
            # noun_phrases = [chunk.text for chunk in self.nlp(sent).noun_chunks]
            noun_phrases = [chunk.text+chunk.whitespace_ if chunk.whitespace_ else chunk.text for chunk in self.nlp(sent)]
            noun_phrases[-1] = noun_phrases[-1] + ' '
            noun_phrases_info = _unit_info(tokens, self_info, noun_phrases)

            return noun_phrases, noun_phrases_info
        
        else:
            # in this case, the self_info is for the entire context
            # we need to first calculate the self_info for each sentence
            # then calculate the self_info for each phrase

            sents = re.split(self.sent_tokenize_pattern, ''.join(tokens))
            sents = [sents[0][:-1]] + [' ' + sent[:-1] for sent in sents[1:-1]] + [' ' + sents[-1]]

            sent_self_info = _unit_info(tokens, self_info, units)

            # now we got sentence self_info, we need to calculate the self_info for each phrase
            all_noun_phrases = []
            all_noun_phrases_info = []
            for sent, sent_info in zip(sents, sent_self_info):
                noun_phrases = [chunk.text+chunk.whitespace_ if chunk.whitespace_ else chunk.text for chunk in self.nlp(sent)]
                noun_phrases[-1] = noun_phrases[-1] + ' '
                noun_phrases_info = _unit_info(tokens, self_info, noun_phrases)
                all_noun_phrases.extend(noun_phrases)
                all_noun_phrases_info.extend(noun_phrases_info)

            return LexicalUnits('sent', text = sents, self_info = sent_self_info), LexicalUnits('phrase', text = all_noun_phrases, self_info = all_noun_phrases_info)
    
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
                sections = [self.beautify_context(section) for section in sections if section.strip()]
            
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

    def generate_context(self, mask_method: str, mask_level: str = 'sent') -> List[ArxivContext]:
        assert mask_method in ["Random", "self-info-sent", "no"]
        resulting_contexts = []

        for article in tqdm(self.articles, desc="Generating contexts"):
            if mask_level == 'sent':
                lexical_units = article.units[0]
                assert lexical_units.unit_type == 'sent'
            elif mask_level == 'phrase':
                lexical_units = article.units[1]
                assert lexical_units.unit_type == 'phrase'
            elif mask_level == 'token':
                lexical_units = article.units[2]
                assert lexical_units.unit_type == 'token'

            if mask_method == "Random":
                context, masked_sents = self.random_mask_context(lexical_units.text, mask_level)
            elif mask_method == "self-info-sent":
                context, masked_sents = self.self_info_mask(lexical_units.text, lexical_units.self_info, mask_level)
            elif mask_method == "no":
                context = article.sections[0]
                masked_sents = None

            resulting_contexts.append(
                ArxivContext(text=article.text, entry_id=article.entry_id, context=context, context_masked=(mask_method != "no"), masked_sents=masked_sents)
            )

        logging.info(f"Finish generating {len(resulting_contexts)} contexts.")
        return resulting_contexts
    
    def _check_point(self, message = '') -> bool:
        pickle_file = os.path.join(self.path, f"{self.__class__.__name__}_{'sent' if self.sent_level_self_info else 'paragraph'}.pkl")
        logging.info(f"saved to {pickle_file}. {message}")
        with open(pickle_file, "wb") as f:
            pickle.dump(self, f)
    
    def self_info_mask(self, sents: List[str], self_info: List[float], mask_level):
        sents_after_mask = []
        masked_sents = []
                
        self.ppl_threshold = np.nanpercentile(self_info, self.mask_ratio * 100)

        # if title is not None:
        #     with open(os.path.join(self.path, title+'_prob_token.tsv'), 'w', encoding='utf-8') as f:
        #         for token, info in zip(tokens, self_info):
        #             f.write(f"{token}\t{info}\n")
        #     with open(os.path.join(self.path, title+'_prob_sent.tsv'), 'w', encoding='utf-8') as f:
        #         for sent, info in zip(sents, sent_self_info):
        #             f.write(f"{sent}\n{info}\n\n")

        for sent, info in zip(sents, self_info):
            if info < self.ppl_threshold:
                masked_sents.append(sent)
                sents_after_mask.append(self.mask_a_sent(sent, mask_level))
            else:
                sents_after_mask.append(sent)
        masked_context = " ".join(sents_after_mask) if mask_level == 'sent' else "".join(sents_after_mask)
        
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
    
    def random_mask_context(self, sents: List[str], level) -> str:
        sents_after_mask = []
        masked_sents = []
        for sent in sents:
            if random.random() < self.mask_ratio:
                masked_sents.append(sent)
                sents_after_mask.append(self.mask_a_sent(sent, level))
            else:
                sents_after_mask.append(sent)
        masked_context = " ".join(sents_after_mask)
        return masked_context, masked_sents

    def mask_a_sent(self, sent, level):
        if level == 'phrase':
            return self.phrase_mask_token
        elif level == 'sent':
            if self.keep_leading_word:
                leading_few_words = " ".join(word_tokenize(sent)[:self.num_lead_words]) + " "
            else:
                leading_few_words = ""
            return leading_few_words + self.mask_token
        elif level == 'token':
            return ''
    
    @classmethod
    def from_checkpoint(cls, pickle_path):
        with open(pickle_path, 'rb') as f:
            manager = pickle.load(f)
        manager._prepare_self_info()
        return manager

class ConversationContextManager(ArxivContextManager):

    def __init__(
        self,
        path : str,
        mask_ratio = 0.2, 
        keep_leading_word = True,
        num_lead_words = 3,
        ppl_threshold = None,
        tokenizer = None,
    ):
        super().__init__(path, mask_ratio, keep_leading_word, num_lead_words, ppl_threshold, tokenizer)
    
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
                if random.random() < self.mask_ratio:
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
        self.ppl_threshold = np.percentile(sent_self_info, self.mask_ratio * 100)
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
    dataset_type = "arxiv"
    dataset_manager = {
        'arxiv': ArxivContextManager,
        'conversations': ConversationContextManager,
    }
    context_manager = dataset_manager[dataset_type](context_path)
    contexts = context_manager.generate_context(mask_type, mask_level='phrase')
    print(contexts)
    with open(f'{dataset_type}_contexts_{mask_type}.pkl', 'wb') as f:
        pickle.dump(contexts, f)
