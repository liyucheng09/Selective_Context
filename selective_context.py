from transformers import GPT2Tokenizer, GPT2LMHeadModel
import openai
import torch
import re
from context_manager import *

class SelectiveContext(ArxivContextManager):

    def __init__(self, model_type = 'gpt2'):

        self.model_type = model_type

        # this means we calculate self-information sentence by sentence
        self.sent_level_self_info = True

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.nlp.add_pipe('merge_noun_chunks')
        self.sent_tokenize_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        self.phrase_mask_token = ''
        self.sent_mask_token = "<...some content omitted.>"

        self._prepare_model()

    def _prepare_model(self):
        if self.model_type == 'gpt2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.model.to(DEVICE)
            self.model.eval()

            print('model loaded')

            self.max_token_length = self.model.config.n_positions
            self.get_self_information = self._get_self_info_via_gpt2
        
        elif self.model_type == 'curie':
            self.max_token_length = 2048

            self.get_self_information = self._get_self_info_via_curie
    
    def get_self_information(self, text: str) -> Tuple[List[str], List[float]]:
        # it takes text as input, and return a list of words and a list of self-information scores
        raise NotImplementedError

    def _get_self_info_via_gpt2(self, text: str) -> Tuple[List[str], List[float]]:
        text = f"<|endoftext|>{text}"
        with torch.no_grad():
            encoding = self.tokenizer(text, return_tensors='pt')
            encoding = encoding.to(DEVICE)
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)
        
        input_ids = encoding['input_ids']
        input_ids_expaned = input_ids[:, 1:].unsqueeze(-1)

        tokens = [self.tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]]
        return tokens, self_info[:, :-1].gather(-1, input_ids_expaned).squeeze(-1).squeeze(0).tolist()

    def _get_self_info_via_curie(self, text: str) -> Tuple[List[str], List[float]]:
        num_retry = 3
        openai.api_key = os.environ["OPENAI_API_KEY"]

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
        return tokens, self_info

    def _lexical_unit(self, sents):

        if self.sent_level_self_info:
            sent_self_info = []
            all_noun_phrases = []
            all_noun_phrases_info = []
            all_tokens = []
            all_token_self_info = []

            for sent in sents:
                print(sent)
                tokens, self_info = self.get_self_information(sent)
                sent_self_info.append(np.mean(self_info))

                all_tokens.extend(tokens)
                all_token_self_info.extend(self_info)

                noun_phrases, noun_phrases_info = self._calculate_lexical_unit(tokens, self_info)

                # We need to add a space before the first noun phrase for every sentence except the first one
                if len(all_noun_phrases) != 0:
                    noun_phrases[0] = f" {noun_phrases[0]}"
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
            tokens, self_info = self.get_self_information(context)
            sent_lexical_units, phrase_lexical_units =  self._calculate_lexical_unit(tokens, self_info)

            return [
                sent_lexical_units,
                phrase_lexical_units,
                LexicalUnits('token', text=tokens, self_info=self_info)
            ]

    def __call__(self, text: str, reduce_ratio: float = 0.35, reduce_level :str = 'phrase') -> List[str]:
        context = self.beautify_context(text)

        self.mask_ratio = reduce_ratio

        sents = re.split(self.sent_tokenize_pattern, context)
        sents = [sent.strip() for sent in sents if sent.strip()]

        # You want the reduce happen at sentence level, phrase level, or token level?
        assert reduce_level in ['sent', 'phrase', 'token'], f"reduce_level should be one of ['sent', 'phrase', 'token'], got {reduce_level}"
        sent_lus, phrase_lus, token_lus = self._lexical_unit(sents)
        lexical_level = {
            'sent': sent_lus,
            'phrase': phrase_lus,
            'token': token_lus
        }

        # context is the reduced context, masked_sents denotes what context has been filtered out
        context, masked_sents = self.self_info_mask(lexical_level[reduce_level].text, lexical_level[reduce_level].self_info, reduce_level)
        return context, masked_sents


def main(
    model_type = 'gpt2',
    file_to_process: str = None,
    file_to_save: str = None,
):
    
    global DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    sc = SelectiveContext(model_type=model_type)

    if file_to_process is None:
        while True:
            text = input("Please input the text you want to reduce: ")
            if text == 'exit':
                break
            context, masked_sents = sc(text)
            print('The resultsing context is: \n')
            print(context, '\n\n')

            print('The content that has been filtered out is: \n')
            print(masked_sents, '\n\n')
    else:
        with open(file_to_process, 'r') as f:
            text = f.read()
        context, masked_sents = sc(text)

        with open(file_to_save, 'w') as f:
            f.write(context)

if __name__ == '__main__':
    main(model_type='curie')