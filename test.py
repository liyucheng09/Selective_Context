from transformers import pipeline, LlamaForCausalLM

generator = pipeline('text-generation', model='gpt2')
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

print(output)
