<p align="center">
    <img src="https://github.com/liyucheng09/Selective_Context/blob/main/results/sc.png" alt="Logo of Selective Context" width="auto" height="150" />
</p>

# Selective Context for LLMs

Selective Context compresses your prompt and context to allows LLMs (such as ChatGPT) to process 2x more content. It is especially useful in dealing with long documents and maintaining long conversations without compromising their performance on various tasks!

This repository contains the code and data for the paper: [Compressing Context to Enhance Inference Efficiency of Large Language Models](https://arxiv.org/abs/2310.06201).



### Updates!!

- **Oct 9 2023**: This work has been accepted for the main proceedings of **EMNLP 2023** :partying_face:. The paper link above is the latest conference version. If you are looking for the previous arxiv version of the paper: :point_right: [Unlocking Context Constraints of LLMs](https://arxiv.org/abs/2304.12102).

- **May 6 2023**: Try our demo on [Huggingface Space](https://huggingface.co/spaces/liyucheng/selective_context).

## Key Features

- **Efficient Context Management**: Selective Context maximizes the utility of fixed context length in LLMs, allowing them to process long documents and extended conversations more efficiently.
- **Informativeness Evaluation**: Our method employs a base language model to compute self-information for lexical units (sentences, phrases, or tokens) in a context and use it to evaluate their informativeness.
- **Extensive Evaluation**: We provide extensive evaluations of Selective Context on three data sources (arxiv papers, BBC news articles, and conversation transcripts) and four different NLP tasks (summarization, question answering, original context reconstruction, and conversation).

## Getting Started

To get started, follow these steps:

1. Install `selective-context` via Pypi:
   ```
   pip install selective-context
   python -m spacy download en_core_web_sm
   ```
   If you are processing Chinese, run `python -m spacy download zh_core_web_sm` as well.

2. Import `SelectiveContext`:
   ```
   from selective_context import SelectiveContext
   ```

3. Compress your prompt and context. The `context` contains the compressed context:
   ```
   sc = SelectiveContext(model_type='gpt2', lang='en')
   context, reduced_content = sc(text)
   ```

4. You can also adjust the reduce ratio:
   ```
   context, reduced_content = sc(text, reduce_ratio = 0.5)
   ```

5. If you prefer to try with web interface, try our streamlit app:
   ```
   streamlit run app/app.py
   ```
   Or directly visit our [Space](https://huggingface.co/spaces/liyucheng/selective_context) on Hugging Face Hub.

## Code Structure

- `selective_context.py`: A demo for performing context reduction using Selective Context.
- `context_manager.py`: The main module for managing context and implementing the Selective Context algorithm.
- `main.py`: The main script for running experiments and evaluating the effectiveness of Selective Context.
- `qa_manager.py`: A helper module for managing question answering tasks during the experiments.

## Experiments

To reproduce the experiments from the paper:

1. First, you download the datasets required in the experiments:
```
wget https://github.com/liyucheng09/Selective_Context/releases/download/v0.1.0rc1/datasets_dumps.zip
unzip datasets_dumps.zip
```
2. You run:
```
python main.py datasets_dumps/arxiv datasets_dumps/news datasets_dump/conversation <output_path_to_save_results> <num_articles> <HF_model_name_or_path>
```

## Dataset in the paper

The dataset used in the paper can be found at:

- Arxiv: [HF Hub](https://huggingface.co/datasets/liyucheng/arxiv-march-2023)
- BBC News: [HF Hub](https://huggingface.co/datasets/liyucheng/bbc_new_2303)
- ShareGPT.com: [HF Hub](https://huggingface.co/datasets/liyucheng/sharegpt-500)

The datasets are created by ourselves so if you need citation just use the citation of this tool.

If you have trouble accessing Huggingface Hub, download the data via:
```
wget https://github.com/liyucheng09/Selective_Context/releases/download/v0.1.0rc1/data_dumps.zip
```

## Citation

If you find this repository helpful or use our method in your research, please consider citing our paper:

```
@misc{li2023compressing,
      title={Compressing Context to Enhance Inference Efficiency of Large Language Models}, 
      author={Yucheng Li and Bo Dong and Chenghua Lin and Frank Guerin},
      year={2023},
      eprint={2310.06201},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

The previous version:
```
@misc{li2023unlocking,
      title={Unlocking Context Constraints of LLMs: Enhancing Context Efficiency of LLMs with Self-Information-Based Content Filtering}, 
      author={Yucheng Li},
      year={2023},
      eprint={2304.12102},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
