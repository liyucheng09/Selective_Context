import arxiv
import sys
from glob import glob
import tarfile
from parse_tex import pruned_latex_to_text, TextExtractor
from tqdm import tqdm
import os
import traceback
import shutil
import logging
import json
import re
import time

def resolve_input_commands(latex_code, base_dir="."):
    input_pattern = re.compile(r"(?<!\\)\\input\{(.*?)\}")
    comment_pattern = re.compile(r"(?<!\\)%.*")

    def replace_input(match):
        filename = match.group(1)
        file_path = os.path.join(base_dir, filename)
        if not file_path.endswith(".tex"):
            file_path += ".tex"
        with open(file_path, "r", encoding='utf-8', errors='ignore') as input_file:
            content = input_file.read()
        return resolve_input_commands(content, base_dir=os.path.dirname(file_path))

    # Remove comments
    code_no_comments = comment_pattern.sub("", latex_code)

    # Resolve input commands
    resolved_code = input_pattern.sub(replace_input, code_no_comments)

    return resolved_code

if __name__ == '__main__':
    output_dir = sys.argv[1]

    logging.basicConfig(filename=os.path.join(output_dir, '2_arxiv_downloader.log'), level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    search = arxiv.Search(
        query='submittedDate:[20230201181133 TO 20230316181133]',
        max_results = 8000,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    # print('num of search results', len(search.results()))
    text_save_dir = os.path.join(output_dir, 'text_2')
    if not os.path.exists(text_save_dir):
        os.makedirs(text_save_dir)

    count = 1
    text_extractor = TextExtractor()
    for result in tqdm(search.results()):
        meta_data = {
            'entry_id': result.entry_id,
            'published': result.published.strftime("%Y%m%d%H%M%S"),
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'primary_category': result.primary_category,
            'categories': result.categories
        }
        # logging.info(f'------ META {meta_data}')
        try:
            result.download_source(output_dir, filename = f'{count}.arxiv_source')
        except Exception as e:
            logging.error(f'ERROR: {e}')
            time.sleep(4)
            continue
    
        try:
            path = os.path.join(output_dir, str(count))
            source_file = os.path.join(output_dir, f'{count}.arxiv_source')
            with tarfile.open(source_file) as tar:
                tar.extractall(path)
                
            logging.info(f"Processing source file: {source_file}")

            extracted_files = os.listdir(path)
            tex_files = [file for file in extracted_files if file.endswith('.tex')]
            if len(tex_files)>1:
                if 'main.tex' in tex_files: tex_files=['main.tex']
                else:
                    logging.info(f'Multiple tex files for {source_file}')
                    shutil.rmtree(path)
                    continue
            tex_file = tex_files[0]
            with open(os.path.join(path, tex_file), encoding='utf-8', errors='ignore') as f:
                latex_code = f.read()
                if '\\input' in latex_code:
                    latex_code = resolve_input_commands(latex_code, path)
                text = text_extractor.extract(latex_code)
            
            meta_data['text'] = text
            text_save_path = os.path.join(text_save_dir, f'{count}.json')
            with open(text_save_path, 'w', encoding='utf-8', errors='ignore') as f:
                json.dump(meta_data, f)

            logging.info(f'saved to {text_save_path}')
            shutil.rmtree(path)
            os.remove(source_file)
            count += 1
        except Exception as e:
            logging.error(f'ERROR: {e}')
            traceback.print_exc()
            continue