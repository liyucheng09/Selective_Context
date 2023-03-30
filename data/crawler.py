import arxiv
import sys
from glob import glob
import tarfile
from parse_tex import pruned_latex_to_text
from tqdm import tqdm
import os
import traceback
import shutil
import logging
import json

if __name__ == '__main__':
    output_dir = sys.argv[1]

    logging.basicConfig(filename=os.path.join(output_dir, '2_arxiv_downloader.log'), level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    search = arxiv.Search(
        query='submittedDate:[20230201181133 TO 20230315181133]',
        max_results = 8000,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    # print('num of search results', len(search.results()))
    text_save_dir = os.path.join(output_dir, 'text_2')
    if not os.path.exists(text_save_dir):
        os.makedirs(text_save_dir)

    count = 1
    for result in tqdm(search.results()):
        meta_data = {
            'entry_id': result.entry_id,
            'published': result.published.strftime("%Y%m%d%H%M%S"),
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'primary_category': result.primary_category,
            'categories': result.categories
        }
        logging.info(f'------ META {meta_data}')
        result.download_source(output_dir, filename = f'{count}.arxiv_source')
    
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
                text = pruned_latex_to_text(f.read())
            
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