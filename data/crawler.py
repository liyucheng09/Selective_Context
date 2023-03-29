import arxiv
import sys
from glob import glob
import tarfile
from parse_tex import pruned_latex_to_text
from tqdm import tqdm
import os
import traceback
import shutil

if __name__ == '__main__':
    output_dir = sys.argv[1]

    search = arxiv.Search(
        query='submittedDate:[20230301181133 TO 20230329181133]',
        max_results = 1000,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    # print('num of search results', len(search.results()))
    text_save_dir = os.path.join(output_dir, 'text')
    if not os.path.exists(text_save_dir):
        os.makedirs(text_save_dir)

    count = 1
    for result in tqdm(search.results()):
        result.download_source(output_dir, filename = f'{count}.arxiv_source')
    
        try:
            path = os.path.join(output_dir, str(count))
            source_file = os.path.join(output_dir, f'{count}.arxiv_source')
            with tarfile.open(source_file) as tar:
                tar.extractall(path)

            extracted_files = os.listdir(path)
            tex_files = [file for file in extracted_files if file.endswith('.tex')]
            if len(tex_files)>1:
                print('Multiple tex files for', source_file)
                shutil.rmtree(path)
                continue
            tex_file = tex_files[0]
            with open(os.path.join(path, tex_file)) as f:
                text = pruned_latex_to_text(f.read())
            
            text_save_path = os.path.join(output_dir, 'text', str(count))
            with open(text_save_path, 'w') as f:
                f.write(text)

            shutil.rmtree(path)
            os.remove(source_file)
            count += 1
        except:
            traceback.print_exc()
            continue
    
