import arxiv
import sys
from glob import glob
import tarfile
from parse_tex import pruned_latex_to_text

if __name__ == '__main__':
    output_dir = sys.argv[1]

    search = arxiv.Search(
        query='submittedDate:[20230301181133 TO 20230329181133]',
        max_results = 1000,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order=SortOrder.Descending
    )

    for result in search.results():
        result.download_source(output_dir)
    
    os.mkdir(os.path.join(output_dir, 'text'))
    
    for source_file in os.listdir(output_dir):
        count = 1
        try:
            path = os.path.join(output_dir, count)
            with tarfile.open(source_file) as tar:
                tar.extractall(path)

            extracted_files = os.listdir(path)
            tex_files = [file for file in extracted_files if file.endswith('.tex')]
            if len(tex_files)>1:
                print('Multiple tex files for', source_file)
                os.rmdir(path)
            tex_file = tex_files[0]
            with open(tex_file) as f:
                text = pruned_latex_to_text(f.read())
            
            text_save_path = os.path.join(output_dir, 'text', count)
            with open(text_save_path, 'w') as f:
                f.write(text)

            count += 1
        except:
            continue
    
