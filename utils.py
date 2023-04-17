from context_manager import *

def checkout_prob(text, file_path = 'prob.tsv'):
    tokens, self_info = get_self_information(text)
    with open(file_path, 'w') as f:
        for token, info in zip(tokens, self_info):
            print(token, info)
            f.write(token + '\t' + str(info) + '\n')
    print('Finished writing to file: ', file_path)

def read_lexical_units(article: ArxivArticle, mask_level = 'phrase'):
    if mask_level == 'sent':
        lexical_units = article.units[0]
        assert lexical_units.unit_type == 'sent'
    elif mask_level == 'phrase':
        lexical_units = article.units[1]
        assert lexical_units.unit_type == 'phrase'
    elif mask_level == 'token':
        lexical_units = article.units[2]
        assert lexical_units.unit_type == 'token'

    tokens = lexical_units.text[:50] + lexical_units.text[360:421]
    self_info = lexical_units.self_info[:50] + lexical_units.self_info[360:421]
    self_info = [x**1.2 for x in self_info]

    max_score = max(self_info)
    min_score = min(self_info)

    mid = np.percentile(self_info, 50)

    lines = []
    highlighted = []
    buffer = []
    for token, score in zip(tokens, self_info):
        normalized_score = ((score - min_score) / (max_score - min_score)) * 100
        line = f"\\colorize{{{normalized_score}}}{{{token}}}"
        if score > mid:
            if len(buffer) > 0:
                str_ = '\n'.join(buffer)
                lines.append(f"\\underline{{{str_}}}")
                buffer = []
            highlighted.append(line)
            lines.append(line)
        else:
            # token = f"\\sdelete{{{token}}}"
            # line = f"\\colorize{{{normalized_score}}}{{{token}}}"
            buffer.append(line)

    return '\n'.join(lines) + '\n\n\n' + '\n'.join(highlighted)