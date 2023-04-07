from context_manager import get_self_information

def checkout_prob(text, file_path = 'prob.tsv'):
    tokens, self_info = get_self_information(text)
    with open(file_path, 'w') as f:
        for token, info in zip(tokens, self_info):
            print(token, info)
            f.write(token + '\t' + str(info) + '\n')
    print('Finished writing to file: ', file_path)
