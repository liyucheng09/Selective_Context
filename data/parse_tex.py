from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode
from pylatexenc import latex2text

def remove_figure_nodes(node_list):
    filtered_node_list = []
    for node in node_list:
        # Ignore the 'figure' environment
        if node.isNodeType(LatexEnvironmentNode):
            if node.environmentname in [ 'figure', 'figure*', 'algorithm', 'table', 'table*', 'algorithmic']:
                continue
            elif hasattr(node, 'nodelist'):
                node.nodelist = remove_figure_nodes(node.nodelist)
        filtered_node_list.append(node)
    return filtered_node_list

def parse_tex_ignore_figures(tex_code):
    walker = LatexWalker(tex_code)
    parsed = walker.get_latex_nodes()[0]

    for node in parsed:
        if node.isNodeType(LatexEnvironmentNode):
            if node.environmentname == 'document':
                parsed = [node]
                break

    filtered_nodes = remove_figure_nodes(parsed)
    return filtered_nodes

def pruned_latex_to_text(latex_code):
    result = parse_tex_ignore_figures(latex_code)
    return latex2text.LatexNodes2Text().nodelist_to_text(result)

if __name__ == "__main__":
    tex_code = r"""
    \documentclass{article}
    \begin{document}
    This is a sample \LaTeX document.
    \begin{figure}
        \includegraphics{example.png}
        \caption{An example figure}
    \end{figure}
    This is after the figure.
    \end{document}
    """
    with open('data/arxiv/2303.16195v1.When_to_be_critical_Performance_and_evolvability_in_different_regimes_of_neural_Ising_agents/Alife_journal_2021_arXiv_submission.tex', 'r') as f:
        result = parse_tex_ignore_figures(f.read())
    # print(result)
    print(latex2text.LatexNodes2Text().nodelist_to_text(result))