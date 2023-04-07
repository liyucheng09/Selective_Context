from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode
from pylatexenc import latex2text
from pylatexenc.macrospec import LatexContextDb

def filter_element(context, exclude_elements = []):
    
    new_context = LatexContextDb()

    new_context.unknown_macro_spec = context.unknown_macro_spec
    new_context.unknown_environment_spec = context.unknown_environment_spec
    new_context.unknown_specials_spec = context.unknown_specials_spec

    filter_element_func = lambda dict_to_filter: {k:v for k,v in dict_to_filter.items() if k not in exclude_elements}.values()
    for cat in context.category_list:

        # include this category
        new_context.add_context_category(
            cat,
            macros=filter_element_func(context.d[cat]['macros']),
            environments=filter_element_func(context.d[cat]['environments']),
            specials=filter_element_func(context.d[cat]['specials']),
        )

    return new_context

class TextExtractor:

    def __init__(self):
        self.l2t_context_db = latex2text.get_default_latex_context_db()
        self.l2t_context_db = filter_element(self.l2t_context_db, ['href'])

        self.l2t = latex2text.LatexNodes2Text(latex_context=self.l2t_context_db)
    
    def extract(self, latex_code):
        result = parse_tex_ignore_figures(latex_code)
        return self.l2t.nodelist_to_text(result)

def remove_figure_nodes(node_list):
    filtered_node_list = []
    for node in node_list:
        # Ignore the 'figure' environment
        if node.isNodeType(LatexEnvironmentNode):
            if node.environmentname in [ 'figure', 'figure*', 'algorithm', 'table', 'table*', 'algorithmic']:
                continue
        if hasattr(node, 'nodelist'):
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

def pruned_latex_to_text(latex_code, math_mode = 'remove'):
    result = parse_tex_ignore_figures(latex_code)
    return latex2text.LatexNodes2Text(math_mode = math_mode).nodelist_to_text(result)

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