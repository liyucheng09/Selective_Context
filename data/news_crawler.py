import newspaper
from tqdm import tqdm

def main():
    papers = newspaper.build('https://edition.cnn.com/')
    # papers = newspaper.build('https://www.bbc.co.uk/')

    for article in tqdm(papers.articles):
        print(article.url)
        # article.download()
        # article.parse()
        # if article.publish_date is not None:
        #     print(article.publish_date)
        #     print(article.title)
        #     print()

if __name__ == '__main__':
    main()