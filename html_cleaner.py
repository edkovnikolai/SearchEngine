"""
Extract only the most useful data from the html. Create a new dataset with cleaned data for it

Notes:
    The most useful information is stored in the <div> tag with class 'dialog-off-canvas-main-canvas'. Everything else
    in the html can be removed.

    Remove all the tags. Leave only anchor tags with all attributes removed
    (to make transformer know that this is a link)

    How to do it (use beautiful soup)
    1) Find the element div with class 'dialog-off-canvas-main-canvas'
    2) Remove <header> and <footer> elements. Also remove all the <figure> tags and comments
    3) For every tag, remove all the attributes
"""

from bs4 import BeautifulSoup, Comment
from datasets import load_dataset


raw_dataset_path = './stlawu-webpages.jsonl'
clean_dataset_path = './stlawu-webpages-clean'

# def _text_extractor():


def clean_html(html_doc: str) -> str:
    """
    Get more content dense html code
    :param html_doc: html document code
    :return: clean html
    """

    soup = BeautifulSoup(html_doc, 'html.parser')
    # find 'dialog-off-canvas-main-canvas' div tag
    main_div = soup.find(class_='dialog-off-canvas-main-canvas')
    # if it happens that main_div is None, just do everything for the main_document
    if main_div is None:
        main_div = soup

    # Getting all elements for deletion
    delete_list = []
    delete_list.extend(main_div.find_all('head'))
    delete_list.extend(main_div.find_all('header'))
    delete_list.extend(main_div.find_all('footer'))
    delete_list.extend(main_div.find_all('figure'))
    delete_list.extend(main_div.find_all(string=lambda x: isinstance(x, Comment)))

    # extracting the delete_list elements
    for el in delete_list:
        el.extract()

    # remove all the attributes in the remaining tags
    for el in main_div.descendants:
        el.attrs = {}
    main_div.attrs = {}

    return main_div.prettify()
    # another option is to simply return the text
    # return main_div.get_text()

if __name__ == '__main__':
    # get the raw dataset with htmls containing a lot of dirty information
    raw_dataset = load_dataset('json', data_files=raw_dataset_path, split='train')

    # clean the html
    clean_dataset = raw_dataset.map(lambda x: {'html_doc': clean_html(x['html_doc'])}, num_proc=10)

    # get rid of everything wrong encoded
    clean_dataset = clean_dataset.filter(lambda x: '<div>' in x['html_doc'])

    # saving the dataset
    clean_dataset.save_to_disk(clean_dataset_path)
