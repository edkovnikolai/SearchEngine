from typing import List

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from pandas import DataFrame

main_page_url = "https://www.stlawu.edu"
search_url = main_page_url + '/search?keys='

# headers to make it look like it is a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

}

def get_legible_url(url: str) -> str:
    """
    Check if the url is legible (references stlawu website)
    Add https://www.stlawu.edu in the beginning in case it is a local link
    :param url: string representing url
    :return: legible url or main webpage (https://www.stlawu.edu) if it is not
    """

    lowered_url = url.lower()
    # if the url is picture, return main page url
    if '.jpg' in lowered_url or '.png' in lowered_url:
        return main_page_url
    # cannot require login
    if 'login?' in lowered_url:
        return main_page_url
    # local url
    if url.startswith('/'):
        return main_page_url + url
    # stlawu url
    if url.startswith(main_page_url):
        return url
    # if it is search
    if url.startswith("?keys="):
        return main_page_url + '/search' + url
    return main_page_url

def get_all_legible_links(url: str) -> List[str]:
    """
    Get all the legible links from html page on this url
    :param url: stlawu url
    :return: list of the strings
    """

    legible_links = []

    try:
        response = requests.get(url, headers=headers)
        # print(response.text)

        # if the file not html, return empty list
        if 'html' not in (response.headers.get('Content-Type') or ''):
            return legible_links

        soup = BeautifulSoup(response.text, 'html.parser')

        anchors = soup.find_all('a')

        legible_links_set = set()

        for anchor in anchors:
            legible_links_set.add(get_legible_url(anchor.get('href')))

        legible_links = list(legible_links_set)

    finally:
        return legible_links

# all stlawu links
all_links = {main_page_url, search_url}

# maximum number of links
_link_maximum = 20000

# progress bar
pb = tqdm(desc="web scrapping", total=_link_maximum)

def scrape_website():
    """
    Scrape the whole website and initialize all_links set
    :return:
    """
    links_queue = [main_page_url, search_url]
    i = 0
    pb.update(1)
    while i < len(links_queue) and len(all_links) < _link_maximum:
        cur_link = links_queue[i]
        all_links.add(cur_link)
        new_links = get_all_legible_links(cur_link)
        added_cnt = 0
        for new_link in new_links:
            if new_link not in all_links:
                all_links.add(new_link)
                links_queue.append(new_link)
                added_cnt += 1

        i += 1
        pb.update(n=added_cnt)

scrape_website()

# saving it as a json via pandas DataFrame
def save_data():
    """
    Go through all the links, get the html file and save it as json file
    :return:
    """
    rows = []
    for link in all_links:
        # avoid search links
        if 'search?' in link:
            continue

        response = requests.get(link, headers=headers)
        html_doc = response.text

        # extracting the long background text
        soup = BeautifulSoup(html_doc, 'html.parser')
        bg_text_div = soup.find(class_="feature-words__inner")
        if bg_text_div is not None:
            bg_text_div.extract()

        # extracting all scripts
        scripts = soup.find_all('script')
        for script in scripts:
            script.extract()
        html_doc = soup.prettify()

        row = {'url': link, 'html_doc': html_doc}
        rows.append(row)

    df = DataFrame(data=rows)
    df.to_json('stlawu-webpages.jsonl', orient='records', lines=True)

save_data()



