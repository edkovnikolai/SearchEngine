# from bs4 import BeautifulSoup
import requests

main_page_url = "https://www.stlawu.edu/current-students"

# headers to make it look like it is a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

}

response = requests.get(main_page_url, headers=headers)
print(response.text)

# soup = BeautifulSoup(html_doc, 'html.parser')