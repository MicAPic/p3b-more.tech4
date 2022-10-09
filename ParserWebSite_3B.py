import requests
from bs4 import BeautifulSoup
import feedparser
import csv
from datetime import datetime, timedelta
import re


def GettingDataArticlesFromWeb(URLLink, LinkForFeed):
    TextArticles = ''  # article text
    HeadArticles = ''  # article name

    # Getting HTML page
    req = requests.get(URLLink)
    WebSite = BeautifulSoup(req.text, 'html.parser')

    # Parsing vedomosti.ru
    if (LinkForFeed == 'https://www.vedomosti.ru'):
        PartText = WebSite.find_all('p', class_='box-paragraph__text')
        HeadArticles = WebSite.find_all(
            'h1', class_='article-headline__title')[0].text.split('\n')
        HeadArticles = HeadArticles[1].strip(' ')

        # Making a complete text out of parts
        for element in PartText:
            TextArticles += ' ' + element.text

    # Parsing  consultant.ru
    if (LinkForFeed == 'http://www.consultant.ru/' or
            LinkForFeed == "http://www.consultant.ru/law/review/"):
        HeadArticles = WebSite.find_all("h2")
        HeadArticles = HeadArticles[0].text

        PartText = WebSite.find_all('p', class_='')
        for i in range(2, len(PartText)):
            TextArticles += ' ' + PartText[i].text

    # Parsing  lenta.ru
    if (LinkForFeed == 'https://lenta.ru'):
        HeadArticles = WebSite.find_all('span', class_='topic-body__title')
        HeadArticles = HeadArticles[0].text

        PartText = WebSite.find_all('div', class_='topic-body__content')
        for element in PartText:
            TextArticles += ' ' + element.text

    # Parsing  rbc.ru
    if (LinkForFeed == 'http://www.rbc.ru/'):
        HeadArticles = WebSite.find_all(
            'h1', class_='article__header__title-in js-slide-title')
        HeadArticles = HeadArticles[0].text

        PartText = WebSite.find_all('p')
        for i in range(0, len(PartText)):
            if i != len(PartText)-2:
                TextArticles += ' ' + PartText[i].text

    # Parsing  aif.ru
    if (LinkForFeed == 'https://aif.ru/'):
        HeadArticles = WebSite.find_all('h1')
        HeadArticles = HeadArticles[0].text

        PartText = WebSite.find_all('p')

        for i in range(0, len(PartText)-15):
            TextArticles += ' ' + PartText[i].text

    # Delete all \n
    TextArticles = re.sub(r'\n', ' ', TextArticles).replace('\xa0', ' ')
    HeadArticles = re.sub(r'\n', ' ', HeadArticles).replace('\xa0', ' ')

    return [TextArticles.strip(), HeadArticles.strip()]


# VERIFYTIME bool нужна ли проверка по дате
# QUANITYDAYS bool за какое количество дней нужно брать новости
def GettingNewsToFile(VERIFYTIME, QUANITYDAYS):
    # Иницилизация начальной строчки tsv файла
    with open('dataset.tsv', 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Date', 'Link', 'Title', 'Text'])

    # Открытие .txt содержащий все RSS ресурсы сайтов
    with open('ListRSS_News.txt') as f:
        lines = f.readlines()

    # Получение текущего времени
    if VERIFYTIME:
        DataNow = datetime.now()

    # Running through all the RSS sites in the file
    for RSS_URL in lines:
        # If the site does not respond, continue
        if requests.get(RSS_URL.replace('\n', '')).status_code != 200:
            continue

        # Reading all site's RSS
        feed = feedparser.parse(RSS_URL)

        # Opening .tsv for noting
        with open('dataset.tsv', 'a', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter='\t')

            # Running through each RSS article
            for element in feed.entries:
                DataArtcile = datetime(element.published_parsed.tm_year,
                                       element.published_parsed.tm_mon,
                                       element.published_parsed.tm_mday,
                                       element.published_parsed.tm_hour,
                                       element.published_parsed.tm_min,
                                       element.published_parsed.tm_sec)
                + timedelta(hours=3)
                # Do a time check if it is necessary
                if VERIFYTIME:

                    if (DataNow - DataArtcile).days >= QUANITYDAYS:
                        break

                # Getting data from the site
                DataFromSite = GettingDataArticlesFromWeb(
                    element.links[0].href, feed.feed.link)

                # Write it to a file
                writer.writerow(
                    [DataArtcile, element.links[0].href,
                     DataFromSite[0], DataFromSite[1]])
