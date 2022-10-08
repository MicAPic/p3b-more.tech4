import csv
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

import asyncio
import aioschedule
from aiogram.utils.helper import Helper, HelperMode, ListItem


class States(Helper):
    """
    TODO
    """
    mode = HelperMode.snake_case

    PROFILE_STATE = ListItem()
    TIMESET_STATE = ListItem()


async def get_data(url_link: str, feed_link: str):
    """
    TODO
    """
    txt_arts = ''  # article text
    hd_arts = ''  # article name

    # Getting HTML page
    req = requests.get(url_link)
    website = BeautifulSoup(req.text, 'html.parser')

    # Parsing vedomosti.ru
    if (feed_link == 'https://www.vedomosti.ru'):
        part_txt = website.find_all('p', class_='box-paragraph__text')
        hd_arts = website.find_all(
            'h1', class_='article-headline__title')[0].text.split('\n')
        hd_arts = hd_arts[1].strip(' ')

        # Making a complete text out of parts
        for element in part_txt:
            txt_arts += element.text

    # Parsing  consultant.ru
    elif (feed_link == 'http://www.consultant.ru/' or
            feed_link == 'http://www.consultant.ru/law/review/'):
        hd_arts = website.find_all('h2')
        hd_arts = hd_arts[0].text

        part_txt = website.find_all('p', class_='')
        for i in range(2, len(part_txt)):
            txt_arts += part_txt[i].text

    # Parsing  lenta.ru
    elif (feed_link == 'https://lenta.ru'):
        hd_arts = website.find_all('span', class_='topic-body__title')
        hd_arts = hd_arts[0].text

        part_txt = website.find_all('div', class_='topic-body__content')
        for element in part_txt:
            txt_arts += element.text

    # Parsing  aif.ru
    elif (feed_link == 'https://aif.ru/'):
        HeadArticles = website.find_all('h1')
        HeadArticles = HeadArticles[0].text

        PartText = website.find_all('p')

        for i in range(0,len(PartText)-15):
            TextArticles += ' ' + PartText[i].text

    return [hd_arts.replace('\n', ''), txt_arts.replace('\n', '')]


async def getting_news_to_file(last_time: datetime):
    """
    TODO
    """
    # Initialization of the start string in tsv file
    with open('temp.tsv', 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(
            ['Ссылка статьи', 'Название статьи', 'Содержание статьи'])

    # Opening .txt which contains RSS site resources
    with open('ListRSS_News.txt') as f:
        lines = f.readlines()

    # Running through all the RSS sites in the file
    for rss_url in lines:
        # If the site does not respond, continue
        if requests.get(rss_url.replace('\n', '')).status_code != 200:
            continue

        # Reading all site's RSS
        feed = feedparser.parse(rss_url)

        # Opening .tsv for noting
        with open('temp.tsv', 'a', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter='\t')

            # Running through each RSS article
            for element in feed.entries:
                # Do a time check if it is necessary
                now = datetime.now()
                then = datetime(element.published_parsed.tm_year,
                                element.published_parsed.tm_mon,
                                element.published_parsed.tm_mday,
                                element.published_parsed.tm_hour,
                                element.published_parsed.tm_min)

                if (now - then).total_seconds() < (now - last_time).total_seconds():
                    # Getting data from the site
                    site_data = await get_data(
                        element.links[0].href, feed.feed.link)

                    # Write it to a file
                    writer.writerow(
                        [element.links[0].href, site_data[0], site_data[1]])


async def news_update(last_time: str):
    """
    TODO
    """
    now = datetime.now() - timedelta(days=1)
    last_time = now.strftime('%d/%m/%Y') + ' ' + last_time + ':00'
    last_time = datetime.strptime(last_time, '%d/%m/%Y %H:%M:%S')
    await getting_news_to_file(last_time)


async def scheduler(first_time: str, second_time: str):
    """
    TODO
    """
    aioschedule.every().day.at(first_time).do(news_update,
                                              last_time=second_time)
    aioschedule.every().day.at(second_time).do(news_update,
                                               last_time=first_time)
    while True:
        await aioschedule.run_pending()
        await asyncio.sleep(60)
