'''Parser's auxiliary functions'''
# -*- coding: utf-8 -*-

import asyncio
import csv
import re
from datetime import datetime, timedelta
from typing import List

import aioschedule
import feedparser
import requests
from aiogram.types import ParseMode
from aiogram.utils.markdown import bold, text
from bs4 import BeautifulSoup
from pandas import read_csv

from analytics_wrapper import eval_data, preprocess_df
from config import RSS_LINKS, bot


async def get_data(url_link: str, feed_link: str) -> List[str]:
    '''Retrive data from the site'''
    txt_arts = ''  # article text
    hd_arts = ''  # article name

    # Get HTML page
    req = requests.get(url_link, timeout=5)
    website = BeautifulSoup(req.text, 'html.parser')

    # Parse aif.ru
    if 'aif.ru' in feed_link:
        hd_arts = website.find_all('h1')
        hd_arts = hd_arts[0].text

        part_txt = website.find_all('p')

        for i in range(0, len(part_txt)-15):
            txt_arts += ' ' + part_txt[i].text

    # Parse lenta.ru
    elif 'lenta.ru' in feed_link:
        hd_arts = website.find_all('span', class_='topic-body__title')
        hd_arts = hd_arts[0].text

        part_txt = website.find_all('p', class_='topic-body__content-text')
        for element in part_txt:
            txt_arts += ' ' + element.text

    # Parse rbc.ru
    elif 'rbc.ru' in feed_link:
        hd_arts = website.find_all(
            'h1', class_='article__header__title-in js-slide-title')
        hd_arts = hd_arts[0].text

        part_text = website.find_all('p')
        for i, p_text in enumerate(part_text):
            if i != len(part_text)-2:
                txt_arts += ' ' + p_text.text

    # Parse consultant.ru
    elif 'consultant.ru' in feed_link:
        hd_arts = website.find_all('h2')
        hd_arts = hd_arts[0].text

        part_txt = website.find_all('p', class_='')
        for i in range(2, len(part_txt)):
            txt_arts += ' ' + part_txt[i].text

    # Pare vedomosti.ru
    elif 'vedomosti.ru' in feed_link:
        part_txt = website.find_all('p', class_='box-paragraph__text')
        hd_arts = website.find_all(
            'h1', class_='article-headline__title')[0].text.split('\n')
        hd_arts = hd_arts[1].strip(' ')

        # Make a complete text out of parts
        for element in part_txt:
            txt_arts += ' ' + element.text

    # Delete all \n
    txt_arts = re.sub(r'\n', ' ', txt_arts).replace('\xa0', ' ')
    hd_arts = re.sub(r'\n', ' ', hd_arts).replace('\xa0', ' ')

    return [hd_arts.replace('\n', ''), txt_arts.replace('\n', '')]


async def get_news(last_time: datetime, user_id: int):
    '''Get news from the last_time and send it to user by user_id'''

    now = datetime.now() - timedelta(days=1)
    time = now.strftime('%d/%m/%Y') + ' ' + last_time + ':00'
    time = datetime.strptime(time, '%d/%m/%Y %H:%M:%S')

    # Initialize the start string in tsv file
    with open('temp.tsv', 'w', newline='', encoding='utf8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Date', 'Link', 'Title', 'Text'])

    # Run through all the RSS sites in the file
    for rss_url in RSS_LINKS:
        # If the site does not respond, continue
        if requests.get(rss_url.replace('\n', ''), timeout=10).status_code != 200:
            continue

        # Read all site's RSS
        feed = feedparser.parse(rss_url)

        # Open .tsv for noting
        with open('temp.tsv', 'a', newline='', encoding='utf8') as file:
            writer = csv.writer(file, delimiter='\t')

            # Run through each RSS article
            for element in feed.entries:
                now = datetime.now()
                then = datetime(element.published_parsed.tm_year,
                                element.published_parsed.tm_mon,
                                element.published_parsed.tm_mday,
                                element.published_parsed.tm_hour,
                                element.published_parsed.tm_min)

                if (now - then).total_seconds() \
                        < (now - time).total_seconds():
                    # Get data from the site
                    site_data = await get_data(
                        element.links[0].href, feed.feed.link)

                    # Write it to a file
                    writer.writerow([then, element.links[0].href,
                                     site_data[0], site_data[1]])

    dataframe = read_csv('temp.tsv', sep='\t')
    dataframe = preprocess_df(dataframe)
    digests = eval_data(dataframe)

    msg = text(bold('Top digests from the latest update:'), sep='')
    for i, digest in enumerate(digests):
        article = ''
        for temp in digest[4]:
            article += temp + ' '
        msg += text('\n\n', i+1, ': ', [article], '(', digest[1], ')', sep='')

    await bot.send_message(user_id, msg, parse_mode=ParseMode.MARKDOWN)


async def scheduler(first_time: str, second_time: str, user_id: int) -> None:
    '''Schedule news_update at first_time and second_time daily'''

    fisrt_task = asyncio.create_task(get_news(second_time, user_id))
    second_task = asyncio.create_task(get_news(first_time, user_id))

    aioschedule.every().day.at(first_time).do(fisrt_task)
    aioschedule.every().day.at(second_time).do(second_task)

    while True:
        await asyncio.sleep(60)
        await aioschedule.run_pending()
