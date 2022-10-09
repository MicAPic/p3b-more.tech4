import re
import csv
import json
import pandas as pd
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

import asyncio
import aioschedule
from config import bot
from aiogram.types import ParseMode
from aiogram.utils.markdown import text, bold, italic
from aiogram.utils.helper import Helper, HelperMode, ListItem

from analytics_wrapper import eval_data_4_role, preprocess_df


class States(Helper):
    """
    States of the bot.
    """
    mode = HelperMode.snake_case

    PROFILE_STATE = ListItem()
    TIMESET_STATE = ListItem()


async def get_data(url_link: str, feed_link: str):
    """
    Retrives data from sites.
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
            txt_arts += ' ' + element.text

    # Parsing  consultant.ru
    elif (feed_link == 'http://www.consultant.ru/' or
            feed_link == 'http://www.consultant.ru/law/review/'):
        hd_arts = website.find_all('h2')
        hd_arts = hd_arts[0].text

        part_txt = website.find_all('p', class_='')
        for i in range(2, len(part_txt)):
            txt_arts += ' ' + part_txt[i].text

    # Parsing  lenta.ru
    elif (feed_link == 'https://lenta.ru'):
        hd_arts = website.find_all('span', class_='topic-body__title')
        hd_arts = hd_arts[0].text

        part_txt = website.find_all('div', class_='topic-body__content')
        for element in part_txt:
            txt_arts += ' ' + element.text

    # Parsing  rbc.ru
    elif (feed_link == 'http://www.rbc.ru/'):
        hd_arts = website.find_all(
            'h1', class_='article__header__title-in js-slide-title')
        hd_arts = hd_arts[0].text

        PartText = website.find_all('p')
        for i in range(0, len(PartText)):
            if i != len(PartText)-2:
                txt_arts += ' ' + PartText[i].text

    # Parsing  aif.ru
    elif (feed_link == 'https://aif.ru/'):
        hd_arts = website.find_all('h1')
        hd_arts = hd_arts[0].text

        part_txt = website.find_all('p')

        for i in range(0, len(part_txt)-15):
            txt_arts += ' ' + part_txt[i].text

    # Delete all \n
    txt_arts = re.sub(r'\n', ' ', txt_arts).replace('\xa0', ' ')
    hd_arts = re.sub(r'\n', ' ', hd_arts).replace('\xa0', ' ')
    
    return [hd_arts.replace('\n', ''), txt_arts.replace('\n', '')]


async def getting_news_to_file(last_time: datetime, user_id=0):
    """
    Gets news from the last_time and sends it to user by user_id.
    """
    # Initialization of the start string in tsv file
    with open('temp.tsv', 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(
            ['Date', 'Link', 'Title', 'Text'])

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
                now = datetime.now()
                then = datetime(element.published_parsed.tm_year,
                                element.published_parsed.tm_mon,
                                element.published_parsed.tm_mday,
                                element.published_parsed.tm_hour,
                                element.published_parsed.tm_min)

                if (now - then).total_seconds() \
                        < (now - last_time).total_seconds():
                    # Getting data from the site
                    site_data = await get_data(
                        element.links[0].href, feed.feed.link)

                    # Write it to a file
                    writer.writerow(
                        [then, element.links[0].href,
                         site_data[0], site_data[1]])

    df = pd.read_csv('temp.tsv', sep='\t')
    df = preprocess_df(df)

    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    role = data[str(user_id)]
    digests = eval_data_4_role(role, df)

    msg = text(bold('Главные дайджесты с последнего обновления:'), sep='')

    count = 0
    for digest in digests:
        count += 1
        temp = ''
        for article in digest[1]:
            temp += article + ' '
        msg += text('\n\n', count, ': ', [temp], '(', digest[0], ')', sep='')

    await bot.send_message(user_id, msg, parse_mode=ParseMode.MARKDOWN)


async def news_update(last_time: str, user_id: int):
    """
    Generates a list of news from the last_time using getting_news_to_file
    """
    now = datetime.now() - timedelta(days=1)
    last_time = now.strftime('%d/%m/%Y') + ' ' + last_time + ':00'
    last_time = datetime.strptime(last_time, '%d/%m/%Y %H:%M:%S')
    await getting_news_to_file(last_time, user_id)


async def trends_update():
    """
    Generates a list of news for the last month using getting_news_to_file
    """
    last_time = datetime.now() - timedelta(days=30)
    await getting_news_to_file(last_time)


async def scheduler(first_time: str, second_time: str, user_id: int):
    """
    Schedules news_update at first_time and second_time daily.
    """
    aioschedule.every().day.at(first_time).do(news_update, user_id=user_id,
                                              last_time=second_time)
    aioschedule.every().day.at(second_time).do(news_update, user_id=user_id,
                                               last_time=first_time)
    while True:
        await aioschedule.run_pending()
        await asyncio.sleep(60)
