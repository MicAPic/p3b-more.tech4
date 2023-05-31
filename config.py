'''Parameters from my.telegram.org and @BotFather'''

from aiogram import Bot
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher


# BOT_TOKEN = <Bot's token, str>
bot = Bot(token=BOT_TOKEN)
dispatcher = Dispatcher(bot, storage=MemoryStorage())


RSS_LINKS = [
    # 'https://aif.ru/rss/articles.php',
    # 'https://aif.ru/rss/news.php',
    'https://lenta.ru/rss/articles',
    # 'https://lenta.ru/rss/news',
    # 'https://rssexport.rbc.ru/rbcnews/news/100/full.rss'
    # 'https://www.consultant.ru/rss/curprof.xml',
    # 'https://www.consultant.ru/rss/db.xml',
    # 'https://www.consultant.ru/rss/fd.xml',
    # 'https://www.consultant.ru/rss/fks.xml',
    # 'https://www.consultant.ru/rss/hotdocs.xml',
    # 'https://www.consultant.ru/rss/md.xml',
    # 'https://www.consultant.ru/rss/nw.xml',
    # 'https://www.consultant.ru/rss/ow.xml',
    # 'https://www.consultant.ru/rss/rm.xml',
    # 'https://www.consultant.ru/rss/zw.xml',
    # 'https://www.vedomosti.ru/rss/articles'
    # 'https://www.vedomosti.ru/rss/news'
]
