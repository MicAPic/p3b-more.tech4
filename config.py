from aiogram import Bot
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage


BOT_TOKEN = '5799862343:AAH23tbmtPz3q2dOwzT6qYM_1vU4VwBCIBg'
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
