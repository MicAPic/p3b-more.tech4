# -*- coding: utf-8 -*-

import json
import time
import asyncio
from aiogram import types, executor
from aiogram.utils.helper import Helper, HelperMode, ListItem
from aiogram.utils.markdown import text, bold, italic
from aiogram.types import ParseMode

from config import bot, dp
from utils import States


async def on_shutdown(dp):
    await dp.storage.close()
    await dp.storage.wait_closed()


@dp.message_handler(state=States.PROFILE_STATE)
async def first_test_state_case_met(message: types.Message):
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    data[message.from_user.id] = message.text

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)

    state = dp.current_state(user=message.from_user.id)
    await message.answer('Принято!')
    await state.reset_state()


@dp.message_handler(state=States.TIMESET_STATE)
async def first_test_state_case_met(message: types.Message):
    state = dp.current_state(user=message.from_user.id)
    await state.reset_state()


@dp.message_handler(commands=['start'])
async def process_help_command(message: types.Message):
    """
    Handles /start command. Displays the welcome message and invite to start.
    """

    msg = text(bold('Новостной портал '), '+', bold('3балла'), '!\nПривет! '
               + 'Этот бот подбирает новости, актуальные для Вашей '
               + 'профессиональной деятельности, дважды в день. Чтобы указать '
               + 'свою профессию, нажмите на следующую команду: /profile.',
               sep='')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(state='*', commands=['profile'])
async def process_help_command(message: types.Message):
    """
    Handles /profile command. Changes the state to PROFILE_STATE.
    """

    state = dp.current_state(user=message.from_user.id)
    await state.set_state(States.all()[0])

    msg = text(bold('Напишите Вашу профессию:'), sep='')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(state='*', commands=['timeset'])
async def process_help_command(message: types.Message):
    """
    Handles /timeset command. 
    """

    state = dp.current_state(user=message.from_user.id)
    await state.set_state(States.all()[1])

    msg = text(bold('Укажите часы, в которое Вы хотите получать новости ')
               + bold('(два числа: утренний и вечерний час):'), sep='')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    """
    Handles /help command. Displays a text menu with available options.
    """

    msg = text(bold('Готов ответить на эти команды:'),
               '/profile - указать профессию',
               '/help - увидеть этот список вновь',
               '/timeset - указать время отправки новостей',
               sep='\n')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(
        dispatcher=dp,
        skip_updates=True,
        on_shutdown=on_shutdown
    )
