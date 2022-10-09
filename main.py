# -*- coding: utf-8 -*-

import json
import asyncio
import aioschedule
from aiogram import types, executor
from aiogram.utils.markdown import text, bold, italic
from aiogram.types import ParseMode

from config import bot, dp
from utils import States, scheduler, trends_update


async def on_shutdown(dp):
    """
    Closes RAM storage on shutdown.
    """
    await dp.storage.close()
    await dp.storage.wait_closed()


@dp.message_handler(state=States.PROFILE_STATE)
async def first_test_state_case_met(message: types.Message):
    """
    Records the users' occupation on the data.
    """
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    data[message.from_user.id] = message.text

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)

    msg = 'Записано! Для установки времени нажмите команду /timeset.'
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)

    state = dp.current_state(user=message.from_user.id)
    await state.reset_state()


@dp.message_handler(state=States.TIMESET_STATE)
async def first_test_state_case_met(message: types.Message):
    """
    Sets a scheduler with daily notifications at assigned times.
    """
    first_time, second_time = message.text.split()
    asyncio.create_task(scheduler(first_time, second_time,
                                  message.from_user.id))

    msg = 'Принято! Ожидайте новости в указанное время.'
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)

    state = dp.current_state(user=message.from_user.id)
    await state.reset_state()


@dp.message_handler(commands=['start'])
async def process_help_command(message: types.Message):
    """
    Handles /start command. Displays the welcome message and invite to start.
    """

    msg = text(bold('Новостной портал '), '+', bold('3балла'), '!\nПривет! '
               + 'Этот бот два раза в день подбирает новости, актуальные для '
               + 'Вашей профессиональной деятельности. Чтобы указать '
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

    # markup = ReplyKeyboardMarkup(one_time_keyboard = True)
    # markup.add(...)

    kb = [[types.KeyboardButton(text='Accountant'),
           types.KeyboardButton(text='CEO')]]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder='Используйте кнопки ниже.'
    )
    await message.answer('Укажите Вашу профессию:', reply_markup=keyboard)
    

@dp.message_handler(state='*', commands=['timeset'])
async def process_help_command(message: types.Message):
    """
    Handles /timeset command. Set the time of news output.
    """

    state = dp.current_state(user=message.from_user.id)
    await state.set_state(States.all()[1])

    msg = text('Укажите время, в которое Вы хотите получать новости утром и ',
               'вечером в формате ', bold('часы:минуты часы:минуты'),
               '.\nНапример, это может выглядеть так: 8:07 17:21.', sep='')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(state='*', commands=['stop'])
async def process_help_command(message: types.Message):
    """
    Handles /stop command. Terminate the digest.
    """

    state = dp.current_state(user=message.from_user.id)
    await state.reset_state()
    aioschedule.clear()

    msg = text('Дайджест остановлен!', sep='')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(commands=['trends'])
async def process_help_command(message: types.Message):
    """
    Handles /trends command. Displays a list of trends over the last month.
    """
    pass


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    """
    Handles /help command. Displays a text menu with available options.
    """

    msg = text(bold('Готов ответить на эти команды:'),
               '/profile - указать профессию',
               '/timeset - указать время отправки новостей',
               '/stop - остновить отправку новостей',
               '/trends - вывод трендов за месяц',
               '/help - увидеть этот список вновь',
               sep='\n')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(
        dispatcher=dp,
        skip_updates=True,
        on_shutdown=on_shutdown
    )
