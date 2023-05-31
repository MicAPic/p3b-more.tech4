'''Human-machine communication interface'''
# -*- coding: utf-8 -*-

from aiogram import executor, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import ParseMode
from aiogram.utils.helper import Helper, HelperMode, ListItem
from aiogram.utils.markdown import bold, text

from config import dispatcher
from utils import scheduler


class States(Helper):
    '''States of the bot'''

    mode = HelperMode.snake_case
    TIMESET_STATE = ListItem()


@dispatcher.message_handler(state=States.TIMESET_STATE)
async def first_test_state_case_met(message: types.Message) -> None:
    '''Set up a scheduler with daily notifications at scheduled times'''

    first_time, second_time = message.text.split()
    await scheduler(first_time, second_time, message.from_user.id)

    msg = 'Got it! Expect news at the specified time.'
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)

    state = dispatcher.current_state(user=message.from_user.id)
    await state.reset_state()


@dispatcher.message_handler(commands=['start'])
async def process_help_command(message: types.Message) -> None:
    '''Handle /start command. Display a welcome message and invite to start'''

    msg = text(bold('News portal '), '+', bold('3balla'), '!\nHello! ',
               'This bot aggregates news twice a day and sends out a digest.',
               sep='')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


@dispatcher.message_handler(state='*', commands=['timeset'])
async def process_help_command(message: types.Message) -> None:
    '''Handle /timeset command. Set a time of news output'''

    state = dispatcher.current_state(user=message.from_user.id)
    await state.set_state(States.all()[0])

    msg = text('Specify the time at which you would like to receive news ',
               'in the morning and evening in the format ',
               bold('hours:minutes hours:minutes'),
               '.\nFor example, it can look like this: 8:07 17:21.', sep='')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


@dispatcher.message_handler(state='*', commands=['stop'])
async def process_help_command(message: types.Message) -> None:
    '''Handle /stop command. Terminate the digest'''

    state = dispatcher.current_state(user=message.from_user.id)
    await state.reset_state()

    msg = text('Digest stopped!', sep='')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


@dispatcher.message_handler(commands=['help'])
async def process_help_command(message: types.Message) -> None:
    '''Handle /help command. Display a text menu with available options'''

    msg = text(bold('Ready to respond to these commands:'),
               '/timeset - set the news delivery time',
               '/stop - stop sending news',
               '/help - see this list again', sep='\n')
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN)


async def on_shutdown(disp: Dispatcher) -> None:
    '''Close RAM storage on shutdown'''
    await disp.storage.close()
    await disp.storage.wait_closed()


if __name__ == '__main__':
    executor.start_polling(
        dispatcher=dispatcher,
        skip_updates=True,
        on_shutdown=on_shutdown
    )
