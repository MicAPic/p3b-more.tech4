from aiogram.utils.helper import Helper, HelperMode, ListItem


class States(Helper):
    mode = HelperMode.snake_case

    PROFILE_STATE = ListItem()
    TIMESET_STATE = ListItem()
