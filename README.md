# +3balla (vol. 2)

## Disclaimer

This branch is a fork of the main one, although it deserves its own repository.
Here, we demonstrate the development of our project within the scope of the
"Theoretical Informatics" academic discipline.

---

## Description

Our service allows analyzing a continuous stream of news from RSS various
sources. We can track current trends and create digests based on them.

Interaction with the user is done through a Telegram bot.

Using the collected data, our service identifies trends, which are the most
discussed topics recently. These topics are then compiled into a unified digest
and forwarded to a preselected channel.

## Launch

To run our service, uncomment and fill in the config.py file Bot token
parameter from @BotFather:

- BOT_TOKEN = <Bot token, str>

To launch our bot, install the dependencies from requirements.txt and run the
main.py file from the command line or any IDE. This can be done by first
navigating to the desired installation directory using the following commands:

> git clone git:https://github.com/plus3balla/more.tech4/tree/news-trends

> cd p3b-more.tech4

> pip install -r requirements.txt

> python -m spacy download ru_core_news_lg

> python main.py

The demonstration bot is available at
[@plus3balla_bot](https://t.me/plus3balla_bot).
