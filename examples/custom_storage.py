"""Bring your own database: implement the 6 methods of Storage and pass it as memory=.

This one keeps everything in Python dicts. Swap the bodies for MongoDB / Postgres / Redis / a file.
Run:  GROQ_API_KEY=... python examples/custom_storage.py
"""
from collections import defaultdict
from itertools import count

from botkit import Agent, Fact, Storage


class DictStorage(Storage):
    def __init__(self) -> None:
        self.messages = defaultdict(list)   # session -> [message dicts]
        self.facts = defaultdict(dict)      # session -> {id: text}
        self._ids = count(1)

    def add_messages(self, session, messages):
        self.messages[session].extend(messages)

    def get_messages(self, session, limit):
        return self.messages[session][-limit:]

    def clear_messages(self, session):
        self.messages[session] = []

    def add_fact(self, session, text):
        fact_id = next(self._ids)
        self.facts[session][fact_id] = text
        return fact_id

    def list_facts(self, session, limit=30):
        return [Fact(i, t) for i, t in list(self.facts[session].items())[-limit:]]

    def delete_fact(self, session, fact_id):
        return self.facts[session].pop(fact_id, None) is not None


if __name__ == "__main__":
    Agent("groq:openai/gpt-oss-120b", memory=DictStorage()).run()
