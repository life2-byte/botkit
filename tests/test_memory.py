import importlib.util
import sqlite3
from pathlib import Path

import pytest

from botkit import Agent, BotError, Memory, SQLiteStorage, Storage, SupabaseStorage
from fake_postgrest import FakePostgrest
from test_botkit import FakeLLM, call, say

EXAMPLES = Path(__file__).parent.parent / "examples"


def load_dict_storage():
    spec = importlib.util.spec_from_file_location("custom_storage", EXAMPLES / "custom_storage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DictStorage()


@pytest.fixture
def fake():
    server = FakePostgrest()
    yield server
    server.stop()


@pytest.fixture(params=["sqlite", "supabase", "custom"])
def storage(request, tmp_path, fake):
    if request.param == "sqlite":
        return SQLiteStorage(tmp_path / "s.db")
    if request.param == "supabase":
        return SupabaseStorage(fake.url, "service-key")
    return load_dict_storage()


# ---------------------------------------------------------------- the same contract for EVERY backend
def test_storage_contract(storage):
    assert isinstance(storage, Storage)
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    storage.add_messages("a", msgs)
    storage.add_messages("b", [{"role": "user", "content": "other"}])
    assert storage.get_messages("a", 100) == msgs                    # order kept
    assert storage.get_messages("a", 2) == msgs[-2:]                 # last N, oldest first
    assert storage.get_messages("b", 100) == [{"role": "user", "content": "other"}]
    assert storage.get_messages("nobody", 10) == []
    storage.add_messages("a", [])                                    # empty is fine
    storage.clear_messages("a")
    assert storage.get_messages("a", 10) == [] and len(storage.get_messages("b", 10)) == 1

    i1, i2 = storage.add_fact("a", "likes chai"), storage.add_fact("a", "lives in Lahore")
    assert i1 != i2
    assert [f.text for f in storage.list_facts("a")] == ["likes chai", "lives in Lahore"]
    assert [f.text for f in storage.list_facts("a", 1)] == ["lives in Lahore"]
    assert storage.list_facts("b") == []
    assert storage.delete_fact("b", i1) is False                     # other session cannot delete it
    assert storage.delete_fact("a", i1) is True and storage.delete_fact("a", i1) is False
    assert [f.text for f in storage.list_facts("a")] == ["lives in Lahore"]


def test_unicode_and_tool_messages_roundtrip(storage):
    msgs = [
        {"role": "user", "content": "chai پیتا ہوں ☕"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "2"},
    ]
    storage.add_messages("u", msgs)
    assert storage.get_messages("u", 10) == msgs


# ---------------------------------------------------------------- an Agent on every backend
def test_agent_memory_persists_on_any_backend(storage):
    Agent(llm=FakeLLM(say("noted")), memory=storage, session="u1").chat("I am Ahmed")
    llm = FakeLLM(say("Ahmed"))
    bot = Agent(llm=llm, memory=storage, session="u1")
    bot.chat("who am I?")
    assert [m["role"] for m in llm.seen[0][0]] == ["system", "user", "assistant", "user"]
    llm_other = FakeLLM(say("hi"))
    Agent(llm=llm_other, memory=storage, session="u2").chat("hello")
    assert len(llm_other.seen[0][0]) == 2                            # other user sees nothing


def test_facts_flow_through_agent_tools(storage):
    llm = FakeLLM(call("remember", {"fact": "Prefers short answers"}), say("ok"))
    Agent(llm=llm, memory=storage).chat("remember that")
    llm2 = FakeLLM(say("hi"))
    Agent(llm=llm2, memory=storage).chat("hello")
    assert "Prefers short answers" in llm2.seen[0][0][0]["content"]


def test_memory_flag_variants(tmp_path):
    assert Agent(llm=FakeLLM()).memory is None
    assert Agent(llm=FakeLLM(), memory=False).memory is None
    assert Agent(llm=FakeLLM(), memory=tmp_path / "a.db").memory is not None
    assert isinstance(Agent(llm=FakeLLM(), memory=str(tmp_path / "b.db")).memory.storage, SQLiteStorage)
    mem = Memory(SQLiteStorage(), session="x")
    assert Agent(llm=FakeLLM(), memory=mem).memory is mem
    with pytest.raises(TypeError):
        Agent(llm=FakeLLM(), memory=123)
    assert "remember" not in Agent(llm=FakeLLM()).tool_names         # no memory flag, no memory tools


# ---------------------------------------------------------------- SQLite: your own tables
def test_sqlite_extra_tables(tmp_path):
    db = SQLiteStorage(tmp_path / "x.db", extra_schema="CREATE TABLE orders(id INTEGER PRIMARY KEY, item TEXT)")
    assert db.execute("INSERT INTO orders(item) VALUES(?)", ("tea",)) == 1
    assert db.query("SELECT * FROM orders WHERE item=?", ("tea",)) == [{"id": 1, "item": "tea"}]
    db.add_fact("s", "f")
    db.close()
    raw = sqlite3.connect(tmp_path / "x.db")
    names = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"orders", "botkit_messages", "botkit_facts"} <= names


def test_sqlite_prefix_validated_and_used(tmp_path):
    with pytest.raises(ValueError):
        SQLiteStorage(prefix="x; DROP TABLE y;--")
    db = SQLiteStorage(tmp_path / "p.db", prefix="myapp_")
    names = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "myapp_messages" in names and "botkit_messages" not in names


# ---------------------------------------------------------------- Supabase specifics
def test_supabase_sends_key_and_session_filter(fake):
    store = SupabaseStorage(fake.url, "service-key")
    store.add_fact("user 1&x=y", "hello")                           # awkward characters must be escaped
    store.list_facts("user 1&x=y")
    method, path, query, headers = fake.requests[-1]
    assert headers["apikey"] == "service-key" and headers["authorization"] == "Bearer service-key"
    assert query["session"] == ["eq.user 1&x=y"] and path == "/rest/v1/botkit_facts"


def test_supabase_bad_key_missing_tables_and_config(fake, monkeypatch):
    with pytest.raises(BotError, match="401"):
        SupabaseStorage(fake.url, "wrong").list_facts("s")
    fake.tables.pop("botkit_facts")
    with pytest.raises(BotError, match="schema_sql"):
        SupabaseStorage(fake.url, "service-key").list_facts("s")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(BotError, match="SUPABASE_URL"):
        SupabaseStorage()
    with pytest.raises(BotError, match="https"):
        SupabaseStorage("xyz.supabase.co", "k")
    with pytest.raises(BotError, match="Could not reach"):
        SupabaseStorage("http://127.0.0.1:1", "k", timeout=2).list_facts("s")


def test_supabase_env_vars_and_url_forms(fake, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", fake.url + "/")
    monkeypatch.setenv("SUPABASE_KEY", "service-key")
    store = SupabaseStorage()
    store.add_fact("s", "x")
    assert SupabaseStorage(fake.url + "/rest/v1", "service-key").list_facts("s")[0].text == "x"


def test_supabase_custom_tables(fake):
    orders = SupabaseStorage(fake.url, "service-key").table("orders")
    orders.insert([{"item": "tea", "qty": 2}, {"item": "milk", "qty": 5}])
    assert [r["item"] for r in orders.select(order="id.desc")] == ["milk", "tea"]
    assert orders.select({"item": "tea"}, columns="item,qty") == [{"item": "tea", "qty": 2}]
    assert [r["item"] for r in orders.select({"qty": ("gt", 3)})] == ["milk"]
    assert orders.update({"qty": 9}, {"item": "tea"})[0]["qty"] == 9
    assert len(orders.delete({"item": "milk"})) == 1 and len(orders.select()) == 1
    for bad in (lambda: orders.delete({}), lambda: orders.update({"a": 1}, {}),
                lambda: orders.select({"bad col;": 1}), lambda: orders.select(order="id;drop"),
                lambda: orders.select({"a": ("sql", 1)})):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        SupabaseStorage(fake.url, "service-key").table("bad name")


def test_supabase_schema_sql():
    sql = SupabaseStorage.schema_sql()
    assert "botkit_messages" in sql and "botkit_facts" in sql and "jsonb" in sql and "enable row level security" in sql
    assert "app_messages" in SupabaseStorage.schema_sql("app_")
    with pytest.raises(ValueError):
        SupabaseStorage.schema_sql("x;--")


def test_supabase_key_is_redacted_from_tool_output(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_KEY", "service-key-very-secret")
    bot = Agent(llm=FakeLLM(), tools=True, workdir=tmp_path / "w")
    bot._exec({"id": "1", "function": {"name": "write_file",
              "arguments": '{"path": "a.txt", "content": "key service-key-very-secret"}'}}, {})
    out = bot._exec({"id": "2", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}, {})
    assert out == "key [REDACTED]"
