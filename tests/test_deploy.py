import json
import urllib.request

from botkit import Agent, chain
from botkit.cli import FILES, main
from botkit.testing import FakeLLM, say


def post(url, body, token=None):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req).read())


def test_server_serves_a_whole_workflow_with_one_graph_per_user(tmp_path):
    g = chain(Agent(name="a", llm=FakeLLM(fn=lambda m, t: say("A")), workdir=tmp_path),
              Agent(name="b", llm=FakeLLM(fn=lambda m, t: say("B")), workdir=tmp_path))
    httpd = g.serve(host="127.0.0.1", port=0, token="t", block=False)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert post(base + "/chat", {"message": "hi", "session": "u1"}, "t")["reply"] == "B"
        health = json.loads(urllib.request.urlopen(base + "/health").read())
        assert health["ok"] and health["version"]
    finally:
        httpd.shutdown()


def test_server_reads_settings_from_the_environment(monkeypatch):
    monkeypatch.setenv("BOTKIT_TOKEN", "secret")
    monkeypatch.setenv("PORT", "0")
    httpd = Agent(llm=FakeLLM(say("x"))).serve(block=False)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            post(base + "/chat", {"message": "hi"})
            assert False, "token from the environment was not enforced"
        except urllib.error.HTTPError as e:
            assert e.code == 401
        assert post(base + "/chat", {"message": "hi"}, "secret")["reply"] == "x"
    finally:
        httpd.shutdown()


def test_cli_init_writes_deploy_files_and_app_compiles(tmp_path, capsys):
    assert main(["init", str(tmp_path)]) == 0
    for name in FILES:
        assert (tmp_path / name).is_file()
    compile((tmp_path / "app.py").read_text(), "app.py", "exec")
    (tmp_path / "Dockerfile").write_text("custom")
    main(["init", str(tmp_path)])
    assert (tmp_path / "Dockerfile").read_text() == "custom"          # never overwritten without --force
    main(["init", str(tmp_path), "--force"])
    assert (tmp_path / "Dockerfile").read_text().startswith("FROM python")


def test_cli_sql_and_version(capsys):
    main(["sql"])
    assert "grant all on table" in capsys.readouterr().out
    main(["version"])
    assert capsys.readouterr().out.strip().count(".") == 2
