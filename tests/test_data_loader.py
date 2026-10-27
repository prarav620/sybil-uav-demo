import json

from src.data_loader import find_scenarios, load_all, load_scenario


def _write(path, recs):
    path.write_text("\n".join(json.dumps(r) for r in recs))


def make_scenario(root):
    s = root / "scenA"
    s.mkdir()
    _write(s / "traceJSON-10-7-A0-25200-1.json", [
        {"type": 2, "rcvTime": 1.0, "pos": [0, 0, 0]},
        {"type": 3, "rcvTime": 1.1, "sendTime": 1.0, "sender": 20, "senderPseudo": 201,
         "messageID": 1, "pos": [10.0, 5.0, 0], "spd": [3.0, 4.0, 0], "hed": [1.0, 0.0, 0]},
        {"type": 3, "rcvTime": 1.2, "sendTime": 1.1, "sender": 30, "senderPseudo": 301,
         "messageID": 2, "pos": [50.0, 5.0, 0], "spd": [0.0, 1.0, 0], "hed": [0.0, 1.0, 0]},
    ])
    _write(s / "traceGroundTruthJSON-20-8-A0-25200-1.json", [
        {"messageID": 1, "attackerType": 0}, {"messageID": 2, "attackerType": 16}])
    return s


def test_load_scenario(tmp_path):
    s = make_scenario(tmp_path)
    df = load_scenario(s)
    assert len(df) == 2
    assert df.loc[df.message_id == 1, "speed"].iloc[0] == 5.0
    assert df.set_index("message_id")["is_attacker"].to_dict() == {1: 0, 2: 1}
    assert (df.receiver == 10).all()


def test_load_all(tmp_path):
    make_scenario(tmp_path)
    assert find_scenarios(tmp_path)
    assert len(load_all(tmp_path)) == 2
