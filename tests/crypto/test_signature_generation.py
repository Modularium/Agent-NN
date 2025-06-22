from core.crypto import generate_keypair, sign_payload, verify_signature


def test_signature_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("KEY_DIR", str(tmp_path))
    agent = "tester"
    generate_keypair(agent)
    payload = {"foo": "bar"}
    sig = sign_payload(agent, payload)
    assert verify_signature(agent, payload, sig["signature"])
