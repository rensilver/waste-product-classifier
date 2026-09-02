from waste_product_classifier.benchmark import vlm_zero_shot
from waste_product_classifier.benchmark.vlm_zero_shot import classify_with_vlm, parse_json_response


def test_parse_json_response_extracts_valid_json_object():
    text = 'Sure, here you go: {"label": "organic", "confidence": 0.8, "reason": "peel"}'

    result = parse_json_response(text)

    assert result == {"label": "organic", "confidence": 0.8, "reason": "peel"}


def test_parse_json_response_returns_unparsable_when_no_json_object_present():
    result = parse_json_response("I cannot answer that.")

    assert result == {"label": None, "confidence": None, "reason": "unparsable"}


def test_parse_json_response_returns_unparsable_on_malformed_json():
    result = parse_json_response("{not valid json}")

    assert result == {"label": None, "confidence": None, "reason": "unparsable"}


class FakeOllamaClient:
    def __init__(self, response_content):
        self._response_content = response_content
        self.last_call = None

    def chat(self, model, messages):
        self.last_call = {"model": model, "messages": messages}
        return {"message": {"content": self._response_content}}


def test_classify_with_vlm_returns_parsed_label_confidence_and_latency(tmp_path, monkeypatch):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    fake_client = FakeOllamaClient('{"label": "recyclable", "confidence": 0.9, "reason": "plastic"}')
    monkeypatch.setattr(vlm_zero_shot, "client", fake_client)

    result = classify_with_vlm(image_path, model_name="qwen2.5vl:3b")

    assert result["label"] == "recyclable"
    assert result["confidence"] == 0.9
    assert result["reason"] == "plastic"
    assert result["latency_s"] >= 0
    assert fake_client.last_call["model"] == "qwen2.5vl:3b"


def test_classify_with_vlm_defaults_to_the_3b_model(tmp_path, monkeypatch):
    """Docker only pulls qwen2.5vl:3b, so the default model_name must match that
    exact tag or callers that omit it (e.g. the Streamlit app) will 404."""
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    fake_client = FakeOllamaClient('{"label": "organic", "confidence": 0.5, "reason": "n/a"}')
    monkeypatch.setattr(vlm_zero_shot, "client", fake_client)

    classify_with_vlm(image_path)

    assert fake_client.last_call["model"] == "qwen2.5vl:3b"
