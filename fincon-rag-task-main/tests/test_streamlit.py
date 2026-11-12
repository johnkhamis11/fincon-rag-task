from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_runs_and_answers_without_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=45).run()
    assert not app.exception
    assert app.metric[0].value == "29"
    app.text_area[0].set_value("When must a material cybersecurity incident be disclosed under Form 8-K?")
    next(button for button in app.button if button.label == "Retrieve & answer →").click().run()
    assert not app.exception
    result = app.session_state["result"]
    assert result["answer_mode"] == "Source excerpts"
    assert "four business days" in result["answer"].lower()
    next(button for button in app.button if button.label == "Compare pipelines").click().run()
    assert not app.exception
    assert app.session_state["comparison"][0]["hits"]
