from io import BytesIO
from pathlib import Path
from unittest import TestCase, mock
from PIL import Image
from google.genai import types
from streamlit.testing.v1 import AppTest
from gemini_chat import normalize_sprite, send_turn

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def sprite_bytes():
    output = BytesIO()
    Image.new("RGBA", (8, 8), (200, 100, 0, 0)).save(output, format="PNG")
    return output.getvalue()


class SpriteTests(TestCase):
    def test_transparency_is_preserved(self):
        image = Image.open(BytesIO(normalize_sprite(sprite_bytes())))
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.getpixel((0, 0)), (200, 100, 0, 0))

    def test_invalid_upload_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_sprite(b"not an image")


class ChatTests(TestCase):
    @mock.patch("gemini_chat.genai.Client")
    def test_image_prompt_and_full_history(self, client_class):
        client = client_class.return_value
        chat = client.chats.create.return_value
        content = types.Content(role="model", parts=[
            types.Part(text="Готово"),
            types.Part(inline_data=types.Blob(data=sprite_bytes(), mime_type="image/png"),
                       thought_signature=b"retain-signature"),
        ])
        chat.send_message.return_value = types.GenerateContentResponse(
            candidates=[types.Candidate(content=content)])
        history = [types.Content(role="user", parts=[types.Part(text="Предыдущая правка")])]
        chat.get_history.return_value = history + [content]
        items, updated = send_turn("test-key", "test-model", history, "Сделай рыцаря", sprite_bytes())
        self.assertEqual(client.chats.create.call_args.kwargs["history"], history)
        config = client.chats.create.call_args.kwargs["config"]
        self.assertEqual(config.response_modalities, ["TEXT", "IMAGE"])
        sent = chat.send_message.call_args.args[0]
        self.assertEqual(sent[0].text, "Сделай рыцаря")
        self.assertEqual(sent[1].inline_data.data, sprite_bytes())
        self.assertEqual([x["kind"] for x in items], ["text", "image"])
        self.assertEqual(updated[-1].parts[1].thought_signature, b"retain-signature")
        client.close.assert_called_once()

    @mock.patch("gemini_chat.genai.Client")
    def test_empty_response_does_not_commit_history(self, client_class):
        chat = client_class.return_value.chats.create.return_value
        chat.send_message.return_value = types.GenerateContentResponse(candidates=[])
        with self.assertRaises(ValueError):
            send_turn("test-key", "test-model", [], "Привет", None)
        chat.get_history.assert_not_called()
        client_class.return_value.close.assert_called_once()


class InterfaceTests(TestCase):
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_without_key_input_disabled(self):
        app = AppTest.from_file(APP_PATH).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertIn("GEMINI_API_KEY", app.info[0].value)

    @mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    def test_sprite_chat_download_and_reset(self):
        # UI runs as a separate script; patch its imported service module below.
        with mock.patch("gemini_chat.send_turn", return_value=(
            [{"kind": "text", "text": "Готово"},
             {"kind": "image", "data": sprite_bytes(), "mime": "image/png"}], [])) as send:
            app = AppTest.from_file(APP_PATH).run()
            app.session_state["sprites"] = {"sprite": {"name": "hero.png", "data": sprite_bytes()}}
            app.run()
            app.selectbox[0].set_value("sprite").run()
            app.chat_input[0].set_value("Сделай рыцаря").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["messages"]), 2)
            self.assertEqual(send.call_args.args[-1], sprite_bytes())
            self.assertEqual(len(app.get("download_button")), 1)
            app.button[0].click().run()
            self.assertEqual(app.session_state["messages"], [])
