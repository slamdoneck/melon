"""Gemini multimodal chat; retain complete SDK contents, including signatures."""
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.1-flash-lite-image"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def normalize_sprite(data: bytes) -> bytes:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Спрайт должен быть меньше 10 МБ.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Изображение слишком большое: максимум 16 млн пикселей.")
            image.load()
            output = BytesIO()
            image.convert("RGBA").save(output, format="PNG")
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Не удалось прочитать изображение.") from exc


def create_chat(api_key: str, model: str, history=None):
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
    chat = client.chats.create(
        model=model,
        history=history or [],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            system_instruction=(
                "Ты помощник художника по спрайтам. Следуй запросам пользователя и учитывай "
                "приложенный спрайт. Сам решай, нужно ли уточнение, текстовый ответ или "
                "генерация/редактирование изображения. Сохраняй особенности персонажа, если "
                "пользователь не просит их изменить. Отвечай на языке пользователя."
            ),
        ),
    )
    return client, chat


def send_turn(api_key: str, model: str, history: list, prompt: str, sprite: bytes | None):
    client, chat = create_chat(api_key, model, history)
    try:
        message = [types.Part.from_text(text=prompt)]
        if sprite:
            message.append(types.Part.from_bytes(data=sprite, mime_type="image/png"))
        response = chat.send_message(message)
        items = []
        for part in response.parts or []:
            if part.thought:
                continue
            if part.text:
                items.append({"kind": "text", "text": part.text})
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                items.append({"kind": "image", "data": part.inline_data.data,
                              "mime": part.inline_data.mime_type})
        if not items:
            raise ValueError("Gemini не вернул текст или изображение. Попробуйте изменить запрос.")
        return items, chat.get_history()
    finally:
        client.close()
