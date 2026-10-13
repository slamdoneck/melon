import os
import hashlib
import streamlit as st
from google.genai import errors
from gemini_chat import DEFAULT_MODEL, normalize_sprite, send_turn

st.set_page_config(page_title="Melon · Sprite Studio", page_icon="🍈", layout="wide")
st.title("🍈 Melon · Sprite Studio")
st.caption("Выбери спрайт, опиши идею и дорабатывай результат в диалоге с Gemini.")

for name, value in (("messages", []), ("history", []), ("sprites", {})):
    if name not in st.session_state:
        st.session_state[name] = value

with st.sidebar:
    st.header("Подключение Gemini")
    entered_key = st.text_input("API-ключ Gemini", type="password", key="gemini_api_key",
                               placeholder="Вставь ключ из Google AI Studio")
    st.caption("Ключ хранится только в текущей сессии и не сохраняется в файл.")
    st.link_button("Получить API-ключ", "https://aistudio.google.com/apikey")
    api_key = entered_key.strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        st.caption("Ключ задан. Доступ проверяется при отправке сообщения.")
    def clear_key():
        st.session_state.gemini_api_key = ""
    st.button("Удалить введённый ключ", on_click=clear_key, disabled=not entered_key)
    st.divider()
    st.header("Твои спрайты")
    uploads = st.file_uploader("Загрузить изображения", type=["png", "jpg", "jpeg", "webp"],
                               accept_multiple_files=True, max_upload_size=10)
    for upload in uploads or []:
        data = upload.getvalue()
        identity = hashlib.sha256(data).hexdigest()
        if identity not in st.session_state.sprites:
            try:
                st.session_state.sprites[identity] = {
                    "name": upload.name, "data": normalize_sprite(data)
                }
            except ValueError as exc:
                st.error(f"{upload.name}: {exc}")
    sprite_labels = {key: value["name"] for key, value in st.session_state.sprites.items()}
    selected = st.selectbox("Спрайт для следующего сообщения",
        options=[None, *st.session_state.sprites],
        format_func=lambda key: "Без вложения" if key is None else sprite_labels[key])
    sprite = st.session_state.sprites.get(selected)
    if sprite:
        st.image(sprite["data"], caption=sprite["name"])
    st.caption("Выбранный спрайт прикладывается к следующему сообщению. Предыдущие изображения остаются в контексте чата.")
    with st.expander("Настройки модели"):
        model = st.text_input("Модель Gemini с генерацией изображений",
                             value=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    if st.session_state.get("active_model") != model:
        st.session_state.history = []
        st.session_state.messages = []
        st.session_state.active_model = model
    if st.button("Новый диалог", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

if not api_key:
    st.info("Вставь API-ключ Gemini в поле слева, затем выбери спрайт и напиши запрос.")


def show_message(message, index):
    with st.chat_message(message["role"]):
        for number, item in enumerate(message["items"]):
            if item["kind"] == "text":
                st.markdown(item["text"])
            else:
                st.image(item["data"])
                if message["role"] == "assistant":
                    suffix = {"image/jpeg": "jpg", "image/webp": "webp"}.get(item["mime"], "png")
                    st.download_button("Скачать изображение", item["data"],
                        file_name=f"melon-{index}-{number}.{suffix}", mime=item["mime"],
                        key=f"download-{index}-{number}")


for index, message in enumerate(st.session_state.messages):
    show_message(message, index)

prompt = st.chat_input("Например: сделай этого персонажа рыцарем, сохрани пиксельный стиль",
                       disabled=not api_key or not model.strip())
if prompt:
    user_items = [{"kind": "text", "text": prompt}]
    if sprite:
        user_items.append({"kind": "image", "data": sprite["data"], "mime": "image/png"})
    show_message({"role": "user", "items": user_items}, len(st.session_state.messages))
    try:
        with st.spinner("Gemini работает над ответом…"):
            items, history = send_turn(api_key, model.strip(), st.session_state.history,
                                       prompt, sprite["data"] if sprite else None)
        st.session_state.history = history
        st.session_state.messages.extend([
            {"role": "user", "items": user_items}, {"role": "assistant", "items": items}
        ])
        st.rerun()
    except errors.APIError as exc:
        hints = {
            400: "Проверь модель и запрос: модель должна поддерживать ответы TEXT и IMAGE.",
            401: "Проверь API-ключ Gemini в поле слева.",
            403: "Проверь доступ ключа к Gemini API и выбранной модели.",
            404: "Модель недоступна. Выбери доступную модель с генерацией изображений.",
            429: "Исчерпана квота Gemini. Проверь лимиты и оплату или повтори позже.",
        }
        st.error(hints.get(exc.code, "Gemini временно недоступен. Попробуй ещё раз позже."))
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Не удалось связаться с Gemini. Проверь доступ к сети и повтори запрос.")
