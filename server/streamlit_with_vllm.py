#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json

import requests
import streamlit as st
from loguru import logger

MODEL_NAME = "yi-34b-chat"
title = MODEL_NAME
st.set_page_config(page_title=title)
st.title(title)


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，Yi-34B-Chat大模型，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model_name = st.sidebar.text_input("Model name", MODEL_NAME)
    open_ai_server_url = st.sidebar.text_input("Open AI server URL", "http://localhost:8082/v1/chat/completions")
    max_tokens = st.sidebar.slider("Max tokens", 0, 2048, 512, step=1)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.6, step=0.01)
    stop_token_ids = st.sidebar.text_input("Stop token IDs", "7")

    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        logger.debug(f"[user] {prompt}")
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()

            header = {"accept": "application/json;charset=utf-8",
                      "Content-Type": "application/json"}
            session = requests.Session()
            session.trust_env = False

            d = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "temperature": temperature
            }

            if len(stop_token_ids) > 0:
                d["stop_token_ids"] = [int(t) for t in stop_token_ids.split(",")]

            r = session.post(open_ai_server_url, headers=header, stream=True, json=d)

            if r.encoding is None:
                r.encoding = 'utf-8'
            response = ""
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    try:
                        item = json.loads(line[6:])
                    except Exception:
                        st.toast(line)
                        break

                    if item['choices'][0]['finish_reason'] is not None:
                        break
                    if 'content' in item['choices'][0]['delta']:
                        #                         logger.debug(item['choices'][0]['delta']['content'])
                        response += item['choices'][0]['delta']['content']
                        placeholder.markdown(response)

        messages.append({"role": "assistant", "content": response})
        logger.info(json.dumps(messages, ensure_ascii=False, indent=2))

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
