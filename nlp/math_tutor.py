import pandas as pd
import time
from openai import OpenAI
import streamlit as st

def math_solver(user_input,thread,assistant, client):
    message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content=user_input
    )
    run = client.beta.threads.runs.create_and_poll(
      thread_id=thread.id,
      assistant_id=assistant.id,
    )
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id = thread.id,
            run_id = run.id
        )
        if run.status=="completed":     
            messages = client.beta.threads.messages.list(
                thread_id = thread.id
            )
        latest_msg = messages.data[0]
        latest_text = latest_msg.content[0].text.value
        break
    return latest_text

def math_solver_page():
    message = st.chat_message("Math Assistant")
    message.write("Hello human, I can solve math problem for you")
    API_KEY = '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'
    client = OpenAI(api_key = API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me any math question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

    # Display assistant response in chat message container
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # prompt = st.chat_input("Ask something")
    # if prompt:
    #     st.write(f"User: {prompt}")
    
    # API_KEY = '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'
    # client = OpenAI(api_key = API_KEY)
    # assistant = client.beta.assistants.create(
    #     name="Math Tutor",
    #     instructions="You are a personal math tutor. Write and run code to answer math questions.",
    #     tools=[{"type": "code_interpreter"}],
    #     model="gpt-4o-mini",
    #     )
    # thread = client.beta.threads.create()
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() in ["quit", "exit", "bye"]:
    #         break
    #     response = math_solver(user_input,thread,assistant, client)
    #     st.write("Chatbot:", response)

    