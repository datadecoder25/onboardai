import time
from openai import OpenAI
import streamlit as st
import os


def load_password(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        password = file.read().strip()  # Remove any leading/trailing whitespace
    return password

# Use the function
password_file = 'password.txt'
try:
    api_key = load_password(password_file)
except FileNotFoundError as e:
    print(e)

def get_openai_response(user_input, openai_client):
     """
     This function sends the user input to OpenAI's Chat API and returns the model's response.
     """
     try:
         response = openai_client.chat.completions.create(
             model="gpt-3.5-turbo",  # Specify the model for chat applications
             messages=[
                 {"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": user_input},
             ]
         )
         # Extracting the text from the last response in the chat
         if response.choices:
             return response.choices[0].message.content
         else:
             return "No response from the model."
     except Exception as e:
         return f"An error occurred: {str(e)}"
     
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

API_KEY = api_key
openai_client = OpenAI(api_key = API_KEY)

assistant = openai_client.beta.assistants.create(
    name="Data Jornalist",
    instructions='''You are a personal math tutor. Write and run code to answer math questions.''',
    model="gpt-4o-mini",
    tools=[{"type": "code_interpreter"}],
    )

assistant = openai_client.beta.assistants.update(
    assistant_id=assistant.id,
    )

# Create a new thread with a message that has the uploaded file's ID
thread = openai_client.beta.threads.create()

def math_solver_page():
    # Create an OpenAI client with your API key

    st.title("Data Journalist")

    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for d in st.session_state.chat_history:
        st.write("You:", d['user'])
        st.write("Chatbot:", d['chatbot'])

    input_container_1 = st.empty()
    user_input = input_container_1.text_input(label = 'Question', key = '1')

    # Check for exit commands
    if user_input:
        input_container_2 = st.empty()
        with input_container_2.form(key = 'my_form', clear_on_submit = True):	
            submit_button = st.form_submit_button(label = 'Submit')
            if submit_button:
                response = math_solver(user_input, thread, assistant, openai_client)
                st.session_state.chat_history.append({"user": user_input, "chatbot": response})

                # Refresh the page to show the new message at the bottom
                st.rerun()
                time.sleep(1)			
    
                user_input = input_container_1.text_input(label = 'Question', key = '2')
            
                input_container_2.empty()
    else:
        # Update the session state with the current user input
        st.session_state.input_text = user_input

    