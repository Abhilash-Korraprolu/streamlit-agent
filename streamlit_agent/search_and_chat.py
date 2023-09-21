from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="AI Data Analyst", page_icon="")
# col1, col2, col3 = st.columns(3)
# with col2:
#     st.image(image='SSAFull2 copy.png')

st.sidebar.header('Upload')
uploaded_file = st.sidebar.file_uploader(
    "Please upload the file you'd like to analyze.",
    help='Supported formats: CSV, XLS, XLSX, XLSM, XLSB',
    on_change=clear_submit,
    )

if uploaded_file:
    df = load_data(uploaded_file)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": 'How can I help?'}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


prompt_template = """
You are a Data Analyst at a Supermarket. I'm your CFO. Your job is to analyze data to answer my questions. These are the rules you must follow while answering:
- Respond like you are reporting on a streamlit app
- Structure the response with headings
- Additionally explain what factors in the data lead this answer. For example, if your answer is that the sales increased by 10 percent this month. You also explain what are the factors in the data that lead to this increase.
- Suggest follow-up questions I could ask you and you could answer with the available data.
- You must use at least one graph. The goal of a graph is to help me understand your presentation easily
- Use plotly when drawing a graph. Draw plots with dark blue shades.
- Assume today's date is 1st Dec 2011

CFO's Question:
"""

# Draw plots with dark blue shades.

if prompt := st.chat_input(placeholder="You question"): # Suggested questions chat hover
    st.session_state.messages.append({"role": "user", "content": prompt_template + prompt})
    st.chat_message("user").write(prompt) # User chat history after hitting enter.

    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key, streaming=True)
    pandas_df_agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant", avatar="🧑‍💻"): 
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # st.session_state.messages = st.session_state.messages.replace(prompt_template, '')
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])

        for monolog in st.session_state.messages:
            if monolog['role'] == 'user':
               monolog['content'] =  monolog['content'].replace(prompt_template, '')

        st.session_state.messages.append({"role": "assistant", "content": response}) #Memory GPT response
        st.write(response)
        print('SESSION STATE MSG_____________________/n', st.session_state.messages)


# Future Work

# Chat Icons
