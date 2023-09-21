import streamlit as st
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain import PromptTemplate
import pandas as pd

# Code----------------------------------------------------------------------

# def create_agent():
#     PAID_KEY_OPENAI = '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'
#     data = ['shortlisted_datasets/sales 2yr data.csv']
#     # data = []
#     llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=PAID_KEY_OPENAI)
#     agent = create_csv_agent(llm, data, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    
#     return agent

def create_agent(df):
    PAID_KEY_OPENAI = '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'
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=PAID_KEY_OPENAI)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    
    return agent


def make_api_call(prompt, df):
    agent = create_agent(df)
    ai_response = agent.run(prompt)
    st.write(ai_response)
    print(ai_response)
    return ai_response


def create_prompt(user_query):
    prompt = PromptTemplate(input_variables=['user_query'], template=prompt_template)
    prompt = prompt.format(user_query=user_query)

    return prompt


# Work on causality
# Still having black graphs
prompt_template = """
You are a Data Analyst at a Supermarket. Your job is to analyze data to answer your CFO's questions. These are the rules you must follow while answering:
- Respond like you are reporting to your CFO on a streamlit app
- Structure the response with headings
- Additionally explain what factors in the data lead this answer. For example, if your answer is that the sales increased by 10 percent this month. You also explain what are the factors in the data that lead to this increase.
- Suggest follow-up questions that the CFO could ask you and you could answer with the available data.
- You must use at least one graph. The goal of a graph is to help the CFO understand your presentation easily
- Use plotly when drawing a graph. Draw plots with blue shades. These blues should be darker than #caf0f8
- Assume today's date is 1st Dec 2011

CFO's Question: ### {user_query} ###
"""

prompt = PromptTemplate(input_variables=['user_query'], template=prompt_template)

# UI-----------------------------------------------------------------------
# col1, col2, col3 = st.columns(3)
# with col2:
#     st.image(image='SSAFull2 copy.png')

# Create a sidebar
# st.sidebar.title('Upload your file')
st.sidebar.header('Upload')

# Use the file_uploader function to allow the user to upload a file
uploaded_file = st.sidebar.file_uploader("Please upload the file you'd like to analyze.", type="csv")

# If a file is uploaded, read it into a pandas dataframe
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# st.markdown("<h1 style='text-align: center;'>AI CFO</h1>", unsafe_allow_html=True)
    user_query = st.text_area(label='How can I help?')

    if user_query:
        st.write('Thinking...')
        prompt = create_prompt(user_query=user_query)
        ai_response = make_api_call(prompt, df)
        st.write('------------')

else:
    # File is not uploaded, show a message
    user_query = st.text_area(label='Please upload a file to get started.')


# Exception handling: outputparserexception
