import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Setting Up the Streamlit App
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemma 2")

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")



groq_api_key=st.sidebar.text_input(label="Groq Api Key", type="password")


if not groq_api_key:
    st.info("Please enter API")
    st.stop()


llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the various information on the topics mentioned"
)

## Initializing Math tool
math_chain=LLMMathChain.from_llm(llm=llm)

calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. It should return the final answer as plain text, without any Python code."
)



prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""


prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm, prompt=prompt_template)

## Initializing Reasoning tool

reasoning_tool=Tool(
    name="Reasoning",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

## Initializing Agent

assistant_agent=initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True

)

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant", "content":"Hi, I'm a Math chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])  

## let's start the conversation

question=st.text_area("Enter your question ",)

if st.button("Find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant","content":response})
            
            st.write('### Response:')
            st.success(response)



    else:
        st.warning("Please enter a quesyion")        





