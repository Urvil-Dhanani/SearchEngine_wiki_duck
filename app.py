# importing required libraries 
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# preparing tools 
api_wrap_arxiv=ArxivAPIWrapper()
arxiv=ArxivQueryRun(api_wrapper=api_wrap_arxiv)

api_wrap_wiki=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrap_wiki)

searchDuck=DuckDuckGoSearchRun(name="Duck_Search")

tools=[arxiv, wiki, searchDuck]

# setting up streamlit app

st.title("🔎 Search Engine ChatBot")

# sidebar settings

st.sidebar.title("Settings")
groc_API=st.sidebar.text_input(label="Enter Groq API key", type="password")

# runtime memory 
if "messages" not in st.session_state:
    st.session_state["messages"]=[{
        "role":"Assistant",
        "content":"Hi,I'm a chatbot who can search from the web. How can I help you?"
    }]

# chatbot --> chat_message//chat_input
# displaying system message on chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if groc_API:
    # assigning and checking at the same time 
    if prompt:=st.chat_input(placeholder="Please ask your query ...."):
        # update the user question in store and then printin on the screen also to see like chat
        st.session_state.messages=[{"role":"user", "content":prompt}]
        st.chat_message("user").write(prompt)

        # llm with tools  & agents
        llm=ChatGroq(groq_api_key=groc_API, model="Llama3-8b-8192", streaming=True)
        
        # Agent
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
        # ZERO_SHOT will not remeber the past conversation while CHAT_ZERO_SHOT will remember
        search_agent=initialize_agent(tools=tools,
                                    llm=llm,
                                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handlin_parsing_error=True)
        
        # now we disply assistant's result
        with st.chat_message("Assistant"):
            # This callback will let us see the search process of our llm & tools in a creative way 
            st_cb=StreamlitCallbackHandler(parent_container=st.container(),
                                        expand_new_thoughts=False)
            response=search_agent.run(st.session_state.messages,
                                    callbacks=[st_cb])
            st.session_state.messages.append({"role":"Assistant", "content":response})
            st.write(response)
else:
    st.warning(body="Please Enter your Groq API")
    
















