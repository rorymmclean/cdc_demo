### Imports
import streamlit as st
import os
import io
import pymysql
from datetime import datetime
import pandas as pd
import mysql.connector as database
import chromadb
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
# from langchain.cache import SQLiteCache
# langchain.llm_cache = SQLiteCache(database_path="langchain.db")
from langchain.agents import initialize_agent, AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.llms import OpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import LLMMathChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_python_agent
from contextlib import redirect_stdout
from typing import Optional, Type
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

raw_documents = TextLoader('content/runbook.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db3 = Chroma.from_documents(documents, OpenAIEmbeddings())

myusername = "streamlit"
mypassword = "streamlitpass"

myconnection = database.connect(
    user=myusername,
    password=mypassword,
    host="10.1.0.4",
    database="streamlit")

mycursor = myconnection.cursor()

class MySQLTool(BaseTool):
    name = "MySQLTool"
    description = """
    This tool is used for running SQL queries against a MariaDB database. 
    It is useful for when you need to answer questions about employee access 
    activities by running MariaDB queries. Always end your SQL queries with a ";". 

    The following table information is provided to help you write your sql statement. 
    The "demo_access_log" table is a timestamp file of employee activities. The employee is identified by the badge number. 
    The event types in the demo_access_log table include: 
       "BE" = "Building Entry"
       "BX" = "Building Exit"
       "RE" = "Room Entry"
       "RX" = "Room Exit"
    You should not see a "BX" entry type without a "BE" occuring first. 
    You should not see a "RX' entry type without a "RE" occuring first.
    The "demo_emp" table translates tthe badge number to an employee name and assigned department. This table can be joined to the demo_access_log table using the "badge" column
    The "demo_site" table translates the site number to a city and state for that site. This table can be joined to the demo_access_log table using the "site" column.
    The "demo_room" table translates the room at a particular site into a room name. This table can be joined to the demo_access_log table using the "site" and "room" columns.
            
    demo_access_log: (timestamp, badge, event, site, room)
    demo_emp: (badge, name, department)
    demo_site: (site, city, state)
    demo_room: (site, room, name)    
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        if query.startswith(prefix1):
            newquery = query[7:-4]
        elif query.startswith(prefix2):
            newquery = query[4:-4]
        else: 
            newquery = query

        try:
            mycursor.execute(newquery)
            results = cursor.fetchall()
        except database.Error as e:
            results = "Error running query"
        
        return results  
                
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")



### CSS
st.set_page_config(
    page_title='GAMEPLAN', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
padding_top = 0
st.markdown(f"""
    <style>
        .block-container, .main {{
            padding-top: {padding_top}rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# OpenAI Credentials
if not os.environ["OPENAI_API_KEY"]:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = os.environ["OPENAI_API_KEY"]

### UI
col1, col2 = st.columns( [1,5] )
col1.write("")
col1.image('GTLogo.png', width=170)
col2.write("")
col2.subheader('GT AI Driven Cyber Analytics Platform')
# col2.subheader('Generative AI Managed Enterprise PLAtform Network')
# st.markdown('---')


def run_inquiry(myquestion):
    st.session_state.messages.append({"role": "user", "content": myquestion})
    st.chat_message("user").write(myquestion)

    ### Bring in my controlling documents and the additonal template
    relevancy_cutoff = .9
    myprompts = pd.read_csv('content/myprompts.csv')

    docs = db3.similarity_search_with_relevance_scores(myquestion, k=8)

    # st.write(docs)
    mytasks = ""
    for x,v in docs:
        if mytasks == "":
            mytasks = str(x) + "\n\n"
            continue
        if v > relevancy_cutoff:
            mytasks = mytasks + str(x) + "\n\n"

    # st.write(mytasks)
    template=f"""{myprompts[myprompts['name']=='Template']['prompt'].values[0]}
    
    TEXT:
    {mytasks}

    QUESTION:
    {myquestion}
    Provide your justification for this answer.
    """
    
    ### Build an agent that can be used to run SQL queries against the database
    llm4 = ChatOpenAI(model="gpt-4", temperature=0, verbose=False)
    # mydb = SQLDatabase.from_uri("mysql+pymysql://streamlit:streamlitpass@10.1.0.102/streamlit?charset=utf8mb4")
    mydb = SQLDatabase.from_uri("mariadb+pymysql://streamlit:streamlitpass@10.1.0.4/streamlit?charset=utf8mb4")
    toolkit = SQLDatabaseToolkit(db=mydb, llm=llm4)

    sql_agent = create_sql_agent(
        llm=llm4,
        toolkit=toolkit,
        verbose=True,
        AgentType = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        )


    ### Build an agent that can perform mathematics...not used but provided as an example.
    llm_math_chain = LLMMathChain.from_llm(llm=llm4, verbose=False)

    ### Build a chain from the three tools
    tools = [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description=myprompts[myprompts['name']=='Calculator']['prompt'].values[0]
        ),
        Tool(
            name="SQL",
            func=sql_agent.run,
            # func=MySQLTool.run,
            # description=myprompts[myprompts['name']=='SQL']['prompt'].values[0]
            description = """
This tool allows the chatbot to run queries against a MariaDB database. It is useful for when you 
need to answer questions about employee access activities. 

The following tables and columns are the only data objects you can include in your sql statement:
    demo_access_log: (timestamp, badge, event, site, room)
    demo_emp: (badge, name, department)
    demo_site: (site, city, state)
    demo_room: (site, room, name)

The "demo_access_log" table is a timestamp file of employee activities. The employee is identified by the badge number. 
The event types in the demo_access_log table include: 
    "BE" = "Building Entry"
    "BX" = "Building Exit"
    "RE" = "Room Entry"
    "RX" = "Room Exit"
    
The "demo_emp" table translates tthe badge number to an employee name and assigned department. This table can be joined to the demo_access_log table using the "badge" column

The "demo_site" table translates the site number to a city and state for that site. This table can be joined to the demo_access_log table using the "site" column.

The "demo_room" table translates the room at a particular site into a room name. This table can be joined to the demo_access_log table using the "site" and "room" columns.
"""

        )
    ]


    planner = load_chat_planner(llm=llm4)
    executor = load_agent_executor(
        llm4, 
        tools, 
        verbose=True,
    )
    pe_agent = PlanAndExecute(
        planner=planner, 
        executor=executor,  
        verbose=True, 
        max_iterations=2,
        max_execution_time=180,
        AgentType = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )

    if show_detail:
        f = io.StringIO()
        with redirect_stdout(f):
            with st.spinner("Processing..."):
                response = pe_agent(template)
    else:
        with st.spinner("Processing..."):
            response = pe_agent(template)

    st.session_state.messages.append({"role": "assistant", "content": response['output']})    
    st.chat_message('assistant').write(response['output'])

    if show_detail:
        with st.expander('Details', expanded=False):
            s = f.getvalue()
            st.write(s)

with st.sidebar: 
    mysidebar = st.selectbox('Select Model', ['Employee Access'])
    if mysidebar == 'Employee Access':
        show_detail = st.checkbox('Show Details')
        llm_model = st.selectbox('Select Model', ['gpt-4', 'gpt-3.5-turbo'])
        st.markdown("---")
        st.markdown("### Standard Questions:")
        bt1 = st.button('Still Inside')
        bt2 = st.button('What about Bob?')
        st.markdown("---")
        tz = st.container()

if mysidebar == 'Employee Access':
    with st.expander("**:blue[Employee Access Overvew]**"):
        st.markdown("**:blue[This LLM chain uses a Planner/Executor SuperChain.]**")
        st.markdown("**:blue[The workflow first enters the Planner phase where it uses vector semantic KNN search of our \"run book\" document to determine how to answer the question. This document resembles an FAQ document and provides steps for completing these tasks. These are tasks a human might take. The LLM will translate these steps into steps that LangChain can execute.]**")
        st.markdown("**:blue[In the Executor phase the LLM instructs LangChain how to perform each step. Answers from each step are preserved and a final answer is generated by the LLM to the proposed questions. For this demo LLM largely relies on SQL queries. However, the workflow could include other operations including dynamic Python using SciKit, querying the internet, running shell scripts, running REST queries, or any act that might be defined in the CSIO's document. Chains are toolsets that must be assembled to meet a general area of inquiry.]**")

        col1, col2, col3 = st.columns([15, 70, 15])
        col2.image('cyber.jpg',caption='LangChain Structure')

    st.markdown('---')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Ask a question?"):
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        run_inquiry(prompt)
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
    if bt1:
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        run_inquiry("Is Bob still inside the building?")
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
    if bt2:
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        run_inquiry("Is Bob exhibiting unusual behavior?") 
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
