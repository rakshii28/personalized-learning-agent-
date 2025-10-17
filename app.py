# app.py
import streamlit as st
import os
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Import components from your project files
from resources_indexer import index_resources, get_resource_retriever
from agent_tools import fetch_real_time_performance, retrieve_learner_profile

# Import the core system prompt from the agent file
from personalized_agent import SYSTEM_PROMPT 

# --- Setup and Caching ---

load_dotenv()
# The indexer initializes GOOGLE_API_KEY, but we ensure it's loaded here too
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Use Streamlit's caching decorator to run the expensive setup only ONCE
@st.cache_resource
def setup_agent():
    """Initializes the LLM, Vector Store, and Agent Executor."""
    
    # 1. Index Resources (RAG Setup)
    st.info("Initializing Agent: Indexing learning resources (this runs only once)...")
    vectorstore = index_resources()
    
    if vectorstore is None:
        st.error("FATAL ERROR: Could not initialize vector store. Check API key and logs.")
        return None
        
    resource_retriever = get_resource_retriever(vectorstore)
    
    # 2. LLM Setup
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    # 3. Define Tools
    rag_tool = Tool(
        name="Resource_Search",
        func=resource_retriever.invoke,
        description=("Use this tool to search the resource database for specific "
                     "learning materials, lecture notes, or problem sets related to a topic.")
    )
    
    tools = [fetch_real_time_performance, retrieve_learner_profile, rag_tool]
    
    # 4. Initialize Agent
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={'system_message': SYSTEM_PROMPT}
    )
    st.success("Agent Initialized! Ready for personalized learning path requests.")
    return agent_executor

# --- Frontend Layout ---

st.set_page_config(page_title="Personalized Academic Advisor AI", layout="centered")
st.title("🧠 Personalized Academic Advisor AI")
st.caption("Powered by Gemini, LangChain, and ChromaDB")

# Hardcoded student for the demo (S1001)
STUDENT_ID = "S1001"
st.subheader(f"Current Student: {STUDENT_ID}")
st.markdown("---")


# Initialize the agent
agent_executor = setup_agent()


if agent_executor:
    # Get user input
    user_query = st.text_area(
        f"Enter your academic status or request (e.g., 'I failed the latest quiz on thermodynamics'):",
        height=150
    )

    if st.button("Generate Personalized Study Path"):
        if not user_query:
            st.warning("Please enter a query or status update.")
        else:
            # Construct the full query for the agent
            full_query = (
                f"Student ID: {STUDENT_ID}. "
                f"Context: {user_query}. "
                f"Please generate the personalized study plan."
            )
            
            with st.spinner("Agent is generating your plan... (Checking profile, performance, and resources)"):
                # Run the agent (capture verbose output for debugging/insight)
                try:
                    # Capture verbose logs for display
                    with st.expander("Agent's Reasoning (Chain-of-Thought)"):
                        st.session_state.logs = []
                        
                        # Note: LangChain's verbose output goes to the server console.
                        # For a real UI, you'd use callbacks. For simplicity here, 
                        # we print output and show the final result.

                        response = agent_executor.invoke({"input": full_query})
                        
                    # Display the final recommendation clearly
                    st.subheader("✅ Personalized Study Plan Recommendation")
                    st.markdown(response.get("output", "Could not generate a plan. Check agent logs."))
                    
                except Exception as e:
                    st.error(f"An unexpected error occurred during agent execution: {e}")