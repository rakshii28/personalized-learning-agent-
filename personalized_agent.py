# personalized_agent.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Tool
from langchain_core.prompts import ChatPromptTemplate

# Import functions from your other project files
from resources_indexer import index_resources, get_resource_retriever
from agent_tools import fetch_real_time_performance, retrieve_learner_profile

# Ensure environment variables are loaded for the LLM and the frontend
load_dotenv()
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# -----------------------------------------------------------
# 1. SYSTEM PROMPT (Agent's Core Instruction Set)
# -----------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert Personalized Academic Advisor AI. Your goal is to optimize "
    "a student's study habits and maximize learning outcomes. "
    "Follow these steps to generate a complete, personalized learning path:"
    "1. **Identify Need:** First, use the `retrieve_learner_profile` and "
    "`fetch_real_time_performance` tools to understand the student's goals, style, "
    "and current knowledge gaps (e.g., low quiz score in a specific topic)."
    "2. **Find Resources:** Next, use the `Resource_Search` tool to find 3-4 "
    "highly relevant learning materials that match the student's *topic of weakness* "
    "and their *learning style* (Visual, Auditory, Kinesthetic, Reading/Writing)."
    "3. **Synthesize Path:** Finally, generate a **Tailored 7-Day Study Plan** that "
    "integrates the retrieved resources. The plan must be actionable and address "
    "the student's specific academic goal and style. "
    "Output the plan clearly."
)

# -----------------------------------------------------------
# 2. HELPER FUNCTION
# -----------------------------------------------------------
def get_personalized_recommendation(agent_executor, student_query: str):
    """
    Runs the Agent Executor and formats the final output. 
    This is what the Streamlit app imports and uses.
    """
    
    print(f"\n--- Executing Agent for Query ---")
    
    # Invoke the agent executor
    response = agent_executor.invoke({"input": student_query})
    
    # Note: For the Streamlit app, we usually return just the output,
    # but for isolated testing, we print the whole structure.
    print("\n" + "="*50)
    print("✅ FINAL PERSONALIZED LEARNING PATH RECOMMENDATION")
    print("="*50)
    print(response.get("output", "Agent failed to generate a plan."))
    print("="*50)
    
    return response.get("output", "Agent failed to generate a plan.")

# -----------------------------------------------------------
# 3. MAIN EXECUTION BLOCK (Isolated Testing Only)
# -----------------------------------------------------------
if __name__ == "__main__":
    
    print("\n--- Running Isolated Agent Test ---")
    
    # Example to run the agent once for testing:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2) 
    
    # A single test query
    test_query = (
        f"Student ID: S1001. "
        f"Context: I failed a quiz on thermodynamics, what should I study? "
        f"Please generate the personalized study plan."
    )
    
    # Running the full initialization for a single test run
    vectorstore = index_resources()
    if vectorstore:
        resource_retriever = get_resource_retriever(vectorstore)
        rag_tool = Tool(name="Resource_Search", func=resource_retriever.invoke, description="Searches indexed learning materials.")
        
        agent_executor = initialize_agent(
            [fetch_real_time_performance, retrieve_learner_profile, rag_tool], 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True, 
            handle_parsing_errors=True,
            agent_kwargs={'system_message': SYSTEM_PROMPT} 
        )
        print("\n--- Test Agent Execution ---")
        get_personalized_recommendation(agent_executor, test_query)