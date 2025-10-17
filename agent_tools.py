# agent_tools.py
from langchain.tools import tool
import requests
import os
from pydantic import BaseModel, Field

# Mock data for demonstration - Replace with actual DB/API calls
MOCK_LEARNER_PROFILE = {
    "student_id": "S1001",
    "learning_style": "Visual and Kinesthetic",
    # Updated Academic Goal
    "academic_goal": "Achieve an A in Advanced Computer Science",
    "preferred_resources": "Video lectures, practical simulations, problem sets",
    # Updated low score and weak topic context
    "last_quiz_score": 55, 
    "target_score": 90
}

# 1. Pydantic Schemas for Tool Input/Output
class PerformanceSchema(BaseModel):
    student_id: str = Field(..., description="The unique identifier for the student.")

class ProfileSchema(BaseModel):
    student_id: str = Field(..., description="The unique identifier for the student.")


# 2. Tool 1: Get Real-Time Performance
@tool(args_schema=PerformanceSchema)
def fetch_real_time_performance(student_id: str) -> str:
    """
    Fetches the student's latest quiz score (e.g., 55) and the specific weak 
    topic (e.g., Binary Tree Traversal) from the simulated LMS performance data.
    """
    if student_id == MOCK_LEARNER_PROFILE["student_id"]:
        return (f"Last score: {MOCK_LEARNER_PROFILE['last_quiz_score']} (out of 100). "
                f"Target: {MOCK_LEARNER_PROFILE['target_score']}. "
                f"Current Topic: Binary Tree Traversal.")
    return f"Performance data for {student_id} not found."

# 3. Tool 2: Get Learner Profile
@tool(args_schema=ProfileSchema)
def retrieve_learner_profile(student_id: str) -> str:
    """
    Retrieves the student's static profile including their learning style (e.g., 
    Visual/Kinesthetic) and long-term academic goal for the advisor.
    """
    if student_id == MOCK_LEARNER_PROFILE["student_id"]:
        return (f"Student ID: {MOCK_LEARNER_PROFILE['student_id']}. "
                f"Style: {MOCK_LEARNER_PROFILE['learning_style']}. "
                f"Goal: {MOCK_LEARNER_PROFILE['academic_goal']}. "
                f"Preferred: {MOCK_LEARNER_PROFILE['preferred_resources']}.")
    return f"Profile for {student_id} not found."