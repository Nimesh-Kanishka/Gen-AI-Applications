from typing import Literal
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider


model = GroqModel(
    model_name="llama-3.3-70b-versatile",
    provider=GroqProvider(api_key="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")
)

subject_system_prompt = """
You are an agent that determines the subject corresponding to the user's questions.
If the user's question is related to math, reply 'math'.
If the user's question is related to science, reply 'science'.
If the user's question is related to history, reply 'history'.
If the user's question is related to coding, reply 'coding'.
If the user's question is not related to math, science, history, or coding, reply 'general'.
Always reply with a single word only ('math', 'science', 'history', 'coding', or 'general').
"""
subject_agent = Agent(model=model,
                      output_type=Literal["math", "science", "history", "coding", "general"],
                      system_prompt=subject_system_prompt)

math_system_prompt = """
You are an agent that supports students with their math questions.
Provide accurate answers for the user's questions,
including detailed workouts and explanations where necessary.
"""
math_agent = Agent(model=model,
                   system_prompt=math_system_prompt)

science_system_prompt = """
You are an agent that supports students with their science questions.
Provide accurate answers for the user's questions,
including detailed explanations where necessary.
"""
science_agent = Agent(model=model,
                      system_prompt=science_system_prompt)

history_system_prompt = """
You are an agent that supports students with their history questions.
Provide accurate answers for the user's questions,
including detailed explanations where necessary.
"""
history_agent = Agent(model=model,
                      system_prompt=history_system_prompt)

coding_system_prompt = """
You are an agent that supports students with coding.
Provide correct and optimized codes along with detailed explanations of each line.
"""
coding_agent = Agent(model=model,
                     system_prompt=history_system_prompt)

general_system_prompt = """
You are an agent that responds to general questions from the user.
Interact with the user in a friendly and supportive manner.
If any question is not clear, ask the user to clarify.
"""
general_agent = Agent(model=model,
                      system_prompt=general_system_prompt)

def subject_agent_run_sync(message: str):
    response = subject_agent.run_sync(message)
    return response.output

def math_agent_run_sync(message: str):
    response = math_agent.run_sync(message)
    return response.output

def science_agent_run_sync(message: str):
    response = science_agent.run_sync(message)
    return response.output

def history_agent_run_sync(message: str):
    response = history_agent.run_sync(message)
    return response.output

def coding_agent_run_sync(message: str):
    response = coding_agent.run_sync(message)
    return response.output

def general_agent_run_sync(message: str):
    response = general_agent.run_sync(message)
    return response.output