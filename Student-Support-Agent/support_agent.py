from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from agents import (
    subject_agent_run_sync,
    math_agent_run_sync,
    science_agent_run_sync,
    history_agent_run_sync,
    coding_agent_run_sync,
    general_agent_run_sync
)


def build_query(message, subject, chat_history):
    query = ""
    # Loop over each dictionary in the chat history
    for msg in chat_history:
        # Append user's message to the query
        query += f"User: {msg['message']}\n"
        # If the response was provided by the same agent that will be providing the response
        # to the current message, append the response to the query with the source as "You"
        if msg["subject"] == subject:
            query += f"You: {msg['response']}\n"
        # If the response was provided by another agent, append the
        # response to the query with the source as that agent's name
        else:
            match msg["subject"]:
                case "math":
                    query += f"Math Agent: {msg['response']}\n"
                case "science":
                    query += f"Science Agent: {msg['response']}\n"
                case "history":
                    query += f"History Agent: {msg['response']}\n"
                case "coding":
                    query += f"Coding Agent: {msg['response']}\n"
                case _:
                    query += f"General Agent: {msg['response']}\n"
    # Append the current message to the query
    query += f"User: {message}\nYou:"
    return query


class AgentState(TypedDict):
    chat_history: dict
    message: str
    subject: Literal["math", "science", "history", "coding", "general"]
    response: str


def determine_subject(state: AgentState) -> AgentState:
    state["subject"] = subject_agent_run_sync(state["message"])
    return state

def branch(state: AgentState) -> str:
    return state["subject"]

def answer_math_question(state: AgentState) -> AgentState:
    state["response"] = math_agent_run_sync(build_query(message=state["message"],
                                                        subject=state["subject"],
                                                        chat_history=state["chat_history"]))
    return state

def answer_science_question(state: AgentState) -> AgentState:
    state["response"] = science_agent_run_sync(build_query(message=state["message"],
                                                           subject=state["subject"],
                                                           chat_history=state["chat_history"]))
    return state

def answer_history_question(state: AgentState) -> AgentState:
    state["response"] = history_agent_run_sync(build_query(message=state["message"],
                                                           subject=state["subject"],
                                                           chat_history=state["chat_history"]))
    return state

def answer_coding_question(state: AgentState) -> AgentState:
    state["response"] = coding_agent_run_sync(build_query(message=state["message"],
                                                          subject=state["subject"],
                                                          chat_history=state["chat_history"]))
    return state

def answer_general_question(state: AgentState) -> AgentState:
    state["response"] = general_agent_run_sync(build_query(message=state["message"],
                                                           subject=state["subject"],
                                                           chat_history=state["chat_history"]))
    return state


def build_state_graph():
    graph = StateGraph(AgentState)

    graph.add_node("determine_subject", determine_subject)
    graph.add_node("answer_math_question", answer_math_question)
    graph.add_node("answer_science_question", answer_science_question)
    graph.add_node("answer_history_question", answer_history_question)
    graph.add_node("answer_coding_question", answer_coding_question)
    graph.add_node("answer_general_question", answer_general_question)

    graph.add_conditional_edges(
        "determine_subject",
        branch,
        {
            "math": "answer_math_question",
            "science": "answer_science_question",
            "history": "answer_history_question",
            "coding": "answer_coding_question",
            "general": "answer_general_question"
        }
    )

    graph.add_edge(START, "determine_subject")
    graph.add_edge("determine_subject", END)

    return graph.compile()


def get_response(state_graph, message, chat_history):
    out = state_graph.invoke(AgentState(chat_history=chat_history,
                                        message=message,
                                        subject="general",
                                        response=""))    
    return {
        "message": out["message"],
        "subject": out["subject"],
        "response": out["response"]
    }


if __name__ == "__main__":
    graph = build_state_graph()

    chat_history = []

    while True:
        message = input("Enter your message: ")

        if message.strip() == "":
            break

        response = get_response(graph, message, chat_history[-5:])
        print(f"Agent: {response['response']}")
        chat_history.append(response)