from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
import streamlit as st
from IPython.display import Image, display
from io import BytesIO

groq_api_key=st.secrets['groq_api_key']
langsmith=st.secrets['LANGSMITH_API_KEY']

def main():
    st.title("LangGraph Groq Chatbot")
    st.subheader("LangGraph Groq Chatbot")

    llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")
    #llm

    class State(TypedDict):
        # Messages have the type "list". The `add_messages` function
            # in the annotation defines how this state key should be updated
            # (in this case, it appends messages to the list, rather than overwriting them)
        messages:Annotated[list,add_messages]


    def chatbot(state:State):
        return {"messages":llm.invoke(state['messages'])}

    graph_builder=StateGraph(State)

    graph_builder.add_node("chatbot",chatbot)

    graph_builder.add_edge(START,"chatbot")
    graph_builder.add_edge("chatbot",END)

    graph=graph_builder.compile()

    try:
        png_data = st.image(Image(graph.get_graph().draw_mermaid_png()))
        st.image(BytesIO(png_data), caption="Graph Visualization", use_column_width=True)
    except Exception as e:
            print(e)
            print("Error displaying graph image. Please check the graph configuration.")
            pass

  # Reactive input handling
    user_input = st.text_input("Ask me anything", key="user_input")
    if user_input:
        if user_input.lower() in ["quit","q"]:
            print("Good Bye")
            st.stop()
        else:
            for event in graph.stream({'messages':("user",user_input)}):
                #st.write(event.values())
                for value in event.values():
                    #st.write(value['messages'])
                    st.write("AI Agent:",value["messages"].content)


if __name__ == "__main__":
    main()
