from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END


code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","You are a code assistant that is solving Facebook Hackercup problems. "
            "You are an expert in algorithm design and code writing. In the original problem statement "
            "there will be a description of the input and output formats. In your answer "
            "define a single function that will accept in one input case and "
            "runs the algorithm on that input case only. For example, if each test case contains "
            "the integers A,B,C and the problem was named foobar, you should define a function as follows: "
            "def foobar(A,B,C) -> int: #fill in correct code. Ignore the number T as that is the number of "
            "test cases, and I want you to write code to solve a single test case.",
        ),
        ("placeholder", "{messages}"),
    ]
)


class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        problem : HumanMessage with problem ask and statement, including images
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """
    error: str
    messages: List
    generation: str
    iterations: int


def get_graph_for_problem(llm, max_iterations = 3, flag ='reflect'):
    '''
    Returns an agent that tries to solve the Dim Sum Delivery problem
    :params
        model: The LLM to use, must support structured output and multimodal input
        max_iterations: How many times to iterate through genarating code
        flag: Whether to reflect on errors or just retry directly (options: 'reflect','do not reflect')
    :returns
        graph: a compiled graph ready to solve the Dim Sum Delivery problem
    '''

    code_gen_chain = code_gen_prompt | llm.with_structured_output(code)

    ### Nodes
    def generate(state: GraphState):
        """
        Generate a code solution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        print("---GENERATING CODE SOLUTION---")

        # State
        messages = state["messages"]
        iterations = state['iterations'] or 0
        error = state["error"]

        # We have been routed back to generation with an error
        if error == "yes":
            messages += [
                (
                    "user",
                    "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
                )
            ]

        # Solution
        code_solution = code_gen_chain.invoke(
            {"messages": messages}
        )
        messages += [
            (
                "assistant",
                f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
            )
        ]

        # Increment
        iterations = iterations + 1
        return {"generation": code_solution, "messages": messages, "iterations": iterations}


    def code_check(state: GraphState):
        """
        Check code

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        print("---CHECKING CODE---")

        # State
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]

        # Get solution components
        imports = code_solution.imports
        code = code_solution.code

        # Check imports
        try:
            exec(imports)
        except Exception as e:
            print("---CODE IMPORT CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the import test: {e}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }

        # Check execution
        try:
            exec(imports + "\n" + code)
        except Exception as e:
            print("---CODE BLOCK CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the code execution test: {e}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }

        # No errors
        print("---NO CODE TEST FAILURES---")
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "no",
        }


    def reflect(state: GraphState):
        """
        Reflect on errors

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        print("---GENERATING CODE SOLUTION---")

        # State
        messages = state["messages"]
        iterations = state["iterations"]
        code_solution = state["generation"]

        # Prompt reflection

        # Add reflection
        reflections = code_gen_chain.invoke(
            {"messages": messages}
        )
        messages += [("assistant", f"Here are reflections on the error: {reflections}")]
        return {"generation": code_solution, "messages": messages, "iterations": iterations}


    ### Edges
    def decide_to_finish(state: GraphState):
        """
        Determines whether to finish.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations == max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            if flag == "reflect":
                return "reflect"
            else:
                return "generate"

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("generate", generate)  # generation solution
    workflow.add_node("check_code", code_check)  # check code
    workflow.add_node("reflect", reflect)  # reflect
    # Build graph
    workflow.add_edge(START,"generate")
    workflow.add_edge("generate", "check_code")
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "reflect": "reflect",
            "generate": "generate",
        },
    )
    workflow.add_edge("reflect", "generate")
    app = workflow.compile()
    return app
