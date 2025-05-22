# Liçao 4 - Assistente de Pesquisa

## Revisão

Abordamos alguns temas importantes do LangGraph:

- Memória
- Humano no circuito
- Controlabilidade

Agora, reuniremos essas ideias para abordar uma das aplicações mais populares da IA: a automação de pesquisas.

A pesquisa costuma ser um trabalho trabalhoso, repassado aos analistas. A IA tem um potencial considerável para auxiliar nisso.

No entanto, a pesquisa exige personalização: os resultados brutos do LLM costumam ser inadequados para fluxos de trabalho de tomada de decisão no mundo real.

Fluxos de trabalho personalizados [de pesquisa e geração de relatórios](https://jxnl.co/writing/2024/06/05/predictions-for-the-future-of-rag/#reports-over-rag) baseados em IA são uma maneira promissora de abordar isso.

## Objetivo

Nosso objetivo é construir um sistema leve e multiagente em torno de modelos de bate-papo que personalizem o processo de pesquisa.

`Seleção de Fontes`

- Os usuários podem escolher qualquer conjunto de fontes de entrada para sua pesquisa.

`Planejamento`

- Os usuários fornecem um tópico e o sistema gera uma equipe de analistas de IA, cada um com foco em um subtópico.
- O `Human-in-the-loop` será usado para refinar esses subtópicos antes do início da pesquisa.

`Utilização do LLM`

- Cada analista conduzirá entrevistas em profundidade com um especialista em IA, utilizando as fontes selecionadas.
- A entrevista será uma conversa em várias etapas para extrair insights detalhados, conforme mostrado no artigo [STORM](https://github.com/langchain-ai/langgraph/blob/main/examples/storm/storm.ipynb).
- Essas entrevistas serão capturadas usando `subgráficos` com seu estado interno.

`Processo de Pesquisa`

- Os especialistas coletarão informações para responder às perguntas dos analistas em `paralelo`.
- E todas as entrevistas serão conduzidas simultaneamente por meio do `map-reduce`.

`Formato de Saída`

- Os insights coletados em cada entrevista serão sintetizados em um relatório final.
- Usaremos prompts personalizáveis para o relatório, permitindo um formato de saída flexível.

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb164d61c93d48e604091_research-assistant1.png)

## Gerar Analistas: Human-In-The-Loop

Crie analistas e revise-os usando o human-in-the-loop.

```python
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
```

```python
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

def create_analysts(state: GenerateAnalystsState):
    
    """ Create analysts """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')
        
    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback, 
                                                            max_analysts=max_analysts)

    # Generate question 
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
    
    # Write the list of analysis to state
    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass

def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    
    # Otherwise end
    return END

# Add nodes and edges 
builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# View
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

![image.png](attachment:81e9c3ad-ab79-42ea-a47b-caf9f056d68b:image.png)

```python
# Input
max_analysts = 3 
topic = "The benefits of adopting LangGraph as an agent framework"
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream({"topic":topic,"max_analysts":max_analysts,}, thread, stream_mode="values"):
    # Review
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50)  
```

```python
Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: Technical Analyst
Description: Dr. Carter focuses on the technical benefits and challenges of adopting new frameworks. She is particularly interested in the performance, scalability, and integration capabilities of LangGraph.
--------------------------------------------------
Name: Michael Thompson
Affiliation: Business Insights Group
Role: Business Analyst
Description: Michael examines the business implications of adopting new technologies. He is concerned with cost-benefit analysis, return on investment, and the potential for LangGraph to provide a competitive edge.
--------------------------------------------------
Name: Sophia Martinez
Affiliation: User Experience Research Lab
Role: UX Analyst
Description: Sophia is dedicated to understanding how new frameworks impact user experience. She evaluates the usability, learning curve, and overall satisfaction of developers and end-users when using LangGraph.
--------------------------------------------------
```

```python
# Get state and look at next node
state = graph.get_state(thread)
state.next
# ('human_feedback',)
```

```python
# We now update the state as if we are the human_feedback node
graph.update_state(thread, {"human_analyst_feedback": 
                            "Add in someone from a startup to add an entrepreneur perspective"}, as_node="human_feedback")
```

```python
{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1ef6c730-ed11-660c-8002-c9533d30fd7b'}}
```

```python
# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    # Review
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50) 
```

```python
Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: Technical Analyst
Description: Dr. Carter focuses on the technical benefits and challenges of adopting new frameworks. She is particularly interested in the performance, scalability, and integration capabilities of LangGraph.
--------------------------------------------------
Name: Michael Thompson
Affiliation: Business Insights Group
Role: Business Analyst
Description: Michael examines the business implications of adopting new technologies. He is concerned with cost-benefit analysis, return on investment, and the potential for LangGraph to provide a competitive edge.
--------------------------------------------------
Name: Sophia Martinez
Affiliation: User Experience Research Lab
Role: UX Analyst
Description: Sophia is dedicated to understanding how new frameworks impact user experience. She evaluates the usability, learning curve, and overall satisfaction of developers and end-users when using LangGraph.
--------------------------------------------------
Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: Technology Analyst
Description: Dr. Carter focuses on evaluating emerging technologies and their potential impact on various industries. She is particularly interested in how LangGraph can streamline processes and improve efficiency in tech-driven companies.
--------------------------------------------------
Name: Michael Thompson
Affiliation: Green Ventures
Role: Startup Advisor
Description: Michael Thompson is an advisor for early-stage startups, helping them navigate the challenges of growth and innovation. He is keen on understanding how LangGraph can provide a competitive edge to startups by enhancing their operational capabilities.
--------------------------------------------------
Name: Sarah Lee
Affiliation: Global Enterprises Ltd.
Role: Business Strategist
Description: Sarah Lee specializes in developing strategic business plans for large enterprises. She is interested in how LangGraph can be integrated into existing business models to drive scalability and long-term success.
--------------------------------------------------
```

```python
# If we are satisfied, then we simply supply no feedback
further_feedack = None
graph.update_state(thread, {"human_analyst_feedback": 
                            further_feedack}, as_node="human_feedback")
```

```python
{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1ef6c731-5ec7-6e04-8004-626a441d897c'}}
```

```python
# Continue the graph execution to end
for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)
```

```python
final_state = graph.get_state(thread)
analysts = final_state.values.get('analysts')
final_state.next
```

```python
for analyst in analysts:
    print(f"Name: {analyst.name}")
    print(f"Affiliation: {analyst.affiliation}")
    print(f"Role: {analyst.role}")
    print(f"Description: {analyst.description}")
    print("-" * 50) 
```

```python
Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: Technology Analyst
Description: Dr. Carter focuses on evaluating emerging technologies and their potential impact on various industries. She is particularly interested in how LangGraph can streamline processes and improve efficiency in tech-driven companies.
--------------------------------------------------
Name: Michael Thompson
Affiliation: Green Ventures
Role: Startup Advisor
Description: Michael Thompson is an advisor for early-stage startups, helping them navigate the challenges of growth and innovation. He is keen on understanding how LangGraph can provide a competitive edge to startups by enhancing their operational capabilities.
--------------------------------------------------
Name: Sarah Lee
Affiliation: Global Enterprises Ltd.
Role: Business Strategist
Description: Sarah Lee specializes in developing strategic business plans for large enterprises. She is interested in how LangGraph can be integrated into existing business models to drive scalability and long-term success.
--------------------------------------------------
```

## Realizar Entrevista

### Gerar Pergunta

O analista fará perguntas ao especialista.

```python
import operator
from typing import  Annotated
from langgraph.graph import MessagesState

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")
```

```python
question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def generate_question(state: InterviewState):
    """ Node to generate a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question 
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)
        
    # Write messages to state
    return {"messages": [question]}
```

### Gerar Resposta: Paralelização

O especialista coletará informações de diversas fontes em paralelo para responder às perguntas.

Por exemplo, podemos usar:

- Sites específicos, por exemplo, via [`WebBaseLoader`](https://python.langchain.com/v0.2/docs/integrations/document_loaders/web_base/)
- Documentos indexados, por exemplo, via [RAG](https://python.langchain.com/v0.2/docs/tutorials/rag/)
- Busca na web
- Busca na Wikipédia

Você pode experimentar diferentes ferramentas de busca na web, como [Tavily](https://tavily.com/).

```python
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")
```

```python
# Web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_search = TavilySearchResults(max_results=3)
from langchain_community.document_loaders import WikipediaLoader
```

Agora, criaremos nós para pesquisar na web e na Wikipédia.

Também criaremos um nó para responder às perguntas dos analistas.

Por fim, criaremos nós para salvar a entrevista completa e escrever um resumo ("seção") da entrevista.

```python
from langchain_core.messages import get_buffer_string

# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

def search_web(state: InterviewState):
    
    """ Retrieve docs from web search """

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
    
    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state: InterviewState):
    
    """ Retrieve docs from wikipedia """

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
    
    # Search
    search_docs = WikipediaLoader(query=search_query.search_query, 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""

def generate_answer(state: InterviewState):
    
    """ Node to answer a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)
            
    # Name the message as coming from the expert
    answer.name = "expert"
    
    # Append it to state
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    
    """ Save interviews """

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}

def route_messages(state: InterviewState, 
                   name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

section_writer_instructions = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

def write_section(state: InterviewState):

    """ Node to answer a question """

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
   
    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                
    # Append it to state
    return {"sections": [section.content]}

# Add nodes and edges 
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Interview 
memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

# View
display(Image(interview_graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:15e27b84-eaae-4092-b81c-43207a2adeae:image.png)

```python
analysts[0]
```

```python
Analyst(affiliation='Tech Innovators Inc.', name='Dr. Emily Carter', role='Technology Analyst', description='Dr. Carter focuses on evaluating emerging technologies and their potential impact on various industries. She is particularly interested in how LangGraph can streamline processes and improve efficiency in tech-driven companies.')

```

```python
from IPython.display import Markdown
messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
thread = {"configurable": {"thread_id": "1"}}
interview = interview_graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns": 2}, thread)
Markdown(interview['sections'][0])
```

Aqui, executamos a entrevista passando um índice do artigo llama3.1, que está relacionado ao nosso tópico.

```python
from IPython.display import Markdown
messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
thread = {"configurable": {"thread_id": "1"}}
interview = interview_graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns": 2}, thread)
Markdown(interview['sections'][0])
```

```python
## LangGraph: Revolutionizing Agentic Applications in Tech-Driven Industries

### Summary

In the rapidly evolving landscape of emerging technologies, LangGraph stands out as a transformative tool for building stateful, multi-actor applications with large language models (LLMs). Dr. Carter's focus on evaluating such technologies reveals LangGraph's potential to streamline processes and enhance efficiency in tech-driven companies. This report delves into the core principles and benefits of LangGraph, highlighting its unique features and industry applications.

LangGraph, launched in January 2023, is a low-level orchestration framework designed to build agentic applications. It offers unparalleled controllability, allowing developers to define complex workflows involving cycles, which are essential for most agentic architectures [1]. This level of control is crucial for creating sophisticated, customized agentic systems that can adapt to specific business needs [2].

One of the most novel aspects of LangGraph is its ability to balance agent control with agency. This balance is achieved through its core principles: cycles, controllability, and persistence. These principles enable the creation of fully customizable agentic workflows, as opposed to rigid, boilerplate solutions [3]. This flexibility is particularly beneficial for developing advanced chatbots and other interactive applications that require nuanced and dynamic interactions [4].

LangGraph Studio further enhances the development process by providing an integrated development environment (IDE) tailored for agentic applications. This IDE streamlines the creation, testing, and deployment of agentic workflows, making it easier for developers to harness the full potential of LangGraph [5].

The integration of LangGraph with Semantic Web technologies, such as RDF and OWL, offers additional advantages. These technologies enable the encoding of semantics with data, making it machine-readable and facilitating reasoning over heterogeneous data sources [6]. This integration positions LangGraph as a powerful tool for building intelligent systems that can operate across diverse data environments.

In summary, LangGraph represents a significant advancement in the development of agentic applications. Its core principles of cycles, controllability, and persistence, combined with the capabilities of LangGraph Studio and Semantic Web integration, make it a valuable asset for tech-driven companies looking to enhance efficiency and streamline processes.

### Sources
[1] https://github.com/langchain-ai/langgraph/  
[2] https://infohub.delltechnologies.com/en-us/l/rethinking-hierarchical-text-classification-insights-from-multi-agent-experiments-with-small-language-models-1/use-of-langgraph-6/  
[3] https://langchain-ai.github.io/langgraph/concepts/high_level/  
[4] https://www.linkedin.com/pulse/what-langgraph-how-useful-building-llm-based-sarfraz-nawaz-walic/  
[5] https://blog.langchain.dev/langgraph-studio-the-first-agent-ide/  
[6] https://en.wikipedia.org/wiki/Semantic_Web
```

### Paralelizar entrevistas: Mapear-Reduzir

Paralelizamos as entrevistas por meio da API `Send()`, uma etapa de mapeamento.

Combinamos as entrevistas no corpo do relatório em uma etapa de redução.

### Finalizar

Adicionamos uma etapa final para escrever uma introdução e uma conclusão para o relatório final.

```python
import operator
from typing import List, Annotated
from typing_extensions import TypedDict

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report
```

```python
from langgraph.constants import Send

def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """    

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback')
    if human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?"
                                           )
                                                       ]}) for analyst in state["analysts"]]

report_writer_instructions = """You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""

def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
    return {"content": report.content}

intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

# Add nodes and edges 
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report",write_report)
builder.add_node("write_introduction",write_introduction)
builder.add_node("write_conclusion",write_conclusion)
builder.add_node("finalize_report",finalize_report)

# Logic
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

![image.png](attachment:5f475332-40bb-4cb7-bf7f-9be3a0ff1901:image.png)

Vamos fazer uma pergunta aberta sobre o LangGraph.