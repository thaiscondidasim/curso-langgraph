# Lição 2 - Subgrafos

Estamos construindo um **assistente de pesquisa multiagente** que integra todos os módulos deste curso. Para isso, acabamos de ver o tema da **paralelização**, que é um ponto importante quando falamos de **controlabilidade**. Agora, vamos explorar **subgrafos**, outro conceito fundamental de controle.

Agora, vamos abordar os subgrafos](https://langchain-ai.github.io/langgraph/how-tos/subgraph/#simple-example).

## Estado

Os subgrafos permitem criar e gerenciar diferentes estados em diferentes partes do seu grafo.

Isso é particularmente útil para sistemas multiagentes, com equipes de agentes, cada um com seu próprio estado.

Vamos considerar um exemplo prático:

- Tenho um sistema que aceita logs
- Ele executa duas subtarefas separadas por agentes diferentes (resumir logs, encontrar modos de falha)
- Quero executar essas duas operações em dois subgrafos diferentes.

O mais importante a entender é como os gráficos se comunicam!

Resumindo, a comunicação é **feita com chaves sobrepostas**:

- Os subgráficos podem acessar `docs` a partir do gráfico pai
- O gráfico pai pode acessar `summary/failure_report` a partir dos subgráficos

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb1abf89f2d847ee6f1ff_sub-graph1.png)

## Entrada

Vamos definir um esquema para os logs que serão inseridos em nosso gráfico.

Os **subgrafos** permitem criar e gerenciar diferentes estados em diferentes partes do grafo. Isso é muito útil em **sistemas multiagentes**, como equipes de agentes, onde cada um possui seu próprio estado.

Aqui está o subgráfico de análise de falhas, que utiliza `FailureAnalysisState`.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state):
    """ Get logs that contain a failure """
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """ Generate summary of failures """
    failures = state["failures"]
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}

fa_builder = StateGraph(FailureAnalysisState,output=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

graph = fa_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:9cecce87-b1f5-476b-957d-12d933d4f908:image.png)

Aqui está o subgráfico de resumo da pergunta, que usa `QuestionSummarizationState`.

```python
# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    # Add fxn: summary = summarize(generate_summary)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}

def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}

qs_builder = StateGraph(QuestionSummarizationState,output=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

graph = qs_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:88b98fc1-af62-4b6c-84f6-e69f848818e8:image.png)

## Adicionando subgráficos ao nosso gráfico pai

Agora, podemos juntar tudo.

Criamos nosso gráfico pai com `EntryGraphState`.

E adicionamos nossos subgráficos como nós!

```
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

```

```python
# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: Annotated[List[Log], add] # This will be USED BY in BOTH sub-graphs
    fa_summary: str # This will only be generated in the FA sub-graph
    report: str # This will only be generated in the QS sub-graph
    processed_logs:  Annotated[List[int], add] # This will be generated in BOTH sub-graphs
```

Mas por que `cleaned_logs` tem um redutor se ele só entra *em* cada subgráfico como entrada? Ele não é modificado.

```
cleaned_logs: Annotated[List[Log], add] # This will be USED BY in BOTH sub-graphs

```

Isso ocorre porque o estado de saída dos subgráficos conterá **todas as chaves**, mesmo que não sejam modificadas.

Os subgráficos são executados em paralelo.

Como os subgráficos paralelos retornam a mesma chave, eles precisam ter um redutor como `operator.add` para combinar os valores de entrada de cada subgráfico.

Mas podemos contornar isso usando outro conceito que discutimos anteriormente.

Podemos simplesmente criar um esquema de estado de saída para cada subgráfico e garantir que o esquema de estado de saída contenha chaves diferentes para publicar como saída.

Na verdade, não precisamos que cada subgráfico produza `cleaned_logs`.

```python
# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str # This will only be generated in the FA sub-graph
    report: str # This will only be generated in the QS sub-graph
    processed_logs:  Annotated[List[int], add] # This will be generated in BOTH sub-graphs

def clean_logs(state):
    # Get logs
    raw_logs = state["raw_logs"]
    # Data cleaning raw_logs -> docs 
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()

from IPython.display import Image, display

# Setting xray to 1 will show the internal structure of the nested graph
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

![image.png](attachment:ed904f6a-b5cf-4bfc-9422-cba041e8fc97:image.png)

```python
# Dummy logs
question_answer = Log(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)

question_answer_feedback = Log(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)

raw_logs = [question_answer,question_answer_feedback]
graph.invoke({"raw_logs": raw_logs})
```

```python
{'raw_logs': [{'id': '1',
   'question': 'How can I import ChatOllama?',
   'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"},
  {'id': '2',
   'question': 'How can I use Chroma vector store?',
   'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).',
   'grade': 0,
   'grader': 'Document Relevance Recall',
   'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}],
 'cleaned_logs': [{'id': '1',
   'question': 'How can I import ChatOllama?',
   'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"},
  {'id': '2',
   'question': 'How can I use Chroma vector store?',
   'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).',
   'grade': 0,
   'grader': 'Document Relevance Recall',
   'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}],
 'fa_summary': 'Poor quality retrieval of Chroma documentation.',
 'report': 'foo bar baz',
 'processed_logs': ['failure-analysis-on-log-2',
  'summary-on-log-1',
  'summary-on-log-2']}
```

Vamos considerar um exemplo simples. Suponha que eu tenha um sistema que recebe *logs* e faz duas coisas separadas:

1. Gera um **resumo** desses logs;
2. Identifica **modos de falha** nos logs.

Quero realizar essas duas operações em **subgrafos separados**.

O ponto-chave aqui é: como o **grafo pai** (ou grafo de entrada) **se conecta e se comunica** com os dois subgrafos em termos de estado? E como os subgrafos **se comunicam de volta** com o grafo pai?

A resposta curta: isso é feito por meio de **chaves de estado sobrepostas (overlapping keys)** — esse é o conceito central a entender.

### Exemplo prático:

Meu grafo pai possui uma chave chamada `docs`. Quero que **ambos os subgrafos** tenham acesso a `docs`. Para isso, basta adicionar a chave `docs` ao estado de cada subgrafo. Simples assim.

Também quero que o grafo pai receba os **relatórios** gerados pelos subgrafos. Então, ele precisa ter as chaves `summary_report` e `failure_report` no seu estado. Os subgrafos podem ter chaves internas como `summary` ou `failures` que **não precisam estar** no estado do grafo pai — desde que as chaves compartilhadas estejam presentes nos dois lados.

Isso é o mais importante: **para que a comunicação funcione entre grafo pai e subgrafos, basta garantir que as chaves compartilhadas estejam presentes em ambos.**

Vamos tornar isso concreto:

- Definimos o que é um *log* (basicamente um dicionário de dados).
- Criamos o **subgrafo de análise de falhas**, com um estado próprio e um **esquema de saída (output schema)**. Isso determina **quais dados realmente serão retornados ao grafo pai**.
- Esse subgrafo simula funções simples: isola falhas e gera um resumo das falhas.
- Também criamos o **subgrafo de sumarização**, com seu próprio estado e esquema de saída. Esse subgrafo gera um resumo e, em seguida, simula o envio para o Slack.

### Como conectar os subgrafos ao grafo principal:

1. Definimos o estado do grafo pai, com:
    - `cleaned_logs` (logs limpos que serão passados para os subgrafos),
    - `failure_analysis_summary`,
    - `summary_report`,
    - e `process_logs` (chave em comum que será escrita por ambos os subgrafos).
2. Como `process_logs` será escrito por **ambos os subgrafos**, precisamos de um **reducer** para agregar os dados.
3. Agora, por que `cleaned_logs` teria um reducer se é apenas entrada?
    
    Acontece que **cada subgrafo devolve todas as chaves do seu estado de saída**, mesmo que não sejam modificadas. Isso **poderia causar colisão** se `cleaned_logs` estivesse presente nas saídas dos subgrafos **sem reducer**.
    
4. Solução: usamos **schemas de saída personalizados** nos subgrafos, que **excluem** `cleaned_logs`. Assim, o grafo pai **não precisa de reducer** para essa chave.

Por fim, conectamos tudo:

- Iniciamos com o nó que limpa os logs.
- Depois executamos os dois subgrafos **em paralelo**.
- E finalizamos o grafo.

Testamos com dois logs fictícios e tudo funcionou bem.

### Vantagem adicional dos subgrafos:

Eles tornam os rastreamentos (**traces**) muito mais legíveis, especialmente quando lidamos com sistemas grandes e complexos. No **Langsmith**, podemos visualizar os subgrafos como **componentes colapsáveis**, inspecionar o que está acontecendo internamente, e depois recolher novamente para focar em outras partes do sistema.

---

### **Explicações dos termos técnicos:**

- **Subgraph (subgrafo):** um grafo dentro de outro grafo. Ele executa uma parte específica do fluxo geral, com seu próprio estado e lógica interna.
- **Parent graph (grafo pai):** o grafo principal que chama e coordena os subgrafos.
- **State (estado):** conjunto de dados compartilhados e atualizados durante a execução dos grafos.
- **Overlapping keys (chaves sobrepostas):** chaves do estado que existem tanto no grafo pai quanto no subgrafo. Servem para **troca de informações** entre os dois.
- **Reducer:** função usada para **agregar atualizações** em uma chave do estado que pode receber múltiplos valores simultaneamente (como numa paralelização).
- **Output schema (esquema de saída):** define **quais chaves de estado um subgrafo deve retornar**. Isso ajuda a evitar colisões e simplifica o controle do estado final.
- **Langsmith:** ferramenta visual usada para inspecionar e depurar fluxos construídos com LangGraph. Mostra os rastreamentos de execução com detalhes.

---

Se quiser, posso te ajudar a **visualizar esse grafo com subgrafos em PlantUML estilo C4** ou até gerar **um exemplo de código com LangGraph** para simular esse assistente. Quer seguir por qual caminho?