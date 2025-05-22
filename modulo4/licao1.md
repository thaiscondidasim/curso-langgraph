# Lição 1 - **Execução de nós paralelos**

## Revisão

No módulo 3, nos aprofundamos em `humano no loop`, mostrando 3 casos de uso comuns:

(1) `Aprovação` - Podemos interromper nosso agente, exibir o estado para um usuário e permitir que ele aceite uma ação

(2) `Depuração` - Podemos retroceder o gráfico para reproduzir ou evitar problemas

(3) `Edição` - Você pode modificar o estado

## Objetivos

Este módulo se baseará nos conceitos de `humano no loop` e `memória` discutidos no módulo 2.

Aprofundaremos os fluxos de trabalho `multiagentes` e desenvolveremos um assistente de pesquisa multiagente que une todos os módulos deste curso.

Para construir este assistente de pesquisa multiagente, primeiro discutiremos alguns tópicos de controlabilidade do LangGraph.

Começaremos com [paralelização](https://langchain-ai.github.io/langgraph/how-tos/branching/#how-to-create-branches-for-parallel-node-execution).

## Fan out e fan in

Vamos construir um gráfico linear simples que sobrescreve o estado em cada etapa.

```python
from IPython.display import Image, display

from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    state: str

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# Flow
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:847337a2-5944-4c7e-9aa4-1467c6fc9d25:image.png)

Sobrescrevemos o estado, como esperado.

```python
graph.invoke({"state": []})
```

```python
Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm B"]
Adding I'm D to ["I'm C"]
{'state': ["I'm D"]}
```

Agora, vamos executar `b` e `c` em paralelo.

E então executar `d`.

Podemos fazer isso facilmente com um fan-out de `a` para `b` e `c`, e depois um fan-in para `d`.

As atualizações de estado são aplicadas ao final de cada etapa.

Vamos executá-lo.

```python
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# Flow
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:205f5099-fa26-491c-8cc2-3bd3e282757d:image.png)

**Vemos um erro**!

Isso ocorre porque `b` e `c` estão gravando na mesma chave de estado/canal na mesma etapa.

```python
Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm A"]
An error occurred: At key 'state': Can receive only one value per step. Use an Annotated key to handle multiple values.
```

Ao usar o fan out, precisamos ter certeza de que estamos usando um redutor se os passos estiverem gravando no mesmo canal / chave.

Como mencionamos no Módulo 2, `operator.add` é uma função do módulo operator integrado do Python.

Quando `operator.add` é aplicado a listas, ele realiza a concatenação de listas.

```python
import operator
from typing import Annotated

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    state: Annotated[list, operator.add]

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# Flow
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:c1ed6b2b-dd0d-44dc-b3ba-7b4619672fdd:image.png)

```python
graph.invoke({"state": []})
```

```python
Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm A"]
Adding I'm D to ["I'm A", "I'm B", "I'm C"]
```

Agora vemos que acrescentamos ao estado as atualizações feitas em paralelo por `b` e `c`.

## Aguardando a conclusão dos nós

Agora, vamos considerar um caso em que um caminho paralelo tem mais etapas que o outro.

```python
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("b2", ReturnNodeValue("I'm B2"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# Flow
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b2")
builder.add_edge(["b2", "c"], "d")
builder.add_edge("d", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:2a137476-def2-4d0a-87d9-0812c15d0297:image.png)

Neste caso, `b`, `b2` e `c` fazem parte da mesma etapa.

O gráfico aguardará a conclusão de todas elas antes de prosseguir para a etapa `d`.

### Definindo a ordem das atualizações de estado

No entanto, dentro de cada etapa, não temos controle específico sobre a ordem das atualizações de estado!

Em termos simples, é uma ordem determinística determinada pelo LangGraph com base na topologia do grafo que **não controlamos**.

Acima, vemos que `c` é adicionado antes de `b2`.

No entanto, podemos usar um redutor personalizado para personalizar isso, por exemplo, classificar as atualizações de estado.

```python
def sorting_reducer(left, right):
    """ Combines and sorts the values in a list"""
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]
    
    return sorted(left + right, reverse=False)

class State(TypedDict):
    # sorting_reducer will sort the values in state
    state: Annotated[list, sorting_reducer]

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("b2", ReturnNodeValue("I'm B2"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# Flow
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b2")
builder.add_edge(["b2", "c"], "d")
builder.add_edge("d", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:016046ef-fff2-4fb9-9de1-8098aa4a5220:image.png)

```python
graph.invoke({"state": []})
```

```python
Adding I'm A to []
Adding I'm C to ["I'm A"]
Adding I'm B to ["I'm A"]
Adding I'm B2 to ["I'm A", "I'm B", "I'm C"]
Adding I'm D to ["I'm A", "I'm B", "I'm B2", "I'm C"]
```

Agora, o redutor classifica os valores de estado atualizados!

O exemplo `sorting_reducer` classifica todos os valores globalmente. Também podemos:

1. Gravar as saídas em um campo separado no estado durante a etapa paralela
2. Usar um nó "sink" após a etapa paralela para combinar e ordenar essas saídas
3. Limpar o campo temporário após a combinação

Consulte a [documentação](https://langchain-ai.github.io/langgraph/how-tos/branching/#stable-sorting) para mais detalhes.

## Trabalhando com LLMs

Agora, vamos adicionar um exemplo realista!

Queremos coletar contexto de duas fontes externas (Wikipedia e Web-Seach) e pedir que um LLM responda a uma pergunta.

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0) 
```

```python
class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]
```

Você pode tentar diferentes ferramentas de busca na web. [Tavily](https://tavily.com/) é uma boa opção a ser considerada, mas certifique-se de que sua `TAVILY_API_KEY` esteja definida.]

```python
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

def search_web(state):
    
    """ Retrieve docs from web search """

    # Search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state):
    
    """ Retrieve docs from wikipedia """

    # Search
    search_docs = WikipediaLoader(query=state['question'], 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def generate_answer(state):
    
    """ Node to answer a question """

    # Get state
    context = state["context"]
    question = state["question"]

    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, 
                                                       context=context)    
    
    # Answer
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])
      
    # Append it to state
    return {"answer": answer}

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("search_web",search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Flow
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:5e2cf22f-bc07-4632-8333-be2b73ba00a4:image.png)

```python
result = graph.invoke({"question": "How were Nvidia's Q2 2024 earnings"})
result['answer'].content
```

```python
"Nvidia's Q2 2024 earnings were notably strong. The company reported a GAAP net income of $6.188 billion, a significant increase from $656 million in the same quarter the previous year. Revenue for the quarter was $13.507 billion, up from $6.704 billion in Q2 2023. The GAAP gross profit was $9.462 billion with a gross margin of 70.1%. Non-GAAP gross profit was $9.614 billion with a gross margin of 71.2%. Operating income also saw a substantial rise to $6.800 billion from $499 million in the previous year. Overall, Nvidia demonstrated robust financial performance in Q2 2024."
```

No Módulo 3, exploramos profundamente os fluxos de trabalho com *human in the loop* (intervenção humana no processo). Falamos sobre três casos de uso populares, incluindo a **aprovação humana de certas ações** por um agente, como o uso de ferramentas; **debugging** (depuração) com técnicas como *time travel* (viagem no tempo), onde podemos voltar, reproduzir ou criar ramificações de estados anteriores; e também a **edição de estado**.

Agora, vamos avançar para um novo módulo que conecta esses temas com outros relacionados a **controlabilidade** e **fluxos de trabalho com múltiplos agentes**. Vamos reunir tudo isso na criação de um **assistente de pesquisa multiagente**, que integrará todos os conceitos abordados no curso.

Para começar, vamos falar sobre alguns tópicos de controlabilidade, começando com **paralelização**. A ideia principal aqui é o *fan in* e *fan out* — algo que queremos frequentemente aplicar em grafos.

Vamos definir um grafo bem simples: um fluxo linear A → B → C → D. Temos uma única chave de estado e, conforme executamos de A até D, sobrescrevemos o estado a cada nó — isso é simples.

Agora, suponha que queremos rodar os nós B e C **ao mesmo tempo**. Para isso, basta criar uma aresta de A para B e de A para C. Depois, conectamos B e C a D, e então D ao final.

Isso está fazendo um *fan out* (ramificação) a partir de A, rodando B e C **em paralelo** como uma segunda etapa, e depois D como a terceira etapa.

Mas ao rodar isso, um erro acontece: `state key can only receive one value per step`. Ou seja, **B e C estão tentando escrever no mesmo canal ou chave do estado ao mesmo tempo**, o que causa ambiguidade — o grafo não sabe qual valor manter.

Para resolver isso, usamos um **reducer**, que agrega múltiplas atualizações no estado. Definimos um *reducer* simples que **adiciona as atualizações em uma lista**. Agora, B e C podem escrever na mesma chave, pois as atualizações serão apenas anexadas à lista.

Assim, conseguimos executar B e C em paralelo de forma segura. A lição aqui é: **quando executar nós em paralelo que escrevem na mesma chave do estado, use um reducer capaz de agregar os dados.**

Em outro exemplo, desenhamos um grafo com **duas ramificações**. Uma tem duas etapas (B e B2) e a outra tem uma (C). Embora B e C sejam executados quase ao mesmo tempo, o nó D só será executado **quando ambas as ramificações forem concluídas** — isso é importante para o controle do fluxo paralelo.

Depois, mostramos que a **ordem das atualizações no estado** dentro de uma etapa paralela é decidida pelo Landgraf, e você **não tem controle direto** sobre isso — a menos que use um reducer personalizado que, por exemplo, **ordene os valores** após cada atualização.

Então, juntamos tudo isso em um exemplo prático:

- Definimos um **estado com três chaves**: `question`, `answer` e `context` (uma lista).
- Criamos dois nós de busca: um para **busca na web (Tavoli)** e outro para **Wikipedia**.
- Ambos recebem a `question` como entrada e escrevem os resultados em `context`.
- Depois, um terceiro nó usa a `context` e a `question` para gerar uma resposta com um **LLM (modelo de linguagem)**, escrevendo a resposta na chave `answer`.

Executamos isso com uma pergunta sobre os lucros da NVIDIA em 2024 e vimos os resultados.

Esse é um **caso prático útil de paralelização**, onde buscamos dados em múltiplas fontes, agregamos no `context`, e usamos essa base para gerar uma resposta final.

O grafo pode ser executado via API do Landgraf usando o SDK, com um *thread* (sessão de execução), e podemos acompanhar a resposta final ou os estados intermediários no **Studio**, uma interface visual do Landgraf.

---

### **Explicação dos termos técnicos:**

- **Human in the loop**: técnica onde uma pessoa participa ativamente de um processo automatizado, ajudando a tomar decisões, revisar ou corrigir resultados gerados por IA.
- **Time travel**: recurso que permite voltar a estados anteriores do sistema, útil para depuração ou simulação de caminhos alternativos.
- **State (estado)**: dados mantidos e atualizados pelos nós do grafo durante a execução. Pode conter perguntas, respostas, contexto, etc.
- **Reducer**: função que agrega várias atualizações para a mesma chave do estado. Ex: ao invés de sobrescrever, adiciona os valores a uma lista ou escolhe o valor mais relevante.
- **Fan in / Fan out**: conceitos que representam a divisão (*fan out*) e a junção (*fan in*) de caminhos em um grafo de execução.
- **Parallelization (paralelização)**: execução simultânea de múltiplas etapas (nós) do grafo para ganhar eficiência.
- **LangGraph / Landgraf**: framework para criar fluxos de execução baseados em grafos, geralmente usado com LLMs e agentes autônomos.
- **LLM (Large Language Model)**: modelo de linguagem de grande porte, como GPT ou Gemini, usado para gerar texto ou responder perguntas com base em contexto.
- **Context**: informações de apoio usadas para gerar a resposta — no caso, resultados de pesquisas paralelas.
- **Studio**: ferramenta visual para inspecionar, depurar e interagir com fluxos no LangGraph/Landgraf.
- **Tavoli**: ferramenta de busca externa, usada aqui como fonte de dados.

Se quiser, posso montar um **diagrama C4 ou de fluxo** para ilustrar esse grafo multiagente com paralelização e reducer, ou até gerar o código correspondente. Deseja seguir por essa linha?