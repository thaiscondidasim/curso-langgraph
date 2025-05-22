Construímos um grafo que usa mensagens como estado e um modelo de chat com ferramentas vinculadas para fazer uma de duas coisas:

1. Retornar uma **chamada de ferramenta** (tool call), ou
2. Retornar uma **resposta em linguagem natural**, dependendo da decisão do modelo de chat com base na entrada.

Ou seja, se a entrada estiver relacionada com alguma ferramenta, ele retorna uma chamada de ferramenta. Caso contrário, apenas responde diretamente.

Podemos pensar nisso como um roteador, onde o modelo de chat direciona entre uma resposta direta ou uma chamada de ferramenta com base na entrada do usuário.

Este é um exemplo simples de um agente, onde o LLM está direcionando o fluxo de controle, seja chamando uma ferramenta ou apenas respondendo diretamente.

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbac6543c3d4df239a4ed1_router1.png)

Vamos estender nosso grafo para trabalhar com qualquer saída!

Para isso, podemos usar duas ideias:

1. Adicionar um nó que chamará nossa ferramenta.
2. Adicionar uma aresta condicional que verificará a saída do modelo de chat e direcionará para o nó de chamada de ferramenta ou simplesmente terminará se nenhuma ferramenta for chamada.

---

### 🧠 Isso já é um tipo simples de agente

Esse modelo LLM está **direcionando o fluxo de controle** da aplicação:

- Ou chamando uma ferramenta,
- Ou apenas respondendo diretamente.

---

### 🔁 Visão geral do roteador

Aqui está uma representação simples (um “cartoon”) de como um roteador típico funciona:

Um LLM escolhe um dos dois caminhos possíveis com base na entrada.

Agora, vamos expandir um pouco o que fizemos antes com **duas novas ideias**:

1. Vamos adicionar um **nó que executa a chamada da ferramenta**.
    
    Se o modelo responder com uma chamada de ferramenta, executamos isso num nó separado.
    
2. Vamos adicionar uma **aresta condicional** que analisa a saída do modelo e toma uma decisão:
    - Se for uma chamada de ferramenta → vai para o nó da ferramenta.
    - Caso contrário → vai direto para o final.

### 🚧 Implementação

Primeiro, garantimos que a chave da API está configurada.

A função usada como ferramenta ainda é a `multiply`.

Usamos o modelo de linguagem com `bind_tools` para vinculá-la.

Agora, usamos **componentes embutidos** do LangGraph:

- Um **nó de ferramenta** (`tool node`) pronto para usar.
    
    Basta passar a função (em forma de lista), e o nó já está pronto para executar nossa função `multiply`.
    
- Uma **aresta condicional pré-pronta** (`tools_condition`).
    
    Essa aresta verifica se a saída do modelo é uma chamada de ferramenta:
    
    - Se sim, roteia para o nó da ferramenta.
    - Se não, vai para o final.

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([multiply])
```

Utilizamos o [`ToolNode` integrado](https://langchain-ai.github.io/langgraph/reference/prebuilt/?h=tools+condition#toolnode) e simplesmente passamos uma lista de nossas ferramentas para inicializá-lo.

Usamos a [condição `tools_condition` integrada](https://langchain-ai.github.io/langgraph/reference/prebuilt/?h=tools+condition#tools_condition) como nossa aresta condicional.

### Explicação adicional (opcional):

- **`ToolNode`**: Um nó pré-construído no LangGraph que automatiza o processamento de chamadas de ferramentas, executando as ferramentas especificadas quando acionado.
- **`tools_condition`**: Uma função que determina dinamicamente se o fluxo deve seguir para o `ToolNode` (se houver chamadas de ferramenta) ou encerrar (se a resposta for direta).

Esses componentes simplificam a construção de agentes inteligentes que alternam entre respostas naturais e ações específicas.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:d3f4d225-3571-41f9-8b9e-39905d286505:image.png)

---

### ▶️ Executando o roteador

Vamos testar agora.

Primeiro, uma entrada **relevante para a ferramenta**: `"multiplique três por quatro"`.

Resultado:

- A mensagem é processada;
- O modelo responde com uma chamada de ferramenta;
- A aresta condicional detecta isso e roteia para o nó da ferramenta;
- A função `multiply` é executada e retorna a resposta.

Agora vamos testar com uma entrada simples, como `"hello world"`:

- O modelo apenas responde com: **"Olá, como posso ajudar?"**
- Nenhuma chamada de ferramenta é feita;
- A execução vai direto para o fim.

```python
from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

---

### 📊 Visualizando no Studio

Vamos ver como isso funciona no **LangGraph Studio**:

- Abrimos o `module-1/studio`;
- Vários scripts Python representam nossos grafos;
- Abrimos o arquivo `router.py`;
- É o mesmo código do notebook, mas agora salvo como script Python.

Abrimos também o arquivo `langgraph.json`, onde vemos que o grafo "router" está definido como padrão.

Abrimos o Studio, selecionamos o projeto, e pronto.

Testes no Studio:

1. Entrada: `"Hi, I'm Lance"`
    
    → Sem chamada de ferramenta → resposta direta.
    
2. Entrada: `"multiplique dois por três"`
    
    → Agora há uma chamada de ferramenta → roteia para o nó da ferramenta → executa → retorna o resultado.
    

O Studio mostra isso de forma visual:

- Nome da ferramenta chamada (`multiply`);
- Argumentos bem formatados;
- Resultado visível;
- Podemos inspecionar cada etapa e navegar entre execuções anteriores.

---

### ✅ Conclusão

Este é um **roteador básico** que usa:

- Um modelo de linguagem com ferramentas vinculadas,
- Uma **aresta condicional embutida** (`tools_condition`),
- E um **nó de ferramenta embutido** (`tool node`).

Tudo isso torna muito fácil **visualizar**, **testar** e **entender** como o controle de fluxo funciona nos seus grafos no LangGraph.

Se quiser, posso seguir com a próxima parte!