**ReAct** é uma arquitetura de agente genérica muito popular, que realiza três etapas principais:

1. **Agir (Act):**
    
    Permite que o agente chame ferramentas específicas.
    
2. **Observar (Observe):**
    
    Passa a saída da ferramenta de volta para o modelo LLM.
    
3. **Raciocinar (Reason):**
    
    O modelo interpreta essa saída, decide o que fazer em seguida — seja chamar outra ferramenta ou encerrar com uma resposta em linguagem natural.
    

Essa arquitetura é bastante clássica e amplamente utilizada.

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbab7453080e6802cd1703_agent-memory1.png)

---

### 🧠 Agora vamos estender isso com um conceito simples: **memória**

Assim como antes, definimos nossa chave da API e criamos um agente com três ferramentas:

`multiply`, `add` e `divide`.

Ligamos essas ferramentas ao modelo, passamos uma mensagem de sistema informando que ele é um assistente de aritmética. Tudo igual.

```python
from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
   
   
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```

![image.png](attachment:8b21af3b-4ede-4345-aed2-8c3119bbf807:image.png)

### 🧩 Arquitetura

A estrutura do grafo é:

- O nó `assistant` recebe a entrada do usuário.
- Se for uma chamada de ferramenta, vai para o nó de ferramentas.
- Depois, volta ao `assistant`, e repete enquanto houver chamadas de ferramentas.
- Quando não houver mais, o grafo termina.

---

### ▶️ Executando sem memória

Rodamos a entrada: `"some 3 e 4"` → resultado `7`.

Agora, em **uma nova execução**, passamos `"multiplique isso por 2"`.

Problema:

O agente **não sabe o que é "isso"**.

Ele entende como `2 * 2`, e não `7 * 2`.

Por quê? Porque **o estado não é persistido** entre execuções do grafo.

Cada execução começa com um estado novo e **não compartilha memória com execuções anteriores**.

---

### 💾 Solução: usando **memória com checkpointer**

O LangGraph usa **checkpointers** para salvar o estado do grafo após cada etapa.

Isso nos dá **memória persistente** entre execuções.

A forma mais simples de fazer isso é usando o `MemorySaver`, um armazenamento de chave-valor em memória.

Nós não mantemos memória do nosso chat inicial!

Isso ocorre porque [o estado é transitório](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220) durante uma única execução do grafo.

Claro, isso limita nossa capacidade de ter conversas com múltiplas interações e interrupções.

Podemos usar [persistência](https://langchain-ai.github.io/langgraph/how-tos/persistence/) para resolver isso!

LangGraph pode usar um checkpointer para salvar automaticamente o estado do grafo após cada etapa.

Essa camada de persistência integrada nos dá memória, permitindo que LangGraph retome a partir da última atualização de estado.

Um dos checkpointers mais fáceis de usar é o `MemorySaver`, um armazenamento chave-valor em memória para o estado do Grafo.

Tudo o que precisamos fazer é compilar o grafo com um checkpointer, e nosso grafo terá memória!

Tudo o que você precisa fazer é:

1. Importar o `MemorySaver`;
2. Compilar o grafo com `checkpointer=MemorySaver()`.

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

Quando usamos memória, precisamos especificar um `thread_id`.

Esse `thread_id` armazenará nossa coleção de estados do grafo.

Aqui está uma ilustração:

- O checkpointer grava o estado em cada etapa do grafo
- Esses checkpoints são salvos em uma thread
- Podemos acessar essa thread no futuro usando o `thread_id`

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e0e9f526b41a4ed9e2d28b_agent-memory2.png)

---

### 📚 Como funciona o checkpointer?

Imagine um grafo com dois nós.

Cada etapa gera um **checkpoint** contendo:

- O estado atual,
- O próximo nó a ser executado,
- Metadados,
- E um ID de checkpoint.

Esses checkpoints são **agrupados em uma thread** — um encadeamento que representa uma sequência completa de execução.

Quando você invoca seu grafo passando um `thread_id`, o LangGraph carrega todos os checkpoints dessa thread e continua a partir do último estado.

```python
# Specify a thread
config = {"configurable": {"thread_id": "1"}}

# Specify an input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = react_graph_memory.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()
```

Se passarmos o mesmo `thread_id`, podemos continuar a partir do checkpoint de estado registrado anteriormente!

Nesse caso, a conversa acima está capturada na thread.

A `HumanMessage` que passamos (`"Multiplique isso por 2."`) é anexada à conversa acima.

Portanto, o modelo agora sabe que `isso` se refere a `A soma de 3 e 4 é 7.`.

```python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()
```

### ▶️ Executando com memória

1. Rodamos novamente: `"some 3 e 4"` → resultado `7`.
2. Guardamos o `thread_id` gerado.
3. Executamos: `"multiplique isso por 2"` passando o mesmo `thread_id`.

Agora, o agente **sabe** que “isso” é `7`.

Executa `7 * 2` → resultado: **14**.

A memória permite **preservar o contexto de execuções anteriores**, e o grafo se comporta como se fosse uma conversa contínua.

---

### 🧪 Interagindo com isso no LangGraph Studio

Abrimos o projeto no Studio:

- Abrimos `agent.py`, o mesmo código que usamos no notebook.
- Uma observação: **no Studio, não é necessário definir o checkpointer manualmente**.
    
    O LangGraph API já faz isso automaticamente, usando um banco Postgres por trás.
    

No `langgraph.json`, vemos o agente definido como padrão.

---

### 🧾 Testando no Studio

1. Abrimos o projeto no Studio;
2. Vamos para a aba do agente;
3. Criamos uma nova thread;
4. Entrada: `"multiplique dois por três"`.

O que acontece:

- O modelo interpreta o input em linguagem natural;
- Transforma em uma chamada de ferramenta estruturada:
    - `multiply(2, 3)`
- O nó de ferramenta executa a função e retorna `6`;
- O resultado `6` é passado de volta ao modelo;
- O modelo responde: **“O resultado de multiplicar dois por três é seis.”**

---

### ✅ Conclusão

Este exemplo mostra como:

- Uma modificação simples com checkpointer pode adicionar **memória persistente** ao seu agente;
- Você pode **encadear execuções**, preservando o estado e o histórico da conversa;
- O LangGraph Studio e a LangGraph API **cuidam da persistência automaticamente**, facilitando o desenvolvimento.

Esse conceito de memória será **amplamente usado** nas próximas seções e é uma peça fundamental para criar agentes conversacionais inteligentes e contextuais.

Se quiser, posso seguir com a próxima parte!