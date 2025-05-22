## Lição 3 - Editando o estado do gráfico
### 🧠 Contexto: Edição de Estado no Human-in-the-Loop

Até agora, já vimos como usar **breakpoints** para:

- Aprovar ações sensíveis (como chamadas de ferramentas).
- Depurar fluxos.

Mas agora, vamos explorar um **terceiro caso de uso**:

> ✅ Editar ou modificar o estado do grafo enquanto ele está pausado.
> 

---

### 🛠️ Passo a passo: Edição do estado com `graph.updateState`

Anteriormente, introduzimos pontos de interrupção.

Usamos esses pontos para interromper o gráfico e aguardar a aprovação do usuário antes de executar o próximo nó.

Mas os pontos de interrupção também são [oportunidades para modificar o estado do gráfico](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/).

Vamos configurar nosso agente com um ponto de interrupção antes do nó `assistant`.

1. Redefinimos nosso agente para fazer operações simples (ex: multiplicar).
2. Adicionamos um **`interrupt_before="assistant"`**, ou seja, paramos **antes** do nó que executa o modelo LLM.
3. Criamos uma nova `thread`, enviando a entrada `"multiplicar 2 por 3"`.

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
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
```

```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)

# Show
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

![image.png](attachment:c920048d-6c0a-4115-a534-631b6a359bce:image.png)

---

### ✋ Paramos no breakpoint

- O grafo **pausa** antes de o assistente (LLM) rodar.
- O estado nesse momento contém **apenas a mensagem do usuário**.

Podemos ver que o gráfico é interrompido antes que o modelo de bate-papo responda.

```python
# Input
initial_input = {"messages": "Multiply 2 and 3"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```python
================================[1m Human Message [0m=================================

Multiply 2 and 3
```

```python
state = graph.get_state(thread)
state
```

```python
StateSnapshot(values={'messages': [HumanMessage(content='Multiply 2 and 3', id='e7edcaba-bfed-4113-a85b-25cc39d6b5a7')]}, next=('assistant',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a412-5b2d-601a-8000-4af760ea1d0d'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}}, created_at='2024-09-03T22:09:10.966883+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef6a412-5b28-6ace-bfff-55d7a2c719ae'}}, tasks=(PregelTask(id='dbee122a-db69-51a7-b05b-a21fab160696', name='assistant', error=None, interrupts=(), state=None),))

```

---

### ✍️ Editando o estado

Agora, podemos aplicar diretamente uma atualização de estado.

Lembre-se de que as atualizações na chave `messages` usarão o redutor `add_messages`:

- Se quisermos sobrescrever a mensagem existente, podemos fornecer o `id` da mensagem.
- Se quisermos simplesmente adicionar algo à nossa lista de mensagens, podemos passar uma mensagem sem um `id` especificado, como mostrado abaixo.

Chamamos:

```python
graph.update_state(
    thread,
    {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
)

```

➡️ O estado agora contém duas mensagens humanas:

- "multiplicar 2 por 3"
- "na verdade, multiplique 3 por 3"

🧠 O campo `messages` usa o **`addMessagesReducer`**, que **acrescenta mensagens** (ou sobrescreve se o ID for igual).

Vamos dar uma olhada.

Chamamos `update_state` com uma nova mensagem.

O redutor `add_messages` a anexa à nossa chave de estado, `messages`.

```python
new_state = graph.get_state(thread).values
for m in new_state['messages']:
    m.pretty_print()
```

```python
================================[1m Human Message [0m=================================

Multiply 2 and 3
================================[1m Human Message [0m=================================

No, actually multiply 3 and 3!
```

---

### ▶️ Retomando a execução

Agora, vamos prosseguir com nosso agente, simplesmente passando `None` e permitindo que ele prossiga a partir do estado atual.

Emitimos o estado atual e então prosseguimos com a execução dos nós restantes.

Chamamos novamente:

```python
graph.stream(None, thrfor event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()ead_id=...)

```

```python
================================[1m Human Message [0m=================================

No, actually multiply 3 and 3!
==================================[1m Ai Message [0m==================================
Tool Calls:
  multiply (call_Mbu8MfA0krQh8rkZZALYiQMk)
 Call ID: call_Mbu8MfA0krQh8rkZZALYiQMk
  Args:
    a: 3
    b: 3
=================================[1m Tool Message [0m=================================
Name: multiply

9
```

O grafo:

1. Usa a mensagem mais recente ("3 por 3")
2. Executa o nó da ferramenta (`tool call`)
3. Volta ao assistente com a resposta final: **"O resultado de 3 x 3 é 9."**

Agora, estamos de volta ao `assistente`, que contém nosso `ponto de interrupção`.

Podemos passar `None` novamente para prosseguir.

```python
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

---

### 🖥️ Como fazer isso no **LangGraph Studio**

1. Rodamos o grafo com `"multiplicar 2 por 3"`.
2. Definimos um breakpoint antes do nó `assistant`.
3. No Studio, **editamos manualmente o estado**, alterando a entrada para `"multiplicar 3 por 3"`.
4. **“Forkamos” a thread** para não perder o histórico original.
5. Continuamos o grafo até o final, com o resultado esperado.

---

### 🌐 Como editar estado via **LangGraph API**

A API por trás do Studio é acessível via **LangGraph SDK**, o que permite:

- Obter o estado atual com `get_state()`
- Editar o estado com `update_state()`

💡 **Truque legal:**

Se você **mantiver o mesmo `message_id`** e alterar apenas o conteúdo da mensagem, o reducer sobrescreve a mensagem original.

---

### 🧪 Exemplo:

```python
# Obter a última mensagem
msg = state["messages"][-1]
msg["content"] = "na verdade, multiplique 3 por 3"

# Atualizar o estado
graph.update_state(thread_id, {"messages": [msg]})

```

Resultado: a mensagem anterior é sobrescrita.

---

### 🧩 Injetando feedback humano com um "nó fictício" (dummy node)

Agora, ao invés de editar o estado "por fora", podemos **inserir um nó explícito no grafo** chamado, por exemplo, `"human_feedback"`.

1. Criamos esse nó antes do `assistant`.
2. Pausamos no breakpoint (`interrupt_before="human_feedback"`).
3. O humano fornece feedback (ex: corrigir a entrada).
4. Chamamos `update_state(..., as_node="human_feedback")`.

✅ Isso simula como se **aquele nó estivesse executando a modificação** no estado.

---

### 🔁 Fluxo completo

1. Usuário envia `"multiplique 2 por 3"`.
2. Grafo pausa no nó `"human_feedback"`.
3. Humano diz `"não, na verdade multiplique 3 por 3"`.
4. Executamos `update_state()` com esse novo conteúdo.
5. Grafo continua.
6. Ferramenta executa o cálculo → retorna `9`.
7. Assistente finaliza a resposta: **"O resultado de 3 x 3 é 9."**

---

### ✅ Conclusão

Você pode:

| Ação | Comando ou função usada |
| --- | --- |
| Parar o grafo | `interrupt_before="node"` |
| Obter o estado atual | `graph.get_state(thread_id)` |
| Editar ou sobrescrever mensagem | `graph.update_state(thread_id, new_state)` |
| Continuar execução do grafo | `graph.stream(None, thread_id)` |
| Inserir feedback em ponto específico | `update_state(..., as_node="human_feedback")` |

---

Se quiser, posso criar um exemplo **completo em código Python**, com um grafo contendo `assistant`, `tool`, e `human_feedback`, ou gerar uma interface simples em Flask/Gradio para testar isso interativamente. Deseja?