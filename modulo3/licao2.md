# Lição 2 - Breakpoints

### 🧠 **Contexto**

O **streaming** permite que o LangGraph emita o estado do grafo a cada passo da execução. Isso cria a base para implementar o conceito de **Human-in-the-Loop** — ou seja, incluir um ser humano no controle de partes críticas do fluxo.

Para `human-in-the-loop`, frequentemente queremos ver as saídas do nosso gráfico enquanto ele está em execução.

Estabelecemos as bases para isso com streaming.

## Objetivos

Agora, vamos falar sobre as motivações para `human-in-the-loop`:

(1) `Aprovação` - Podemos interromper nosso agente, exibir o estado para um usuário e permitir que ele aceite uma ação.

(2) `Depuração` - Podemos retroceder o gráfico para reproduzir ou evitar problemas.

(3) `Edição` - Você pode modificar o estado.

O LangGraph oferece várias maneiras de obter ou atualizar o estado do agente para oferecer suporte a vários fluxos de trabalho `human-in-the-loop`.

Primeiro, apresentaremos [pontos de interrupção](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/#simple-usage), que fornecem uma maneira simples de interromper o gráfico em etapas específicas.

Mostraremos como isso permite a `aprovação` do usuário.

---

### ✅ Três principais **casos de uso** para Human-in-the-Loop:

1. **Aprovação de ações sensíveis**
    
    Exemplo: se o agente vai usar uma *ferramenta* que escreve em um banco de dados ou sistema externo, o humano pode precisar aprovar antes.
    
2. **Depuração e reexecução (debugging)**
    
    Permite **pausar o grafo**, inspecionar o estado e **reexecutar** a partir de qualquer ponto, útil para diagnosticar e evitar erros.
    
3. **Edição direta do estado**
    
    O humano pode modificar ou adicionar informações no estado do agente antes de continuar a execução.
    

---

### 🛑 **Breakpoints: interrompendo a execução do grafo**

Vamos reconsiderar o agente simples com o qual trabalhamos no Módulo 1.

Vamos supor que estamos preocupados com o uso da ferramenta: queremos aprovar o agente para usar qualquer uma de suas ferramentas.

Tudo o que precisamos fazer é simplesmente compilar o grafo com `interrupt_before=["tools"]`, onde `tools` é o nosso nó de ferramentas.

Isso significa que a execução será interrompida antes do nó `tools`, que executa a chamada da ferramenta.

LangGraph oferece **breakpoints** que interrompem a execução do grafo em pontos específicos. Isso é feito com:

- `interrupt_before`: pausa antes de um nó específico
- `interrupt_after`: pausa após um nó

---

### 🧪 Exemplo com ferramenta (tool node)

1. Suponha que temos um **nó de ferramenta**, que executa uma ação automatizada.
2. Queremos aprovar qualquer ação feita com ferramentas.
3. Definimos o grafo com `interrupt_before="tools"`.

📍 Agora, o grafo:

- Roda até o nó da ferramenta;
- **Pausa a execução**;
- Permite que o humano aprove ou rejeite a continuação.

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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)

# Show
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

![image.png](attachment:fd692428-4433-4c53-b844-181269b983b6:image.png)

```python
# Input
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```python
================================[1m Human Message [0m=================================

Multiply 2 and 3
==================================[1m Ai Message [0m==================================
Tool Calls:
  multiply (call_oFkGpnO8CuwW9A1rk49nqBpY)
 Call ID: call_oFkGpnO8CuwW9A1rk49nqBpY
  Args:
    a: 2
    b: 3
```

Podemos obter o estado e observar o próximo nó a ser chamado.

Esta é uma boa maneira de ver que o gráfico foi interrompido.

```python
state = graph.get_state(thread)
state.next
```

---

### 👁️ Visualizando o estado parado

Usamos `graph.get_state(thread_id)` para obter o **checkpoint atual**.

Podemos também ver:

- Qual é o próximo nó (`next_node`)
- O estado acumulado da conversa
- O histórico completo com `graph.get_state_history()`

---

### ▶️ **Continuando a partir do breakpoint**

Agora, vamos apresentar um truque interessante.

Quando invocamos o grafo com `None`, ele simplesmente continua a partir do último ponto de verificação de estado!

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbae7985b747dfed67775d_breakpoints1.png)

Para maior clareza, o LangGraph reemitirá o estado atual, que contém a `AIMessage` com a chamada da ferramenta.

E então ele prosseguirá com a execução dos seguintes passos no grafo, que começam com o nó da ferramenta.

Vemos que o nó da ferramenta é executado com essa chamada da ferramenta e é passado de volta ao modelo de bate-papo para nossa resposta final.

Depois que o humano aprova, usamos:

```python
graph.stream(None, thread_id=...)

```

Esse comando:

- **Reemite o estado atual**
- **Continua a execução a partir do ponto de parada**

```python
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```python
==================================[1m Ai Message [0m==================================
Tool Calls:
  multiply (call_oFkGpnO8CuwW9A1rk49nqBpY)
 Call ID: call_oFkGpnO8CuwW9A1rk49nqBpY
  Args:
    a: 2
    b: 3
=================================[1m Tool Message [0m=================================
Name: multiply

6
==================================[1m Ai Message [0m==================================

The result of multiplying 2 and 3 is 6.
```

Agora, vamos unir tudo isso com uma etapa específica de aprovação do usuário que aceita a entrada do usuário.

```python
# Input
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

# Thread
thread = {"configurable": {"thread_id": "2"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

# Get user feedback
user_approval = input("Do you want to call the tool? (yes/no): ")

# Check approval
if user_approval.lower() == "yes":
    
    # If approved, continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()
        
else:
    print("Operation cancelled by user.")
```

```python
================================[1m Human Message [0m=================================

Multiply 2 and 3
==================================[1m Ai Message [0m==================================
Tool Calls:
  multiply (call_tpHvTmsHSjSpYnymzdx553SU)
 Call ID: call_tpHvTmsHSjSpYnymzdx553SU
  Args:
    a: 2
    b: 3
==================================[1m Ai Message [0m==================================
Tool Calls:
  multiply (call_tpHvTmsHSjSpYnymzdx553SU)
 Call ID: call_tpHvTmsHSjSpYnymzdx553SU
  Args:
    a: 2
    b: 3
=================================[1m Tool Message [0m=================================
Name: multiply

6
==================================[1m Ai Message [0m==================================

The result of multiplying 2 and 3 is 6.
```

---

### ✅ Exemplo com input de aprovação:

1. O humano envia `"yes"` como input
2. O sistema chama novamente o `stream(None)` com o mesmo `thread_id`
3. O grafo continua do ponto onde parou (ferramenta) e segue até o fim

---

### 🌐 Usando a **API do LangGraph**

Além de definir o breakpoint no código, a **API permite passar interrupções dinamicamente**:

```python
client.stream(interrupt_before="tools", ...)

```

💡 Isso é útil porque você não precisa recompilar o grafo toda vez. Basta especificar o nó onde quer parar.

---

### 📦 No Studio (interface gráfica)

1. O grafo roda localmente no Studio
2. Podemos aplicar `interrupt_before` via código (como antes) **ou**
3. **Via API**: basta passar o argumento no momento da chamada

---

### 🧰 Resumo técnico

| Recurso | O que faz |
| --- | --- |
| `interrupt_before="node"` | Interrompe a execução **antes** do nó especificado |
| `graph.get_state()` | Recupera o **checkpoint atual** do grafo |
| `graph.stream(None)` | **Continua a execução** a partir do ponto em que parou |
| API com `interrupt_before` | Permite aplicar breakpoints dinamicamente, sem alterar o código do grafo |
| Studio | Interface visual com suporte a breakpoints e aprovação manual |

---

### ✅ Conclusão: por que isso é útil?

- Torna o LangGraph ideal para **fluxos críticos**, como automação com validação humana.
- Permite **debug interativo**.
- Cria **assistentes híbridos**, com decisões automatizadas e intervenções humanas.

---

Se quiser, posso gerar um exemplo completo de **fluxo com breakpoint e aprovação humana**, usando `LangGraph + API`, ou integrar isso com uma interface web. Deseja seguir com isso?