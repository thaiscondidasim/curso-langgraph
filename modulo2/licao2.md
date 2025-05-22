# Lição 2 - State Reducer

Exploramos algumas formas diferentes de definir o *schema* de estado no LangGraph, incluindo dicionários do tipo `dict`, objetos do *Pydantic* ou *data classes*. Agora vamos nos aprofundar nos **reducers**, que especificam como as atualizações de estado são feitas em **chaves específicas** ou **canais** do seu esquema. Vamos começar usando um `type dict` como nosso esquema de estado.

```python
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    foo: int

def node_1(state):
    print("---Node 1---")
    return {"foo": state['foo'] + 1}

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:b90faaf1-ca0e-44a6-b2ab-16db6af0aca9:image.png)

Vamos criar um grafo simples com um único nó. Beleza, está feito. Vamos invocar esse grafo. Podemos ver que ele retorna `foo` com valor `2`, tendo recebido `foo = 1`. Isso acontece porque, no nosso nó único, basicamente incrementamos o valor de `foo` em 1. Um exemplo super simples.

```python
graph.invoke({"foo" : 1})
```

Agora vamos refletir um pouco mais sobre isso. Observamos a entrada e a saída, e vemos que no nó tudo que fazemos é sobrescrever o valor de `foo` com `foo + 1`.

Isso traz um ponto interessante: **por padrão**, o LangGraph **não sabe qual é a forma preferida de atualizar o estado**, então ele simplesmente sobrescreve o valor. Ou seja, o estado final do grafo fica como `2` porque substituímos o `1` pela nova atualização.

⚙️ *LangGraph*: uma ferramenta para criação de fluxos (grafos) com estados mutáveis, útil para aplicações com múltiplos passos e controle de estado.

---

### Exemplo com *branching* (ramificação):

Agora vamos ver um caso simples de ramificação. Começamos com um nó inicial (`start`), que vai para o `node2` e também para o `node3`. Ambos vão para o nó final (`end`).

```python
class State(TypedDict):
    foo: int

def node_1(state):
    print("---Node 1---")
    return {"foo": state['foo'] + 1}

def node_2(state):
    print("---Node 2---")
    return {"foo": state['foo'] + 1}

def node_3(state):
    print("---Node 3---")
    return {"foo": state['foo'] + 1}

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:869bb9bb-9bb9-4d0a-b59d-808ef878f1b6:image.png)

Vamos invocar isso, e vou fazer algo interessante: tentar capturar erros de atualização inválida, caso ocorram. E veja só, ocorre o erro:

```python
from langgraph.errors import InvalidUpdateError
try:
    graph.invoke({"foo" : 1})
except InvalidUpdateError as e:
    print(f"InvalidUpdateError occurred: {e}")

```

**"Erro de atualização inválida: a chave `foo` só pode receber um valor por passo. Use uma chave anotada para lidar com múltiplos valores."**

🤔 Por quê isso acontece?

Porque os nós `2` e `3` são executados **em paralelo**, dentro do **mesmo passo**. Ambos tentam atualizar a mesma chave de estado (`foo`). Como resultado, o LangGraph **não sabe qual valor manter**, pois os dois estão sobrescrevendo ao mesmo tempo — e isso gera ambiguidade.

---

### O papel dos **reducers**:

É aí que entram os **reducers**: eles nos permitem especificar **como combinar ou atualizar** os valores de estado quando múltiplas atualizações acontecem no mesmo passo.

[Redutores](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) nos fornecem uma maneira geral de resolver esse problema.

Tudo que precisamos fazer é:

1. Definir nosso schema de estado normalmente.
2. Fornecer um tipo anotado para a chave que inclui uma **função de reducer**.

Exemplo: usamos a função `add` do módulo `operator` do Python, que — quando aplicada a listas — faz a concatenação (junta os valores em uma única lista). Assim, nossa chave `foo` vira uma **lista de eventos**, à qual adicionamos valores conforme o grafo avança.

Podemos usar o tipo `Annotated` para especificar uma função redutora.

Por exemplo, neste caso vamos anexar o valor retornado de cada nó em vez de sobrescrevê-los.

Só precisamos de um redutor que possa executar isso: `operator.add` é uma função do módulo embutido `operator` do Python.

Quando `operator.add` é aplicado a listas, ele realiza a concatenação de listas.

```python
from operator import add
from typing import Annotated

class State(TypedDict):
    foo: Annotated[list[int], add]

def node_1(state):
    print("---Node 1---")
    return {"foo": [state['foo'][0] + 1]}

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:1f061456-f7a7-49f2-8a1a-cd1b7ace088b:image.png)

```python
graph.invoke({"foo" : [1]})
```

### Exemplo com lista:

Vamos ver isso com um único nó. A chave `foo` é uma lista. Cada vez que passamos por um nó, incrementamos o valor e adicionamos à lista.

```python
def node_1(state):
    print("---Node 1---")
    return {"foo": [state['foo'][-1] + 1]}

def node_2(state):
    print("---Node 2---")
    return {"foo": [state['foo'][-1] + 1]}

def node_3(state):
    print("---Node 3---")
    return {"foo": [state['foo'][-1] + 1]}

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:d1aca9f5-0d1b-46aa-bcd9-7b5d641f16b3:image.png)

Resultado: se começamos com `[1]`, depois de passar pelo nó temos `[1, 2]` — **não sobrescrevemos**, apenas adicionamos.

Agora vamos criar um grafo com três nós, com ramificações como antes. Executamos... e tudo funciona! Por quê? Porque cada nó adiciona um valor à lista `foo`, em vez de sobrescrever. Como temos um **reducer configurado**, ele sabe como combinar os dados.

🧠 *Reducer*: função que define **como combinar múltiplas atualizações de estado** quando elas ocorrem simultaneamente. Exemplo: somar, concatenar listas, etc.

---

Lidando com casos especiais: `None`

Agora, e se passarmos `None` para `foo`? Acontece um erro, porque **não dá pra concatenar `None` com uma lista**. Isso nos leva a outro ponto: às vezes é necessário criar **reducers personalizados**.

```python
try:
    graph.invoke({"foo" : None})
except TypeError as e:
    print(f"TypeError occurred: {e}")
```

Então, podemos definir uma função própria que saiba lidar com listas e valores nulos (`None`) de forma segura, e usá-la como nosso reducer.

Resultado: com o novo reducer, mesmo passando `None`, tudo funciona. O sistema entende como lidar com isso.

---

### Redutores Personalizados

Para lidar com casos como esse, [também podemos definir redutores personalizados](https://langchain-ai.github.io/langgraph/how-tos/subgraph/#custom-reducer-functions-to-manage-state).

Por exemplo, vamos definir uma lógica de redutor personalizado para combinar listas e tratar casos onde uma ou ambas as entradas podem ser `None`.

```python
def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class DefaultState(TypedDict):
    foo: Annotated[list[int], add]

class CustomReducerState(TypedDict):
    foo: Annotated[list[int], reduce_list]
```

Em `node_1`, acrescentamos o valor 2.

```python
def node_1(state):
    print("---Node 1---")
    return {"foo": [2]}

# Build graph
builder = StateGraph(DefaultState)
builder.add_node("node_1", node_1)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

try:
    print(graph.invoke({"foo" : None}))
except TypeError as e:
    print(f"TypeError occurred: {e}")
```

![image.png](attachment:bbd74dba-1881-4793-b1b5-3d5f98465a73:image.png)

```python
TypeError occurred: can only concatenate list (not "NoneType") to list

```

Agora, tente com nosso redutor personalizado. Podemos ver que nenhum erro é gerado.

```python
# Build graph
builder = StateGraph(CustomReducerState)
builder.add_node("node_1", node_1)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

try:
    print(graph.invoke({"foo" : None}))
except TypeError as e:
    print(f"TypeError occurred: {e}")
```

![image.png](attachment:ce9746c3-540f-4cc7-8fd9-d6877c0a387f:image.png)

---

### Trabalhando com mensagens (`messages`):

No módulo 1, vimos o uso de `addMessages` para lidar com mensagens no estado. Existe uma chave especial chamada `messages` com um reducer embutido chamado `addMessages`.

Você pode:

- Definir um estado com essa chave manualmente.
- Ou usar o estado padrão `messages`, que já vem com isso pronto.

Os dois funcionam do mesmo jeito.

🗨️ *addMessages reducer*: quando você usa esse reducer, ele **adiciona uma nova mensagem** à lista. Se você passar um `id` de mensagem que já existe, ele **substitui o conteúdo daquela mensagem**.

Exemplo:

- Mensagem com ID `1` já existe.
- Você passa uma nova mensagem com o mesmo ID: ela sobrescreve a anterior.

Usaremos a classe `MessagesState` via `from langgraph.graph import MessagesState` por brevidade.

```python
from typing import Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Define a custom TypedDict that includes a list of messages with add_messages reducer
class CustomMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str
    added_key_2: str
    # etc

# Use MessagesState, which includes the messages key with add_messages reducer
class ExtendedMessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    added_key_1: str
    added_key_2: str
    # etc
```

Vamos falar um pouco mais sobre o uso do redutor `add_messages`.

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                   ]

# New message to add
new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# Test
add_messages(initial_messages , new_message)
```

```python
<IPython.core.display.Image object>
---Node 1---
{'foo': 2}
<IPython.core.display.Image object>
---Node 1---
---Node 2---
---Node 3---
InvalidUpdateError occurred: At key 'foo': Can receive only one value per step. Use an Annotated key to handle multiple values.
<IPython.core.display.Image object>
---Node 1---
{'foo': [1, 2]}
<IPython.core.display.Image object>
---Node 1---
---Node 2---
---Node 3---
{'foo': [1, 2, 3, 3]}
TypeError occurred: can only concatenate list (not "NoneType") to list
<IPython.core.display.Image object>
TypeError occurred: can only concatenate list (not "NoneType") to list
<IPython.core.display.Image object>
---Node 1---
{'foo': [2]}
[AIMessage(content='Hello! How can I assist you?', name='Model', id='f470d868-cf1b-45b2-ae16-48154cd55c12'),
 HumanMessage(content="I'm looking for information on marine biology.", name='Lance', id='a07a88c5-cb2a-4cbd-9485-5edb9d658366'),
 AIMessage(content='Sure, I can help with that. What specifically are you interested in?', name='Model', id='7938e615-86c2-4cbb-944b-c9b2342dee68')]
[AIMessage(content='Hello! How can I assist you?', name='Model', id='1'),
 HumanMessage(content="I'm looking for information on whales, specifically", name='Lance', id='2')]
[RemoveMessage(content='', id='1'), RemoveMessage(content='', id='2')]
/var/folders/l9/bpjxdmfx7lvd1fbdjn38y5dh0000gn/T/ipykernel_17703/3097054180.py:10: LangChainBetaWarning: The class `RemoveMessage` is in beta. It is actively being worked on, so the API may change.
  delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
[AIMessage(content='So you said you were researching ocean mammals?', name='Bot', id='3'),
 HumanMessage(content='Yes, I know about whales. But what others should I learn about?', name='Lance', id='4')]
```

Podemos ver que `add_messages` nos permite anexar mensagens à chave `messages` em nosso estado.

### Reescrevendo

Vamos mostrar alguns truques úteis ao trabalhar com o redutor `add_messages`.

Se passarmos uma mensagem com o mesmo ID de uma existente em nossa lista `messages`, ela será sobrescrita!

```python
# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance", id="2")
                   ]

# New message to add
new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Lance", id="2")

# Test
add_messages(initial_messages , new_message)
```

```python
[AIMessage(content='Hello! How can I assist you?', name='Model', id='1'),
 HumanMessage(content="I'm looking for information on whales, specifically", name='Lance', id='2')]
```

---

### Remover mensagens

Também é possível **remover mensagens** pelo ID usando o `removeMessages`, importado do módulo `langchain_core`.

Exemplo:

- Você tem uma lista de mensagens.
- Cria objetos de remoção (`RemoveMessage`) para os IDs que deseja excluir.
- Passa essas instruções junto com o `addMessages`.

Resultado: ele remove as mensagens com os IDs especificados. Uma forma muito útil de **limpar o histórico de mensagens**.

```python
from langchain_core.messages import RemoveMessage

# Message list
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Isolate messages to delete
delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
print(delete_messages)
```

```python
[RemoveMessage(content='', id='1'), RemoveMessage(content='', id='2')]
/var/folders/l9/bpjxdmfx7lvd1fbdjn38y5dh0000gn/T/ipykernel_17703/3097054180.py:10: LangChainBetaWarning: The class `RemoveMessage` is in beta. It is actively being worked on, so the API may change.
  delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
```

```python
add_messages(messages , delete_messages)
```

```python
[AIMessage(content='So you said you were researching ocean mammals?', name='Bot', id='3'),
 HumanMessage(content='Yes, I know about whales. But what others should I learn about?', name='Lance', id='4')]
```

Podemos ver que as mensagens 1 e 2, conforme observado em `delete messages`, são removidas pelo redutor.
Veremos isso em prática mais adiante.