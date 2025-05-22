# Lição 3 - Multiplos Schemas

### **Normalmente**, todos os nós de um grafo compartilham um **único schema de estado** (*state schema*). Esse schema contém as **chaves de entrada e saída** do grafo (também chamadas de *canais*).

No entanto, **há casos em que queremos mais controle** sobre isso. Por exemplo:

1. **Nós internos do grafo** podem precisar trocar informações que **não são relevantes para o usuário** final. Essas informações seriam apenas de uso interno, entre os nós — e **não devem aparecer como entrada ou saída** visível do grafo.
2. Podemos querer usar **schemas de entrada e saída diferentes**, dependendo da aplicação. A entrada pode conter apenas uma pergunta do usuário, enquanto a saída pode ter outros valores adicionais gerados internamente, que **não existiam no início da execução**.

---

### 🔐 **Estado privado (private state)**

Primeiro, vamos ver o conceito de [**estado privado**](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/), que é útil para armazenar qualquer coisa **necessária para o funcionamento interno do grafo**, mas **não relevante para a entrada ou saída geral** do grafo.

No exemplo:

- Definimos um **estado geral** (`OverallState`) com a chave `foo`.
- E um **estado privado** (`PrivateState`) com a chave `baz`.

Nosso **nó 1** recebe o estado geral como entrada (com `foo`) e escreve `baz` no estado privado.

Depois, o **nó 2** lê o estado privado (`baz`) e escreve de volta no estado geral (`foo`).

🧠 *Dica técnica*: Quando você usa uma anotação de tipo, como `state: OverallState`, está especificando **qual schema aquele nó espera como entrada** ou **para onde ele escreve como saída**.

```python
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

def node_1(state: OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    return {"foo": state['baz'] + 1}

# Build graph
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:ed145ed8-228c-468c-9a9c-32c7cbd438c0:image.png)

```python
graph.invoke({"foo" : 1})
```

```python
---Node 1---
---Node 2---
```

{'foo': 3}

Quando executamos o grafo, o que vemos?

✅ A chave `foo` aparece na saída — porque faz parte do estado geral.

❌ A chave `baz` **não aparece na saída**, porque está no **estado privado**, que **é usado apenas entre os nós**.

✨ **Conclusão**: o estado privado permite a comunicação interna entre nós, **sem expor esses dados ao usuário** no final da execução do grafo.

---

### 🎯 **Schemas de entrada e saída personalizados**

Agora, vamos deixar isso **mais explícito**: queremos um grafo com [definir esquemas explícitos de entrada e saída para um grafo](https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/?h=input+outp).

Imagine um caso de perguntas e respostas:

- Definimos um estado geral (`OverallState`) com: `question`, `answer`, e `notes`.
- Criamos dois nós:
    - Um **"thinking node"** que processa a pergunta.
    - Um **"answer node"** que gera a resposta.

Executando esse grafo, a saída contém `question`, `answer` e `notes`.

Mas... e se quisermos que:

- O **usuário só forneça a pergunta** (`question`) como entrada;
- E **receba apenas a resposta** (`answer`) como saída?

Podemos fazer isso definindo **schemas personalizados para entrada e saída**, que funcionam como **filtros** aplicados sobre o estado geral.

```python
class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: OverallState):
    return {"answer": "bye", "notes": "... his name is Lance"}

def answer_node(state: OverallState):
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:875ae683-1a62-4b73-83eb-bbeab1bfcdd8:image.png)

Observe que a saída de invoke contém todas as chaves em `OverallState`.

```python
graph.invoke({"question":"hi"})
```

```python
{'question': 'hi', 'answer': 'bye Lance', 'notes': '... his name is Lance'}

```

Agora, vamos usar esquemas específicos de `entrada` e `saída` com nosso grafo.

Aqui, os esquemas de `entrada`/`saída` realizam *filtragem* sobre quais chaves são permitidas na entrada e saída do grafo.

Além disso, podemos usar a dica de tipo `state: InputState` para especificar o esquema de entrada de cada um de nossos nós.

Isso é importante quando o grafo está usando múltiplos esquemas.

Usamos dicas de tipo abaixo para, por exemplo, mostrar que a saída do `answer_node` será filtrada para `OutputState`.

```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his is name is Lance"}

def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState, input=InputState, output=OutputState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

graph.invoke({"question":"hi"})
```

![image.png](attachment:989eb20d-4741-4837-aaea-07fda237bdc8:image.png)

```python
{'question': 'hi', 'answer': 'bye Lance', 'notes': '... his is name is Lance'}
```

---

### 🧪 Como aplicar esses filtros?

1. Criamos um schema para a entrada (ex.: só `question`).
2. Criamos um schema para a saída (ex.: só `answer`).
3. Passamos esses filtros quando construímos o grafo.

🧠 O que acontece:

- O nó “thinking” lê o **estado de entrada**, mas escreve no **estado geral**.
- O nó “answer” lê do estado geral e, no final, o filtro de **saída** é aplicado, retornando **apenas `answer` para o usuário**.

---

### ✅ Resultado final:

Quando executamos o grafo com os filtros aplicados:

- O usuário fornece apenas a `question`.
- Ele recebe apenas a `answer`.
- Informações internas como `notes` permanecem **escondidas**.

---

### 📌 Resumo dos conceitos importantes: