# Lição 1 - Schema de estado

No **Módulo 1**, lançamos as bases:

- Construímos um agente com **memória**, capaz de:
    - **Agir** (por exemplo, usando ferramentas);
    - **Observar** os resultados dessas ferramentas;
    - **Raciocinar** sobre o que fazer a seguir com base nessas saídas;
    - **Persistir seu estado**, permitindo conversas longas;
    - E até mesmo **implantar** esse agente na **LangGraph Cloud**.

---

Agora, neste **segundo módulo**, vamos aprofundar ainda mais nos conceitos de **estado** e **memória**.

---

### 📦 Schema de Estado

Quando definimos um `StateGraph` no LangGraph, usamos um [esquema de estado](https://langchain-ai.github.io/langgraph/concepts/low_level/#state).

Esse schema é basicamente a **estrutura e os tipos de dados** que o grafo vai usar.

Todos os nós devem se comunicar com esse esquema.

O LangGraph oferece flexibilidade na forma como você define seu esquema de estado, acomodando vários [tipos](https://docs.python.org/3/library/stdtypes.html#type-objects) do Python e abordagens de validação!

Na prática, temos usado bastante o `TypedDict`, que é:

- Um dicionário cujas **chaves têm dicas de tipo** (type hints);
- Muito flexível e recomendado;
- Mesmo que esses tipos **não sejam verificados em tempo de execução**.

Porém, observe que estas são apenas dicas de tipo.

Elas podem ser usadas por verificadores de tipo estáticos (como [mypy](https://github.com/python/mypy)) ou IDEs para detectar possíveis erros relacionados a tipos antes da execução do código.

Mas elas não são impostas em tempo de execução!

```python
from typing_extensions import TypedDict

class TypedDictState(TypedDict):
    foo: str
    bar: str
```

---

### Exemplo com `TypedDict`

Vamos criar um novo schema com:

- `name`: uma string,
- `mood`: um literal que pode ser `"happy"` ou `"sad"`.

Depois, usamos esse schema como entrada para um grafo com 3 nós:

- Começa no nó 1,
- Depois, com base no **humor (mood)**, vai para o nó 2 ou 3 (usando uma aresta condicional).

Executamos o grafo com `{"name": "Lance", "mood": "happy"}` → Funciona como esperado.

Para restrições de valor mais específicas, você pode usar dicas de tipo como `Literal`.

Aqui, `mood` só pode ser "happy" ou "sad".

```python
from typing import Literal

class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy","sad"]
```

Podemos usar nossa classe de estado definida (por exemplo, aqui `TypedDictState`) no LangGraph simplesmente passando-a para `StateGraph`.

E podemos pensar em cada chave de estado como apenas um "canal" em nosso grafo.

Como discutido no Módulo 1, sobrescrevemos o valor de uma chave ou "canal" específico em cada nó.

```python
import random
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

def node_1(state):
    print("---Node 1---")
    return {"name": state['name'] + " is ... "}

def node_2(state):
    print("---Node 2---")
    return {"mood": "happy"}

def node_3(state):
    print("---Node 3---")
    return {"mood": "sad"}

def decide_mood(state) -> Literal["node_2", "node_3"]:
        
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"

# Build graph
builder = StateGraph(TypedDictState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:f743233d-0f05-47f6-aee4-01f9535d21fb:image.png)

Como nosso estado é um dicionário, simplesmente invocamos o gráfico com um dicionário para definir um valor inicial da chave `name` em nosso estado.

```python
graph.invoke({"name":"Lance"})
```

```python
---Node 1---
---Node 2---
{'name': 'Lance is ... ', 'mood': 'happy'}
```

### 📘 Data Classes

Outra forma de definir um schema é usando **`dataclasses` do Python**.

- Sintaxe concisa;
- Também serve para definir dados estruturados;
- Ao invés de acessar via `state["name"]`, acessamos via `state.name`.

Executamos o grafo com uma instância da `dataclass` → Funciona normalmente.

```python
from dataclasses import dataclass

@dataclass
class DataclassState:
    name: str
    mood: Literal["happy","sad"]
```

Para acessar as chaves de um `dataclass`, só precisamos modificar a indexação usada em `node_1`:

- Usamos `state.name` para o estado `dataclass` em vez de `state["name"]` como fizemos com o `TypedDict` acima

Você notará algo um pouco estranho: em cada nó, ainda retornamos um dicionário para realizar as atualizações de estado.

Isso é possível porque o LangGraph armazena cada chave do seu objeto de estado separadamente.

O objeto retornado pelo nó só precisa ter chaves (atributos) que correspondam às do estado!

Neste caso, o `dataclass` tem a chave `name`, então podemos atualizá-la passando um dict do nosso nó, exatamente como fizemos quando o estado era um `TypedDict`.

```python
def node_1(state):
    print("---Node 1---")
    return {"name": state.name + " is ... "}

# Build graph
builder = StateGraph(DataclassState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:790fb286-bba5-4492-bc37-3f2e21cf3e9f:image.png)

Invocamos com uma `dataclass` para definir os valores iniciais de cada chave / canal em nosso estado!

```python
graph.invoke(DataclassState(name="Lance",mood="sad"))
```

### ⚠️ Limitação: sem validação em tempo de execução

Tanto com `TypedDict` quanto com `dataclass`:

- Os **type hints não são validados em tempo de execução**;
- Isso quer dizer que é possível passar um valor inválido sem erro.

Por exemplo:

- Criamos uma `dataclass` com `mood` aceitando apenas `"happy"` ou `"sad"`;
- Passamos `"mad"` como valor;
- **Nenhum erro é levantado**, mesmo sendo inválido.

---

### ✅ Solução: **Pydantic**

A biblioteca [**Pydantic**](https://docs.pydantic.dev/) resolve esse problema com:

- **Validação de dados automática**;
- Verificação em **tempo de execução**;
- Muito usada em projetos robustos, como APIs.

### Exemplo:

Criamos uma `Pydantic BaseModel` com:

- `name: str`
- `mood: str`, com um **validador** que só aceita `"happy"` ou `"sad"`

Tentamos criar um objeto com `mood = "mad"` →

✅ Recebemos um erro de validação: **"Only happy or sad permitted."**

Podemos passar esse modelo como **estado do grafo no LangGraph**, normalmente.

Tentamos passar um valor inválido →

✅ Erro de validação corretamente exibido.

Passamos um valor válido (`"sad"`) →

✅ Funciona como esperado.

Como mencionado, `TypedDict` e `dataclasses` fornecem dicas de tipo, mas não impõem tipos em tempo de execução.

Isso significa que você poderia potencialmente atribuir valores inválidos sem gerar um erro!

Por exemplo, podemos definir `mood` como `mad` mesmo que nossa dica de tipo especifique `mood: list[Literal["happy","sad"]]`.

```python
dataclass_instance = DataclassState(name="Lance", mood="mad")
```

[Pydantic](https://docs.pydantic.dev/latest/api/base_model/) é uma biblioteca de validação de dados e gerenciamento de configurações que utiliza anotações de tipo do Python.

É particularmente adequado [para definir esquemas de estado no LangGraph](https://langchain-ai.github.io/langgraph/how-tos/state-model/) devido às suas capacidades de validação.

O Pydantic pode realizar validações para verificar se os dados estão em conformidade com os tipos e restrições especificados em tempo de execução.

```python
from pydantic import BaseModel, field_validator, ValidationError

class PydanticState(BaseModel):
    name: str
    mood: str # "happy" or "sad" 

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        # Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value

try:
    state = PydanticState(name="John Doe", mood="mad")
except ValidationError as e:
    print("Validation Error:", e)
```

Podemos usar `PydanticState` em nosso gráfico perfeitamente.

```python
# Build graph
builder = StateGraph(PydanticState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:b577e4d7-d508-4972-b484-db872690139f:image.png)

```python
graph.invoke(PydanticState(name="Lance",mood="sad"))
```

### 🧩 Conclusão

O `Pydantic` é excelente quando você quer:

- Garantir que os dados do seu estado estejam **bem validados**;
- Prevenir erros silenciosos;
- Criar agentes mais robustos.

Você pode usar:

- `TypedDict`: simples e flexível, mas sem validação em tempo real;
- `dataclass`: conciso, mas também sem validação;
- `Pydantic`: com validação completa e controle total sobre os dados.