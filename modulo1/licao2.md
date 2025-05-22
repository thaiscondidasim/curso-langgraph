Vamos construir um grafo simples para apresentar os componentes principais do **LangGraph**.

Vamos começar com três nós: **nó 1**, **nó 2** e **nó 3**, e então vamos terminar no nó **fim**. A conexão entre o **início** e o **nó 1** é o que chamamos de **aresta normal (normal edge)** – ou seja, o grafo sempre começa e vai diretamente para o nó 1.

![image.png](https://img.notionusercontent.com/s3/prod-files-secure%2F43a77b18-9895-4da3-a6fa-c2bf840e8ecd%2Fb8230ebe-c527-4c25-bd1e-c96eea40505c%2Fimage.png/size/w=2000?exp=1747947980&sig=-D6NXuIbx8hS7Ys3oJIs6AEToTdksjDdkZ4h0jEKWt8&id=1f9bd7fa-42ba-8008-b2d9-cb4bb595a2e4&table=block&userId=9bacf5ce-08bc-45e1-b61a-bd4328577f69)

Agora, repare que a conexão entre os nós 2 e 3 é um pouco diferente. Ela parte do nó 1 e **ramifica**. Chamamos isso de **aresta condicional (conditional edge)**, o que significa que, com base em uma condição que definirmos, o grafo seguirá para o nó 2 ou para o nó 3. Vamos ver isso em detalhes.

Por fim, tanto o nó 2 quanto o nó 3 levam até o nó final. Esse é o grafo simples que vamos construir. Ele é uma forma interessante de apresentar os componentes principais do LangGraph.

---

### 1. **Instalação**

Primeiro, executamos o comando:

```bash
pip install langgraph

```

---

### 2. **Definindo o Estado**

O **estado (state)** é o objeto que será passado entre os nós e arestas do grafo. Neste exemplo, vamos defini-lo como um **dicionário simples com uma chave `graph_state`**, que conterá uma string.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

---

### 3. **Criando os Nós**

[Nós](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes) são simplesmente funções em Python.

O primeiro argumento posicional é o estado, conforme definido anteriormente.

Como o estado é um `TypedDict` com o esquema definido acima, cada nó pode acessar a chave `graph_state` usando `state['graph_state']`.

Cada nó retorna um novo valor para a chave de estado `graph_state`.

Por padrão, o novo valor retornado por cada nó [substituirá](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) o valor anterior do estado.

Vamos definir 3 nós (node_1, node_2 e node_3). Cada nó:

- Recebe o `state` (o dicionário).
- Lê o valor da chave `graph_state`.
- Acrescenta uma string ao final do valor.
- Atualiza `graph_state` com esse novo valor.

Exemplo:

- Nó 1: adiciona `" I am"`
- Nó 2: adiciona `" happy"`
- Nó 3: adiciona `" sad"`

```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}
```

---

### 4. **Criando Arestas**

[Arestas](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges) conectam os nós.

**Arestas normais** são usadas quando você quer ir *sempre* de um nó para outro, por exemplo, de `node_1` para `node_2`.

**Arestas condicionais** são usadas quando você quer rotear *opcionalmente* entre nós com base em alguma lógica.

Elas são implementadas como funções que retornam o próximo nó a ser visitado, dependendo de uma condição.

- **Aresta normal**: conecta o nó inicial ao nó 1.
- **Aresta condicional**: conecta o nó 1 ao nó 2 ou 3 **com base numa função** que decide o caminho.

### Exemplo da função `decide_mood`:

Essa função escolhe entre os nós 2 e 3 com base em um número aleatório (50% de chance para cada um).

```python
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"
```

---

### 5. **Montando o Grafo**

Usamos a classe `StateGraph` para:

- Definir o estado inicial.
- Adicionar os nós.
- Conectar as arestas (normais e condicionais).
- Compilar o grafo.

Também podemos exibir o grafo visualmente (útil para debug e entendimento do fluxo).

---

### 6. **Executando o Grafo**

Agora, vamos construir o grafo a partir dos [componentes](https://langchain-ai.github.io/langgraph/concepts/low_level/) definidos anteriormente.

A classe [StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph) é a classe que utilizaremos para criar o grafo.

Primeiro, inicializamos um `StateGraph` com a classe `State` que definimos.

Em seguida, adicionamos nossos nós e arestas.

Utilizamos o [nó `START`, um nó especial](https://langchain-ai.github.io/langgraph/concepts/low_level/#start-node) que envia a entrada do usuário para o grafo, para indicar onde o grafo deve começar.

O [nó `END`](https://langchain-ai.github.io/langgraph/concepts/low_level/#end-node) é um nó especial que representa um ponto final.

Por fim, [compilamos o grafo](https://langchain-ai.github.io/langgraph/concepts/low_level/#compiling-your-graph) para realizar algumas verificações básicas na estrutura.

Podemos visualizar o grafo como um [diagrama Mermaid](https://github.com/mermaid-js/mermaid).

O grafo implementa o protocolo `runnable`, que permite usar o método `invoke`.

Chamamos `invoke` com o estado inicial:

```python
{"graph_state": "hi this is Lance"}

```

E o grafo executa:

- Nó 1 adiciona `" I am"`.
- Nó 2 ou 3 adiciona `" happy"` ou `" sad"`.

Exemplo de resultado final:

```json
{"graph_state": "hi this is Lance I am happy"}

```

Se executarmos várias vezes, vamos ver respostas diferentes, porque a escolha entre 2 e 3 é aleatória.

---

### 🔸 **Resumo Visual do Grafo**

```
start
  |
  v
node_1 (acrescenta "I am")
  |
 (condicional)
 /     \
v       v
node_2  node_3
("happy") ("sad")
  \       /
   v     v
    end

```

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
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

---

Invoca

## 🧠 **Explicações de termos e conceitos técnicos**

| Termo | Explicação |
| --- | --- |
| **LangGraph** | Biblioteca que permite construir fluxos de controle (workflows) baseados em grafos com LLMs. |
| **Grafo (Graph)** | Estrutura de dados com nós (etapas do processo) e arestas (ligações entre etapas). |
| **Node (nó)** | Função que realiza uma operação com o estado atual e o retorna modificado. |
| **State (estado)** | Dado (geralmente um dicionário) que é passado de nó para nó durante a execução do grafo. |
| **Aresta normal (Normal edge)** | Transição direta entre dois nós, sempre acontece. |
| **Aresta condicional (Conditional edge)** | Transição entre dois ou mais nós, controlada por uma função de decisão. |
| **invoke** | Método padrão para executar grafos ou componentes do LangChain/LangGraph. Executa todo o fluxo de uma vez. |
| **síncrono (synchronous)** | A execução espera cada etapa ser concluída antes de passar para a próxima. |
| **Runnable protocol** | Interface padrão para objetos que podem ser executados como fluxos, incluindo grafos e chains. |
| **Compile** | Valida e prepara o grafo para execução. |