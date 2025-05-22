# Lição 3 - Map Reduce

Estamos construindo um **assistente de pesquisa multiagente** que conecta todos os módulos deste curso. Para isso, temos discutido alguns tópicos de **controlabilidade** com o LangGraph. Já cobrimos **paralelização** e **subgrafos**. Agora vamos falar sobre **MapReduce**.

Como antes, vamos usar o **LangSmith** para rastreamento (*tracing*). Vamos primeiro entender o que é MapReduce.

## Revisão

Estamos desenvolvendo um assistente de pesquisa multiagente que integra todos os módulos deste curso.

Para construir este assistente multiagente, apresentamos alguns tópicos de controlabilidade do LangGraph.

Acabamos de abordar paralelização e subgrafos.

## Objetivos

Agora, vamos abordar [map reduce](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/).

---

### **O que é MapReduce?**

**MapReduce** é basicamente um método eficiente para **dividir tarefas** e **processar em paralelo**. Ele possui duas fases:

1. **Fase Map**: você pega uma tarefa, a divide em várias subtarefas e executa todas **em paralelo**.
2. **Fase Reduce**: você **agrega os resultados** dessas subtarefas paralelas e os junta para gerar uma saída final.

---

### **Exemplo prático (toy example):**

Na fase *Map*, vamos criar várias **piadas** sobre um determinado tema.

Na fase *Reduce*, escolhemos a **melhor piada**.

1. Primeiro, criamos um *prompt* para gerar uma lista de **assuntos relacionados ao tema**.
2. Depois, um *prompt* para **gerar piadas** sobre cada assunto.
3. E um terceiro *prompt* para **escolher a melhor piada** da lista.

---

### **Gerenciamento de estado**

Usamos dois **schemas de saída estruturada**:

- `subjects` (lista de assuntos),
- `best_joke` (índice da melhor piada).

O **estado geral** do grafo incluirá:

- o tema original (`topic`),
- a lista de `subjects`,
- a lista de `jokes` (onde usaremos um **reducer** que adiciona à lista),
- e o índice da `best_joke`.

---

## Problema

Operações de mapeamento e redução são essenciais para a decomposição eficiente de tarefas e o processamento paralelo.

Elas têm duas fases:

(1) `Mapear` - Dividir uma tarefa em subtarefas menores, processando cada subtarefa em paralelo.

(2) `Reduzir` - Agregar os resultados de todas as subtarefas concluídas e paralelizadas.

Vamos projetar um sistema que fará duas coisas:

(1) `Mapear` - Criar um conjunto de piadas sobre um tópico.

(2) `Reduzir` - Escolher a melhor piada da lista.

Usaremos um LLM para gerar e selecionar as tarefas.

```python
from langchain_openai import ChatOpenAI

# Prompts we will use
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# LLM
model = ChatOpenAI(model="gpt-4o", temperature=0) 
```

### **Fase Map – geração das piadas**

Primeiro, vamos definir o ponto de entrada do grafo que irá:

- Receber um tópico de entrada do usuário
- Produzir uma lista de tópicos de piadas a partir dele
- Enviar cada tópico de piada para o nosso nó de geração de piadas acima

Nosso estado tem uma chave `jokes`, que acumulará piadas da geração de piadas em paralelo

Após gerar os `subjects`, usamos a API `send` do LangGraph para:

- **iterar sobre a lista** de assuntos,
- enviar cada item a um nó `generate_joke` que gera uma piada para aquele assunto.

Esse `generate_joke`:

- tem um estado próprio (contém apenas o `subject`),
- gera a piada com um LLM e
- escreve a piada de volta no estado principal, na chave `jokes` (com reducer que agrega em lista).

**A API `send` automatiza** a criação de múltiplos nós `generate_joke`, com base no tamanho da lista `subjects`. Você **não precisa desenhar manualmente** todos os caminhos no grafo — o LangGraph cuida disso para você.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel

class Subjects(BaseModel):
    subjects: list[str]

class BestJoke(BaseModel):
    id: int
    
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str
```

```python
def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}
```

Aqui está a mágica: usamos [Send](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) para criar uma piada para cada sujeito.

Isso é muito útil! Ele pode paralelizar automaticamente a geração de piadas para qualquer número de sujeitos.

- `generate_joke`: o nome do nó no grafo
- `{"subject": s`}: o estado a ser enviado

`Send` permite que você passe qualquer estado que queira para `generate_joke`! Ele não precisa estar alinhado com `OverallState`.

Neste caso, `generate_joke` está usando seu próprio estado interno, e podemos preenchê-lo via `Send`.

```python
from langgraph.constants import Send
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
```

### Geração de piadas (mapa)

Agora, definimos um nó que criará nossas piadas, `generate_joke`!

Nós as escrevemos de volta em `jokes` em `OverallState`!

Esta chave possui um redutor que combinará listas.

```python
class JokeState(TypedDict):
    subject: str

class Joke(BaseModel):
    joke: str

def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}
```

---

### **Fase Reduce – escolha da melhor piada**

Com a lista de piadas gerada, usamos um nó `best_joke`:

- Pega a lista `jokes`,
- Passa para o LLM decidir qual é a melhor,
- Retorna o índice (`int`) da melhor piada com base no schema `best_joke`.

A piada vencedora pode então ser exibida ao usuário.

```python
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}
```

```python
from IPython.display import Image
from langgraph.graph import END, StateGraph, START

# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

# Compile the graph
app = graph.compile()
Image(app.get_graph().draw_mermaid_png())
```

![image.png](attachment:8381b80f-48e9-49c9-9b6e-6c015331eab5:image.png)

```python
# Call the graph: here we call it to generate a list of jokes
for s in app.stream({"topic": "animals"}):
    print(s)
```

```python
{'generate_topics': {'subjects': ['mammals', 'reptiles', 'birds']}}
{'generate_joke': {'jokes': ["Why don't mammals ever get lost? Because they always follow their 'instincts'!"]}}
{'generate_joke': {'jokes': ["Why don't alligators like fast food? Because they can't catch it!"]}}
{'generate_joke': {'jokes': ["Why do birds fly south for the winter? Because it's too far to walk!"]}}
{'best_joke': {'best_selected_joke': "Why don't alligators like fast food? Because they can't catch it!"}}
```

---

### **Visualização no LangSmith e Studio**

No **LangSmith**, vemos:

- Primeiro o nó `generate_topics` que gera a lista: ex. `"leão", "elefante", "golfinho", "pinguim"`.
- Em seguida, a **linha pontilhada** no grafo indica a chamada `send` que dispara múltiplos `generate_joke` automaticamente.
- Cada nó `generate_joke` cria uma piada com base no assunto.
- Depois, o nó `best_joke` seleciona a melhor piada da lista (ex.: piada nº 2).
- Finalmente, o grafo termina e retorna essa piada como saída.

Ao rodar isso no **Studio**, podemos inclusive **interromper** a execução (usando `interrupt`) para examinar cada parte do processo com calma, passo a passo.

---

### **Resumo visual do fluxo MapReduce com LangGraph:**

```
Usuário fornece tema → Gerar lista de assuntos (Map)
    ↓
    Para cada assunto → Gerar piada (paralelo, via send API)
    ↓
Coletar todas as piadas → Escolher a melhor (Reduce)
    ↓
Retornar piada escolhida

```

---

### **Explicações dos termos técnicos:**

- **MapReduce**: técnica de computação distribuída em que você divide uma tarefa em várias partes (Map) e depois agrega os resultados (Reduce).
- **LangGraph**: biblioteca para construção de grafos de execução com LLMs, que facilita fluxos como agentes, paralelização, subgrafos, e estados compartilhados.
- **LangSmith**: ferramenta de rastreamento e inspeção visual dos grafos LangGraph. Permite ver a execução passo a passo.
- **Schema estruturado**: definição da forma esperada da saída do LLM, como lista de strings, inteiros, objetos, etc. Garante que o modelo retorne dados no formato esperado.
- **Reducer**: função que controla como valores são combinados em uma chave do estado. Por exemplo, adicionar cada piada a uma lista.
- **Send API**: função do LangGraph que envia dinamicamente entradas para múltiplos nós com base em uma lista. Automatiza a ramificação (*fan out*).
- **Conditional Edge (aresta condicional)**: linha pontilhada no grafo que representa a ativação dinâmica de nós com base em uma condição — como a existência de itens numa lista.

---

Se quiser, posso:

- Gerar um **diagrama PlantUML estilo C4** com esse fluxo,
- Criar um **exemplo de código LangGraph em Python** com esse MapReduce,
- Ou adaptar o exemplo para um caso real, como resumo de reviews ou coleta de notícias.

Quer seguir com algum desses?