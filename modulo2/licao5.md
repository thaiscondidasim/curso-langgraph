# Lição 5 - Chatbot com mensagem sumarizada

Neste trecho, aprendemos uma técnica **avançada e prática** para lidar com **conversas longas em chatbots** usando LangGraph: **criar um resumo contínuo da conversa** (*running summary*) com um **LLM**.

Essa técnica é uma forma de *compressão inteligente* da conversa que **preserva o contexto** sem sobrecarregar o modelo com todas as mensagens antigas — o que é comum em conversas longas e pode aumentar muito o consumo de tokens e latência.

Já aprendemos como personalizar o schema de estado do grafo e como criar reducers personalizados. Também vimos várias maneiras de **filtrar ou truncar mensagens**.

Agora, vamos **dar um passo além**: mostrar uma técnica muito útil que **usa LLMs para produzir um resumo contínuo da conversa**. Essa é uma **alternativa ao filtro ou truncamento de mensagens**, tentando **preservar mais contexto e informação**.

---

### ⚙️ Configuração inicial

1. Definimos nosso **modelo de linguagem** (LLM).

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o",temperature=0)
```

1. Usamos o `messages state`, como antes — que já tem a chave embutida `messages`.

```python
from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str
```

1. **Adicionamos uma nova chave chamada `summary`** — que vai armazenar o **resumo da conversa até o momento**.
2. Definiremos um nó para chamar nosso LLM que incorpora um resumo, se existir, no prompt.

```python
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

# Define the logic to call the model
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": response}
```

---

### 🧩 Criação dos nós:

1. **`call_model`**:
    - Verifica se já existe um resumo (`summary`).
    - Se sim, **adiciona esse resumo às mensagens** (como contexto).
    - Invoca o modelo LLM com as mensagens + resumo.
    - Escreve a resposta no `messages`.
2. **`summarize_conversation`**:
    - Verifica novamente se há um resumo atual.
    - Se sim, o modelo é instruído a **"estender o resumo atual" com as novas informações**.
    - Se não, o modelo é instruído a **"resumir a conversa acima"**.
    - A saída é armazenada na chave `summary`.
3. Após isso, usamos o truque do `removeMessages` que já vimos:
    - **Removemos todas as mensagens**, exceto as duas mais recentes.
    - Assim, mantemos apenas o essencial e evitamos excesso de tokens.

Definiremos um nó para produzir um resumo.

Observe que aqui usaremos `RemoveMessage` para filtrar nosso estado após produzirmos o resumo.

```python
def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
```

Adicionaremos uma aresta condicional para determinar se um resumo deve ser produzido com base na duração da conversa.

```python
from langgraph.graph import END
# Determine whether to end or summarize the conversation
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END
```

---

### 🤖 Lógica de controle

Adicionamos uma **condição** simples:

- **Se houver mais de 6 mensagens**, acionamos o `summarize_conversation`.
- Isso pode ser ajustado para qualquer número (por exemplo, 10, 20 etc.).

---

### 💾 Persistência com Checkpointer

Como  [o estado é transitório](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220) para uma única execução do grafo (ou seja, se perde a cada execução), usamos o **checkpointer** — um componente que salva o estado do grafo ao longo do tempo.

Isso limita nossa capacidade de ter conversas com múltiplos turnos e interrupções.

Como introduzido no final do Módulo 1, podemos usar [persistência](https://langchain-ai.github.io/langgraph/how-tos/persistence/) para resolver isso!

O LangGraph pode usar um checkpointer para salvar automaticamente o estado do grafo após cada passo.

Esta camada de persistência integrada nos dá memória, permitindo que o LangGraph retome a partir da última atualização de estado.

Como mostramos anteriormente, um dos mais fáceis de trabalhar é o `MemorySaver`, um armazenamento chave-valor em memória para o estado do Grafo.

Tudo o que precisamos fazer é compilar o grafo com um checkpointer, e nosso grafo terá memória!

📌 Usamos aqui o **MemorySaverCheckpointer**, que armazena os dados em memória como um **key-value store**.

### Como funciona:

- A cada execução, o checkpointer **salva um checkpoint** do estado.
- Cada **thread** de conversa tem um `thread_id`, que nos permite continuar a conversa depois.
- A cada nova entrada do usuário, a mensagem é **acrescentada ao histórico salvo** e o modelo LLM é invocado com esse estado completo.

```python
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:1c73d2fe-59ef-4dff-8a9b-7f2aca37b4e8:image.png)

## Threads

O checkpointer salva o estado em cada etapa como um checkpoint.

Esses checkpoints salvos podem ser agrupados em uma `thread` de conversação.

Pense no Slack como analogia: diferentes canais carregam diferentes conversas.

Threads são como canais do Slack, capturando coleções agrupadas de estado (ex: conversação).

Abaixo, usamos `configurable` para definir um ID de thread.

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbadf3b379c2ee621adfd1_chatbot-summarization1.png)

### 💬 Exemplo prático

1. Enviamos a primeira mensagem:
    
    `"Oi, sou o Lance"`
    
    ✅ O modelo responde e o estado é salvo.
    
2. Nova entrada:
    
    `"Qual é o meu nome?"`
    
    ✅ O modelo responde: `"Você disse que se chama Lance"`
    
    👉 Mesmo que só tenhamos enviado essa pergunta, o modelo teve acesso ao histórico completo graças ao checkpointer.
    
3. Seguimos:
    
    `"Gosto dos 49ers"`
    
    ✅ O modelo responde com entusiasmo.
    
4. Continuamos com mais interações até atingir **6 mensagens**.
5. Ao atingir esse limite:
    
    ✅ O `summarize_conversation` entra em ação.
    
    ✅ O modelo gera algo como:
    
    `"Lance se apresentou, disse que gosta dos 49ers e perguntou sobre o jogador Nick Bosa."`
    

🧠 Esse resumo é **armazenado em `summary`** e **as mensagens antigas são apagadas** (exceto as duas mais recentes), mantendo o estado do grafo enxuto.

```python
# Create a thread
config = {"configurable": {"thread_id": "1"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Lance")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

Hi Lance! How can I assist you today?
==================================[1m Ai Message [0m==================================

You mentioned that your name is Lance. How can I help you today, Lance?
==================================[1m Ai Message [0m==================================

That's awesome, Lance! The San Francisco 49ers have a rich history and a passionate fan base. Do you have a favorite player or a memorable game that stands out to you?
```

Ainda não temos um resumo do estado porque ainda temos < = 6 mensagens.

Isso foi definido em `should_continue`.

```
# Se houver mais de seis mensagens, resumimos a conversa
if len(messages) > 6:
return "summarize_conversation"

```

Podemos retomar a conversa porque temos o tópico.

```python
graph.get_state(config).values.get("summary","")
```

```python
O `config` com ID do thread nos permite prosseguir a partir do estado registrado anteriormente!
```

```python
input_message = HumanMessage(content="i like Nick Bosa, isn't he the highest paid defensive player?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()
```

```python
==================================[1m Ai Message [0m==================================

Yes, Nick Bosa is indeed one of the highest-paid defensive players in the NFL. In September 2023, he signed a record-breaking contract extension with the San Francisco 49ers, making him the highest-paid defensive player at that time. His performance on the field has certainly earned him that recognition. It's great to hear you're a fan of such a talented player!
/var/folders/l9/bpjxdmfx7lvd1fbdjn38y5dh0000gn/T/ipykernel_18661/23381741.py:23: LangChainBetaWarning: The class `RemoveMessage` is in beta. It is actively being worked on, so the API may change.
  delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
```

```python
graph.get_state(config).values.get("summary","")
```

```python
"Lance introduced himself and mentioned that he is a fan of the San Francisco 49ers. He specifically likes Nick Bosa and inquired if Bosa is the highest-paid defensive player. I confirmed that Nick Bosa signed a record-breaking contract extension in September 2023, making him the highest-paid defensive player at that time, and acknowledged Bosa's talent and Lance's enthusiasm for the player."
```

---

### 📊 Verificação com LangSmith

Usamos o **LangSmith** (ferramenta de observabilidade para fluxos com LLMs) para inspecionar o que de fato é enviado ao modelo.

- Vemos que, mesmo que a entrada atual seja apenas `"Qual é o meu nome?"`, o modelo recebe o histórico todo salvo pelo checkpointer.
- Isso porque o LangGraph **usa o estado persistente da conversa** e só **acrescenta a nova mensagem**.
- Após o sexto passo, confirmamos que o **resumo é gerado corretamente**.

---

### 🧠 Conclusão: estratégia eficiente de memória

Resumindo, o que fizemos foi:

| Etapa | O que acontece |
| --- | --- |
| `call_model` | Adiciona o resumo (se existir) às mensagens e invoca o LLM |
| `summarize_conversation` | Gera um resumo contínuo da conversa ao ultrapassar o limite de mensagens |
| `removeMessages` | Apaga mensagens antigas, mantendo o estado enxuto |
| `MemorySaverCheckpointer` | Persiste o estado da conversa entre execuções |

🧠 **Benefícios:**

- Mantemos **o contexto da conversa**, mesmo após muitas interações.
- **Reduzimos o número de tokens** passados ao modelo.
- Facilitamos a continuidade de **conversas de longo prazo**, como assistentes ou agentes interativos.