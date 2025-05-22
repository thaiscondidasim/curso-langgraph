# Lição 6 - Chatbot com sumarização de mensagens e memória em banco de dados externo

Neste trecho, o autor mostra como tornar um **chatbot mais robusto para conversas de longa duração**, com **persistência de memória entre sessões**, usando:

1. **Resumo contínuo da conversa** (para compressão e economia de tokens).
2. **Checkpointer com persistência local** via **SQLite**.
3. **Persistência automática com Postgres** via **LangGraph Studio**.

Essas abordagens resolvem a limitação da memória "temporária" que só dura enquanto o notebook estiver ativo.

Construímos um chatbot que suporta **conversas longas** de duas maneiras:

1. **Memória persistente na sessão**: usamos um *checkpointer em memória*, que mantém o histórico da conversa enquanto o notebook estiver ativo.
2. **Resumos da conversa**: após certo número de mensagens (ex: 6), o chatbot resume as mensagens antigas. Isso **reduz o uso de tokens** e torna mais viável manter conversas longas.

---

### ⚠️ Limitação

Esse chatbot **não é persistente indefinidamente**, porque usamos um **checkpointer em memória**, que perde os dados ao encerrar o notebook.

---

### 💡 Solução: usar banco de dados externo

O **LangGraph** suporta diferentes checkpointers que funcionam com **bancos de dados externos**, como:

- **SQLite** (banco de dados leve e local);
- **Postgres** (mais robusto, usado em produção).

## Sqlite

Um bom ponto de partida aqui é o [checkpointer SqliteSaver](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer).

O Sqlite é um banco de dados SQL [pequeno, rápido e muito popular](https://x.com/karpathy/status/1819490455664685297).

Se fornecermos `":memory:"`, ele criará um banco de dados SQLite na memória.

```python
import sqlite3
# In memory
conn = sqlite3.connect(":memory:", check_same_thread = False)
```

Mas, se fornecermos um caminho para o banco de dados, ele criará um banco de dados para nós!

```python
# pull file if it doesn't exist and connect to local db
!mkdir -p state_db && [ ! -f state_db/example.db ] && wget -P state_db https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db

db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
```

```python
# Here is our checkpointer 
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver(conn)
```

Vamos redefinir nosso chatbot.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from langgraph.graph import END
from langgraph.graph import MessagesState

model = ChatOpenAI(model="gpt-4o",temperature=0)

class State(MessagesState):
    summary: str

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

Agora, basta recompilar com nosso checkpointer sqlite.

```python
from IPython.display import Image, display
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
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:3cd1f6b3-bd86-462a-ad9f-21c559417fa0:image.png)

---

### 🛠️ Exemplo com SQLite

1. Importamos o módulo do SQLite.
2. Se passarmos `"memory"` na conexão, o banco funciona em memória (igual ao anterior).
3. Se passarmos um **caminho de arquivo**, ele cria/usa um banco local persistente.

---

### 🔗 Criação do checkpointer

- Criamos o checkpointer com SQLite.
- Chamamos isso de `memory`.
- Conectamos.
- Criamos o chatbot igual aos exemplos anteriores.

---

### 🔁 Fluxo da conversa

- Começamos com `"Oi, sou o Lance"`.
- O LLM responde.
- Após 6 mensagens, o grafo dispara um resumo.
- O resumo é salvo na chave `summary` do estado.

Esse fluxo segue o mesmo padrão já demonstrado: a cada passo, usamos o `call_model`, que checa se existe um resumo e o adiciona ao contexto.

Agora, basta recompilar com nosso checkpointer sqlite.

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
/var/folders/l9/bpjxdmfx7lvd1fbdjn38y5dh0000gn/T/ipykernel_18873/2173919996.py:55: LangChainBetaWarning: The class `RemoveMessage` is in beta. It is actively being worked on, so the API may change.
  delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
================================== Ai Message ==================================

Hello again, Lance! It's great to hear from you. Since you like the 49ers, is there a particular player or moment in their history that stands out to you? Or perhaps you'd like to discuss their current season? Let me know!
================================== Ai Message ==================================

Your name is Lance! How can I assist you today? Would you like to talk more about the San Francisco 49ers or something else?
================================== Ai Message ==================================

That's awesome, Lance! The San Francisco 49ers have a rich history and a passionate fan base. Is there a specific aspect of the team you'd like to discuss? For example, we could talk about:

- Their legendary players like Joe Montana and Jerry Rice
- Memorable games and Super Bowl victories
- The current roster and season prospects
- Rivalries, like the one with the Seattle Seahawks
- Levi's Stadium and the fan experience

Let me know what interests you!
```

---

### 🔄 Teste de persistência

1. Executamos o notebook e salvamos a conversa.
2. **Reiniciamos o kernel** (interrompendo tudo).
3. Se fosse um checkpointer em memória, **perderíamos tudo**.
4. Como estamos usando o SQLite, os dados estão **salvos em disco**.
5. Recriamos o grafo e... **o estado persiste!**
    
    ✅ Ao passar o `thread_id`, recuperamos o estado anterior da conversa.
    

---

### 📦 Uso com LangGraph Studio

O **LangGraph Studio** tem uma **camada de persistência embutida**, que:

- Funciona via API;
- Usa **PostgreSQL** por trás;
- Permite manter o histórico de todas as conversas (**threads**) localmente ou remotamente.

---

### 🧪 Exemplo prático no Studio

1. Abrimos o projeto `chatbot.py` no editor.
2. Empacotamos com a API do LangGraph.
3. A API automaticamente **adiciona persistência** via Postgres.
4. Executamos o bot no Studio:
    - `Oi, sou o Lance`
    - `Gosto dos 49ers`
    - `Quem é o técnico?`
    - `Me fale sobre o pai dele` (Mike Shanahan)
5. Após a sexta mensagem, o grafo **gera um resumo da conversa**.

---

### 🧵 Threads e continuidade

- Cada conversa é uma **thread** com `thread_id`.
- Podemos criar uma nova thread:
    - `"Oi, sou o Lance"`
    - `"Quem é o maior rival dos 49ers?"` → Cowboys, anos 80, 90 etc.
- Podemos voltar à thread anterior:
    - `"Quem foi o melhor jogador do Mike Shanahan?"`
    - E o sistema continua **do ponto em que paramos**, com o contexto salvo.

---

### ✅ Conclusão

| Estratégia | O que permite fazer |
| --- | --- |
| `MemorySaverCheckpointer` | Persistência em memória (até encerrar a sessão do notebook) |
| `SQLiteCheckpointer` | Persistência local (salva no disco com SQLite) |
| **LangGraph Studio (Postgres)** | Persistência automática, empacotada via API com suporte a múltiplas threads |

---

📌 **Benefícios combinados**:

- Conversas **longas e contínuas**, sem perda de contexto.
- Compressão de mensagens antigas com **resumo automático**.
- Persistência mesmo após **reinício do sistema** ou mudança de ambiente.

---

Se quiser, posso gerar um exemplo real com `LangGraph + SQLite` ou te ajudar a rodar esse fluxo completo localmente ou na Studio. Deseja seguir por esse caminho?