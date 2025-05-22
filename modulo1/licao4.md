Anteriormente, construímos um grafo simples com nós, arestas normais e arestas condicionais.

Agora, vamos avançar para **chains**, que vão combinar alguns conceitos fundamentais:

- mensagens de chat,
- modelos de chat,
- ligação (binding) de ferramentas,
- e execução de chamadas de ferramentas — tudo isso dentro do **LangGraph**.

## Objetivos

Agora, vamos construir uma cadeia simples que combina 4 [conceitos](https://python.langchain.com/v0.2/docs/concepts/):

- Usar [mensagens de chat](https://python.langchain.com/v0.2/docs/concepts/#messages) como nosso estado de gráfico
- Usar [modelos de chat](https://python.langchain.com/v0.2/docs/concepts/#chat-models) em nós do gráfico
- [Vincular ferramentas](https://python.langchain.com/v0.2/docs/concepts/#tools) ao nosso modelo de chat
- [Executar chamadas de ferramentas](https://python.langchain.com/v0.2/docs/concepts/#functiontool-calling) em nós do gráfico

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbab08dd607b08df5e1101_chain1.png)

---

### 📌 Primeiros conceitos isolados

### **Mensagens**

Modelos de chat interagem com mensagens.

Aqui está um exemplo simples: podemos criar uma lista de mensagens que representam uma conversa entre uma IA e um humano.

Cada mensagem pode ter um **nome** e um **conteúdo**.

Modelos de chat podem usar [`mensagens`](https://python.langchain.com/v0.2/docs/concepts/#messages), que capturam diferentes papéis dentro de uma conversa.

O LangChain suporta vários tipos de mensagens, incluindo:

- `HumanMessage` (Mensagem Humana)
- `AIMessage` (Mensagem de IA)
- `SystemMessage` (Mensagem de Sistema)
- `ToolMessage` (Mensagem de Ferramenta)

Estas representam, respectivamente:

- Uma mensagem do usuário
- Uma mensagem do modelo de chat
- Uma instrução para o modelo definir seu comportamento
- Uma mensagem de retorno de uma chamada de ferramenta

Vamos criar uma lista de mensagens.

Cada mensagem pode ser configurada com:

- `content` - o conteúdo da mensagem
- `name` - opcionalmente, o autor da mensagem
- `response_metadata` - opcionalmente, um dicionário de metadados (geralmente preenchido pelo provedor do modelo para `AIMessages`)

Ao imprimir essa lista, temos algo assim:

```python
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage

messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Lance"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

for m in messages:
    m.pretty_print()
```

![image.png](attachment:0177e1cb-fc61-4de0-b305-be5df67f46c4:image.png)

Agora podemos passar essa lista diretamente para um modelo de chat.

Primeiro, garantimos que a chave da OpenAI está definida:

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

Importamos `ChatOpenAI`, especificamos o modelo (neste caso `gpt-4-0`) e invocamos o modelo com a lista de mensagens.

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
result = llm.invoke(messages)
type(result)
```

O resultado será uma mensagem da IA.

```python
AIMessage(content='One of the best places to see orcas in the United States is the Pacific Northwest, particularly around the San Juan Islands in Washington State. Here are some details:\n\n1. **San Juan Islands, Washington**: These islands are a renowned spot for whale watching, with orcas frequently spotted between late spring and early fall. The waters around the San Juan Islands are home to both resident and transient orca pods, making it an excellent location for sightings.\n\n2. **Puget Sound, Washington**: This area, including places like Seattle and the surrounding waters, offers additional opportunities to see orcas, particularly the Southern Resident killer whale population.\n\n3. **Olympic National Park, Washington**: The coastal areas of the park provide a stunning backdrop for spotting orcas, especially during their migration periods.\n\nWhen planning a trip for whale watching, consider peak seasons for orca activity and book tours with reputable operators who adhere to responsible wildlife viewing practices. Additionally, land-based spots like Lime Kiln Point State Park, also known as “Whale Watch Park,” on San Juan Island, offer great opportunities for orca watching from shore.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 228, 'prompt_tokens': 67, 'total_tokens': 295, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4', 'finish_reason': 'stop', 'logprobs': None}, id='run-57ed2891-c426-4452-b44b-15d0a5c3f225-0', usage_metadata={'input_tokens': 67, 'output_tokens': 228, 'total_tokens': 295, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})

```

O conteúdo será uma string com a resposta do LLM, e também teremos metadados da resposta, como informações sobre os tokens usados no prompt, nome do modelo, etc.

---

### 🛠️ Ferramentas (Tools)

Agora, vamos introduzir a ideia de **ferramentas** — outra forma de usar modelos de chat.

A ideia é simples: às vezes queremos que o modelo se conecte com uma **ferramenta externa**, como uma API que requer um payload específico.

Quando vinculamos uma API como ferramenta, damos ao modelo conhecimento sobre o esquema de entrada necessário.

O modelo decidirá chamar uma ferramenta com base na entrada em linguagem natural do usuário.

E retornará uma saída que adere ao esquema da ferramenta.

[Muitos provedores de LLM suportam chamadas de ferramentas](https://python.langchain.com/v0.1/docs/integrations/chat/) e a [interface de chamada de ferramentas](https://blog.langchain.dev/improving-core-tool-interfaces-and-docs-in-langchain/) no langchain é fácil de fazer.

Você pode simplesmente passar qualquer função Python para `ChatModel.bind_tools(função)`.

![](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbab08dc1c17a7a57f9960_chain2.png)

Exemplo:

Criamos uma função chamada `multiply` que recebe `a` e `b`.

```python
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])
```

Usamos `llm.bind_tools` para associar essa função ao modelo.

Agora o modelo está **ciente** dessa função.

Como no diagrama:

- Entramos com uma linguagem natural,
- E o modelo gera o payload necessário para executar a função.

Vamos testar:

Invocamos o modelo com a pergunta “Qual é 2 multiplicado por 3?”.

```python
tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multiplied by 3", name="Lance")])
```

O resultado: a mensagem da IA não tem um conteúdo direto, mas sim uma **chamada de ferramenta**.

```python
tool_call.tool_calls
[{'name': 'multiply',
  'args': {'a': 2, 'b': 3},
  'id': 'call_lBBBNo5oYpHGRqwxNaNRbsiT',
  'type': 'tool_call'}]
```

Ela inclui os argumentos e o nome da função — bem legal.

---

## Usando mensagens como estado

Com essas bases estabelecidas, podemos agora usar [`mensagens`](https://python.langchain.com/v0.2/docs/concepts/#messages) como estado em nosso grafo.

Vamos definir nosso estado, `MessagesState`, como um `TypedDict` com uma única chave: `messages`.

`messages` é simplesmente uma lista de mensagens, como definimos anteriormente (por exemplo, `HumanMessage`, etc.).

```python
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class MessagesState(TypedDict):
    messages: list[AnyMessage]
```

---

## Reducers

Agora, temos um pequeno problema!

Como discutimos, cada nó retornará um novo valor para nossa chave de estado `messages`.

Porém, esse novo valor [substituirá](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) o valor anterior de `messages`.

À medida que nosso grafo é executado, queremos **acrescentar** mensagens à nossa chave de estado `messages`, não substituí-las.

Podemos resolver isso usando [funções redutoras](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers).

### Como os redutores funcionam:

1. **Comportamento padrão**:
    - Sem redutor especificado: atualizações substituem o valor anterior
    - Exemplo: `messages = novas_mensagens` (sobrescreve)
2. **Redutor `add_messages`**:
    - Especifica que queremos concatenar as listas
    - Exemplo: `messages = messages_anteriores + novas_mensagens`

### Implementação:

Basta anotar nossa chave `messages` com a função redutora `add_messages` como metadado:

```python
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# Definindo o esquema de estado com redutor
class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # Anotação especial

# Criando o grafo
workflow = StateGraph(MessagesState)

```

Isso garante que todas as novas mensagens sejam automaticamente anexadas à lista existente durante a execução do grafo.

Como ter uma lista de mensagens em estado de grafo é muito comum, o LangGraph possui um [`MessagesState`](https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate) pré-construído!

O `MessagesState` é definido:

- Com uma única chave `messages` pré-configurada
- Esta é uma lista de objetos `AnyMessage`
- Ele usa o redutor `add_messages`

Geralmente usaremos o `MessagesState` porque é menos verboso do que definir um `TypedDict` personalizado, como mostrado acima.

```python
from langgraph.graph import MessagesState

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass
```

Para aprofundar um pouco mais, podemos ver como o redutor `add_messages` funciona de forma isolada.  

```python
# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                   ]

# New message to add
new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# Test
add_messages(initial_messages , new_message)
```

Output:

```python
================================== Ai Message ==================================
Name: Model

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, that's right.
================================== Ai Message ==================================
Name: Model

Great, what would you like to learn about.
================================ Human Message =================================
Name: Lance

I want to learn about the best place to see Orcas in the US.
langchain_core.messages.ai.AIMessage
AIMessage(content='One of the best places to see orcas in the United States is the Pacific Northwest, particularly around the San Juan Islands in Washington State. Here are some details:\n\n1. **San Juan Islands, Washington**: These islands are a renowned spot for whale watching, with orcas frequently spotted between late spring and early fall. The waters around the San Juan Islands are home to both resident and transient orca pods, making it an excellent location for sightings.\n\n2. **Puget Sound, Washington**: This area, including places like Seattle and the surrounding waters, offers additional opportunities to see orcas, particularly the Southern Resident killer whale population.\n\n3. **Olympic National Park, Washington**: The coastal areas of the park provide a stunning backdrop for spotting orcas, especially during their migration periods.\n\nWhen planning a trip for whale watching, consider peak seasons for orca activity and book tours with reputable operators who adhere to responsible wildlife viewing practices. Additionally, land-based spots like Lime Kiln Point State Park, also known as “Whale Watch Park,” on San Juan Island, offer great opportunities for orca watching from shore.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 228, 'prompt_tokens': 67, 'total_tokens': 295, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4', 'finish_reason': 'stop', 'logprobs': None}, id='run-57ed2891-c426-4452-b44b-15d0a5c3f225-0', usage_metadata={'input_tokens': 67, 'output_tokens': 228, 'total_tokens': 295, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
{'token_usage': {'completion_tokens': 228,
  'prompt_tokens': 67,
  'total_tokens': 295,
  'completion_tokens_details': {'accepted_prediction_tokens': 0,
   'audio_tokens': 0,
   'reasoning_tokens': 0,
   'rejected_prediction_tokens': 0},
  'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
 'model_name': 'gpt-4o-2024-08-06',
 'system_fingerprint': 'fp_50cad350e4',
 'finish_reason': 'stop',
 'logprobs': None}
[{'name': 'multiply',
  'args': {'a': 2, 'b': 3},
  'id': 'call_lBBBNo5oYpHGRqwxNaNRbsiT',
  'type': 'tool_call'}]
[AIMessage(content='Hello! How can I assist you?', name='Model', id='cd566566-0f42-46a4-b374-fe4d4770ffa7'),
 HumanMessage(content="I'm looking for information on marine biology.", name='Lance', id='9b6c4ddb-9de3-4089-8d22-077f53e7e915'),
 AIMessage(content='Sure, I can help with that. What specifically are you interested in?', name='Model', id='74a549aa-8b8b-48d4-bdf1-12e98404e44e')]
<IPython.core.display.Image object>
================================ Human Message =================================

Hello!
================================== Ai Message ==================================

Hi there! How can I assist you today?
================================ Human Message =================================

Multiply 2 and 3!
================================== Ai Message ==================================
Tool Calls:
  multiply (call_Er4gChFoSGzU7lsuaGzfSGTQ)
 Call ID: call_Er4gChFoSGzU7lsuaGzfSGTQ
  Args:
    a: 2
    b: 3
```

---

## Nosso grafo

Agora, vamos usar o `MessagesState` com um grafo.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
    
# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

```

![image.png](attachment:61621ec1-bcc0-4eb5-9c87-e3725071e8fa:image.png)

---

Se passarmos `"Olá!"`, o LLM responde sem nenhuma chamada de ferramenta.  

```python
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()
```

Output:

```python
================================== Ai Message ==================================
Name: Model

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, that's right.
================================== Ai Message ==================================
Name: Model

Great, what would you like to learn about.
================================ Human Message =================================
Name: Lance

I want to learn about the best place to see Orcas in the US.
langchain_core.messages.ai.AIMessage
AIMessage(content='One of the best places to see orcas in the United States is the Pacific Northwest, particularly around the San Juan Islands in Washington State. Here are some details:\n\n1. **San Juan Islands, Washington**: These islands are a renowned spot for whale watching, with orcas frequently spotted between late spring and early fall. The waters around the San Juan Islands are home to both resident and transient orca pods, making it an excellent location for sightings.\n\n2. **Puget Sound, Washington**: This area, including places like Seattle and the surrounding waters, offers additional opportunities to see orcas, particularly the Southern Resident killer whale population.\n\n3. **Olympic National Park, Washington**: The coastal areas of the park provide a stunning backdrop for spotting orcas, especially during their migration periods.\n\nWhen planning a trip for whale watching, consider peak seasons for orca activity and book tours with reputable operators who adhere to responsible wildlife viewing practices. Additionally, land-based spots like Lime Kiln Point State Park, also known as “Whale Watch Park,” on San Juan Island, offer great opportunities for orca watching from shore.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 228, 'prompt_tokens': 67, 'total_tokens': 295, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4', 'finish_reason': 'stop', 'logprobs': None}, id='run-57ed2891-c426-4452-b44b-15d0a5c3f225-0', usage_metadata={'input_tokens': 67, 'output_tokens': 228, 'total_tokens': 295, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
{'token_usage': {'completion_tokens': 228,
  'prompt_tokens': 67,
  'total_tokens': 295,
  'completion_tokens_details': {'accepted_prediction_tokens': 0,
   'audio_tokens': 0,
   'reasoning_tokens': 0,
   'rejected_prediction_tokens': 0},
  'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
 'model_name': 'gpt-4o-2024-08-06',
 'system_fingerprint': 'fp_50cad350e4',
 'finish_reason': 'stop',
 'logprobs': None}
[{'name': 'multiply',
  'args': {'a': 2, 'b': 3},
  'id': 'call_lBBBNo5oYpHGRqwxNaNRbsiT',
  'type': 'tool_call'}]
[AIMessage(content='Hello! How can I assist you?', name='Model', id='cd566566-0f42-46a4-b374-fe4d4770ffa7'),
 HumanMessage(content="I'm looking for information on marine biology.", name='Lance', id='9b6c4ddb-9de3-4089-8d22-077f53e7e915'),
 AIMessage(content='Sure, I can help with that. What specifically are you interested in?', name='Model', id='74a549aa-8b8b-48d4-bdf1-12e98404e44e')]
<IPython.core.display.Image object>
================================ Human Message =================================

Hello!
================================== Ai Message ==================================

Hi there! How can I assist you today?
================================ Human Message =================================

Multiply 2 and 3!
================================== Ai Message ==================================
Tool Calls:
  multiply (call_Er4gChFoSGzU7lsuaGzfSGTQ)
 Call ID: call_Er4gChFoSGzU7lsuaGzfSGTQ
  Args:
    a: 2
    b: 3
```

O LLM decide usar uma ferramenta quando determina que a entrada ou tarefa requer a funcionalidade fornecida por essa ferramenta.

```python
messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
for m in messages['messages']:
    m.pretty_print()
```

Output:

```python
================================== Ai Message ==================================
Name: Model

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, that's right.
================================== Ai Message ==================================
Name: Model

Great, what would you like to learn about.
================================ Human Message =================================
Name: Lance

I want to learn about the best place to see Orcas in the US.
langchain_core.messages.ai.AIMessage
AIMessage(content='One of the best places to see orcas in the United States is the Pacific Northwest, particularly around the San Juan Islands in Washington State. Here are some details:\n\n1. **San Juan Islands, Washington**: These islands are a renowned spot for whale watching, with orcas frequently spotted between late spring and early fall. The waters around the San Juan Islands are home to both resident and transient orca pods, making it an excellent location for sightings.\n\n2. **Puget Sound, Washington**: This area, including places like Seattle and the surrounding waters, offers additional opportunities to see orcas, particularly the Southern Resident killer whale population.\n\n3. **Olympic National Park, Washington**: The coastal areas of the park provide a stunning backdrop for spotting orcas, especially during their migration periods.\n\nWhen planning a trip for whale watching, consider peak seasons for orca activity and book tours with reputable operators who adhere to responsible wildlife viewing practices. Additionally, land-based spots like Lime Kiln Point State Park, also known as “Whale Watch Park,” on San Juan Island, offer great opportunities for orca watching from shore.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 228, 'prompt_tokens': 67, 'total_tokens': 295, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4', 'finish_reason': 'stop', 'logprobs': None}, id='run-57ed2891-c426-4452-b44b-15d0a5c3f225-0', usage_metadata={'input_tokens': 67, 'output_tokens': 228, 'total_tokens': 295, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
{'token_usage': {'completion_tokens': 228,
  'prompt_tokens': 67,
  'total_tokens': 295,
  'completion_tokens_details': {'accepted_prediction_tokens': 0,
   'audio_tokens': 0,
   'reasoning_tokens': 0,
   'rejected_prediction_tokens': 0},
  'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
 'model_name': 'gpt-4o-2024-08-06',
 'system_fingerprint': 'fp_50cad350e4',
 'finish_reason': 'stop',
 'logprobs': None}
[{'name': 'multiply',
  'args': {'a': 2, 'b': 3},
  'id': 'call_lBBBNo5oYpHGRqwxNaNRbsiT',
  'type': 'tool_call'}]
[AIMessage(content='Hello! How can I assist you?', name='Model', id='cd566566-0f42-46a4-b374-fe4d4770ffa7'),
 HumanMessage(content="I'm looking for information on marine biology.", name='Lance', id='9b6c4ddb-9de3-4089-8d22-077f53e7e915'),
 AIMessage(content='Sure, I can help with that. What specifically are you interested in?', name='Model', id='74a549aa-8b8b-48d4-bdf1-12e98404e44e')]
<IPython.core.display.Image object>
================================ Human Message =================================

Hello!
================================== Ai Message ==================================

Hi there! How can I assist you today?
================================ Human Message =================================

Multiply 2 and 3!
================================== Ai Message ==================================
Tool Calls:
  multiply (call_Er4gChFoSGzU7lsuaGzfSGTQ)
 Call ID: call_Er4gChFoSGzU7lsuaGzfSGTQ
  Args:
    a: 2
    b: 3
```

### 🔁 Integrando tudo no LangGraph

Vamos agora integrar tudo isso ao LangGraph.

Primeiro, como usar mensagens como **estado do grafo**?

Simples: definimos uma classe `MessagesState`, que é um `TypedDict` com uma chave `messages`, que é uma lista de mensagens.

Porém, há um detalhe:

No LangGraph, por padrão, quando fazemos atualizações de estado, o valor da chave é sobrescrito.

Mas nesse caso, **não queremos sobrescrever**, e sim **acrescentar** cada nova mensagem — para manter o histórico da conversa.

Para isso, usamos **funções redutoras** (reducer functions).

Podemos anotar a chave `messages` com uma função redutora que instrui o LangGraph a **acrescentar** novas mensagens em vez de sobrescrever.

Isso é tão comum que já existe uma função redutora pronta chamada `add_messages_reducer`, e inclusive um estado chamado `MessagesState` com essa função embutida.

É só usar.

### Exemplo isolado:

Temos uma lista de mensagens.

Queremos adicionar uma nova mensagem.

Rodamos `add_messages_reducer`, e ela é adicionada corretamente.

Legal, agora sabemos como funciona.

---

### 🔗 Criando o grafo

Vamos criar um grafo simples:

- Definimos o `MessagesState`;
- Criamos um único nó: o modelo de chat com ferramenta ligada;
- Esse nó recebe as mensagens do estado e executa;
- Adicionamos as arestas: de `start` → `llm com ferramenta` → `end`.

Compilamos e visualizamos o grafo.

Perfeito. Começa, passa pelo modelo com ferramenta, termina.

---

### 🧪 Testando o grafo

Agora vamos testar o grafo com dois tipos de entrada:

1. **Entrada simples**: `"hello"`
    
    Executamos o grafo.
    
    A mensagem da IA é: **"Olá, como posso ajudar?"** — exatamente como esperado.
    
2. **Entrada com ferramenta**: `"Qual é 2 vezes 3?"`
    
    Executamos novamente.
    
    A IA **não** responde diretamente, mas retorna uma **chamada de ferramenta**.
    
    Vemos os argumentos e o nome da função `multiply`.
    

---

Tudo funciona como vimos nos testes isolados, mas agora integrado como um grafo no **LangGraph**.

