# Lição 4 - **Filtragem e limpeza de mensagens**

Neste trecho, o foco é mostrar como usar tudo o que foi aprendido até agora no **LangGraph** — incluindo customização de estado, reducers personalizados e múltiplos schemas — para construir um **chatbot com memória de longo prazo**.

Como um chatbot pode acumular muitas mensagens ao longo da conversa, o desafio é **gerenciar esse histórico de mensagens** de maneira eficiente, economizando **tokens** (custo e latência) e **mantendo o contexto relevante**.

---

Até agora, já entendemos bem algumas coisas:

- Como personalizar o estado do grafo;
- Como definir reducers personalizados;
- Como usar múltiplos schemas de estado no grafo.

Agora vamos começar a aplicar esses conceitos **dentro do LangGraph**, e o objetivo final é construir um **chatbot com memória de longo prazo**.

---

### 💬 Trabalhando com mensagens

Antes de tudo, vamos falar sobre **formas de trabalhar com mensagens** (*messages*), porque chatbots usam mensagens como entrada e saída — e **gerenciar isso pode ser desafiador** em algumas situações.

Vamos começar configurando isso:

1. Definimos algumas mensagens — por exemplo, perguntas sobre **mamíferos marinhos**.
2. Podemos passar essa lista de mensagens para um **modelo de chat** (como o GPT).
3. Vamos usar um grafo simples com o `messages state` (estado de mensagens) pré-definido.
4. Definimos o grafo, criamos um nó que invoca o LLM com base nas mensagens do estado.

👉 Pronto! O grafo é executado e vemos:

- A **entrada** com as mensagens;
- A **resposta do modelo**.

```python
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
messages = [AIMessage(f"So you said you were researching ocean mammals?", name="Bot")]
messages.append(HumanMessage(f"Yes, I know about whales. But what others should I learn about?", name="Lance"))

for m in messages:
    m.pretty_print()
```

```python
==================================[1m Ai Message [0m==================================
Name: Bot

So you said you were researching ocean mammals?
================================[1m Human Message [0m=================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
```

Lembre-se de que podemos passá-los para um modelo de bate-papo.

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
llm.invoke(messages)
```

```python
AIMessage(content='Great question, Lance! Ocean mammals are a fascinating group of animals. Here are a few more ocean mammals you might want to learn about:\n\n1. **Dolphins**: These intelligent and social creatures are known for their playful behavior and complex communication skills. There are several species of dolphins, including the bottlenose dolphin and the common dolphin.\n\n2. **Porpoises**: Similar to dolphins but typically smaller and stouter, porpoises are less well-known but equally interesting. The harbor porpoise is one example.\n\n3. **Seals**: These include both true seals (like the harbor seal) and eared seals (which include sea lions and fur seals). They are known for their ability to live both in the water and on land.\n\n4. **Sea Lions**: These are a type of eared seal, easily recognized by their external ear flaps and their ability to "walk" on land using their large flippers.\n\n5. **Walruses**: Known for their distinctive long tusks and whiskers, walruses are social animals that live in Arctic regions.\n\n6. **Manatees and Dugongs**: Often called "sea cows," these gentle herbivores are found in warm coastal waters and rivers. Manatees are found in the Americas and Africa, while dugongs are found in the Indo-Pacific region.\n\n7. **Sea Otters**: Although not exclusively marine, sea otters spend much of their time in the water. They are known for their use of tools to open shellfish.\n\n8. **Polar Bears**: While primarily land animals, polar bears are excellent swimmers and spend a significant amount of time hunting on sea ice.\n\n9. **Sperm Whales**: Known for their large heads and deep diving abilities, sperm whales are the largest of the toothed whales.\n\n10. **Narwhals**: Often called the "unicorns of the sea," these Arctic whales are known for their long, spiral tusk, which is actually an elongated tooth.\n\nEach of these animals has unique adaptations and behaviors that make them fascinating subjects of study. Happy researching!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 434, 'prompt_tokens': 39, 'total_tokens': 473}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_25624ae3a5', 'finish_reason': 'stop', 'logprobs': None}, id='run-513c189f-66e0-4c3c-bdb8-5d59934d10f9-0', usage_metadata={'input_tokens': 39, 'output_tokens': 434, 'total_tokens': 473})
```

Podemos executar nosso modelo de bate-papo em um gráfico simples com `MessagesState`.

```python
from IPython.display import Image, display
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

# Node
def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:33316b4d-7152-43be-8ebf-58eec5c30085:image.png)

```python
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

```python
==================================[1m Ai Message [0m==================================
Name: Bot

So you said you were researching ocean mammals?
================================[1m Human Message [0m=================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
==================================[1m Ai Message [0m==================================

Absolutely, whales are fascinating! But there are many other ocean mammals worth learning about. Here are a few you might find interesting:

1. **Dolphins**: Highly intelligent and social, dolphins are known for their playful behavior and complex communication. There are many species, including the bottlenose dolphin and the orca (killer whale), which is actually the largest member of the dolphin family.

2. **Seals and Sea Lions**: These pinnipeds are often found lounging on beaches or frolicking in the water. Seals tend to be more solitary, while sea lions are social and known for their barking calls.

3. **Manatees and Dugongs**: Often referred to as sea cows, these gentle herbivores graze on seagrasses in shallow coastal areas. Manatees are found in the Atlantic waters, while dugongs are found in the Indo-Pacific region.

4. **Walruses**: Known for their distinctive tusks, walruses are large, social pinnipeds that inhabit the Arctic region. They use their tusks for various purposes, including pulling themselves out of the water and breaking through ice.

5. **Narwhals**: Sometimes called the "unicorns of the sea," narwhals are known for their long, spiral tusks, which are actually elongated teeth. They live in Arctic waters and are relatively elusive.

6. **Porpoises**: Similar to dolphins but generally smaller and with different physical characteristics, porpoises are also highly intelligent and social animals. They are less acrobatic than dolphins and have more triangular dorsal fins.

7. **Sea Otters**: Found along the coasts of the northern and eastern North Pacific Ocean, sea otters are known for their use of tools and their dense fur, which is the thickest of any animal.

8. **Polar Bears**: Though they spend a lot of time on ice, polar bears are excellent swimmers and are considered marine mammals because they depend on the ocean for their primary food source, seals.

Each of these ocean mammals has unique adaptations and behaviors that make them interesting subjects of study. If you're into marine biology, you might find their various ecosystems, social structures, and survival strategies particularly compelling.
```

## Redutor

Um desafio prático ao trabalhar com mensagens é gerenciar conversas de longa duração.

Conversas prolongadas resultam em alto uso de tokens e latência se não formos cuidadosos, porque passamos uma lista crescente de mensagens para o modelo.

Temos algumas formas de lidar com isso.

Primeiro, lembre-se do truque que vimos usando `RemoveMessage` e o redutor `add_messages`.

```python
from langchain_core.messages import RemoveMessage

# Nodes
def filter_messages(state: MessagesState):
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):    
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:db2eca76-9a00-4cc1-a26f-572459f2a1c4:image.png)

```python
# Message list with a preamble
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Invoke
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

```python
==================================[1m Ai Message [0m==================================
Name: Bot

So you said you were researching ocean mammals?
================================[1m Human Message [0m=================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
==================================[1m Ai Message [0m==================================

That's great that you know about whales! There are a variety of other fascinating ocean mammals you might be interested in learning about. Here are a few:

1. **Dolphins**: These highly intelligent and social animals are part of the cetacean family, which also includes whales and porpoises. There are many species of dolphins, including the common bottlenose dolphin and the orca, or killer whale, which is actually the largest member of the dolphin family.

2. **Porpoises**: Similar to dolphins but generally smaller and with different facial structures and teeth. The harbor porpoise is one of the more well-known species.

3. **Seals and Sea Lions**: These pinnipeds are known for their playful nature and agility in water. Seals typically have smaller flippers and no visible ear flaps, while sea lions have larger flippers and visible ear flaps.

4. **Walruses**: Recognizable by their large tusks, whiskers, and significant bulk, walruses are pinnipeds as well and are usually found in Arctic regions.

5. **Manatees and Dugongs**: These gentle giants, often called sea cows, are slow-moving and primarily herbivorous. Manatees are found in the Caribbean and the Gulf of Mexico, while dugongs inhabit the coastal waters of the Indian and western Pacific Oceans.

6. **Sea Otters**: Known for their use of tools to open shells and their thick fur, sea otters are a keystone species in their ecosystems, particularly in kelp forest habitats along the Pacific coast of North America.

7. **Polar Bears**: While not exclusively marine, polar bears depend heavily on the ocean for hunting seals and are excellent swimmers.

Each of these groups has unique adaptations and behaviors that make them fascinating subjects of study. Happy researching!
```

---

### ⚠️ Desafio: conversas longas

Um problema comum ao trabalhar com mensagens é que, em **conversas longas**, o número de tokens aumenta muito.

Imagine um agente com 50 interações: isso pode consumir **muitos tokens**, o que:

- Aumenta a **latência** (tempo de resposta);
- Eleva o **custo financeiro**.

---

### ✂️ Solução: filtragem de mensagens

Vamos ver uma técnica usando `removeMessages`.

Criamos uma função `filter_messages` que:

- Recebe o estado do grafo;
- Remove todas as mensagens, **exceto as duas mais recentes**.

Isso é feito criando objetos `RemoveMessage` com os IDs das mensagens a serem apagadas. O **addMessagesReducer** (reducer padrão usado pelo `messages state`) **reconhece isso automaticamente** e remove as mensagens com os IDs especificados.

### 🚀 Na prática

1. Criamos um grafo simples.
2. Montamos uma lista de mensagens (com "oi, oi" e outras interações).
3. Invocamos o grafo com a função de filtragem.
4. Resultado: apenas as **duas últimas mensagens** são mantidas. As anteriores são eliminadas do estado.

💡 Isso é útil para manter a conversa "enxuta" e relevante.

Por exemplo, basta passar uma lista filtrada: `llm.invoke(mensagens[-1:])` para o modelo.

```python
# Node
def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![image.png](attachment:90a4d09a-a934-44e7-8b6f-111deaa588bd:image.png)

Vamos pegar nossa lista existente de mensagens, anexar a resposta do LLM acima e anexar uma pergunta complementar.

```python
messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me more about Narwhals!", name="Lance"))
```

```python
for m in messages:
    m.pretty_print()
   
```

```python
==================================[1m Ai Message [0m==================================
Name: Bot

Hi.
================================[1m Human Message [0m=================================
Name: Lance

Hi.
==================================[1m Ai Message [0m==================================
Name: Bot

So you said you were researching ocean mammals?
================================[1m Human Message [0m=================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
==================================[1m Ai Message [0m==================================

That's great that you know about whales! There are many other fascinating ocean mammals you can learn about. Here are a few:

1. **Dolphins**: Highly intelligent and social animals, dolphins are known for their playful behavior and sophisticated communication skills. There are many species of dolphins, including the well-known bottlenose dolphin.

2. **Porpoises**: Often confused with dolphins, porpoises are smaller and have different body shapes and teeth. They are generally more reclusive and less acrobatic than dolphins.

3. **Seals**: Seals are part of the pinniped family, which also includes sea lions and walruses. They have streamlined bodies and flippers, making them excellent swimmers. Common types of seals include harbor seals and elephant seals.

4. **Sea Lions**: Similar to seals but with some key differences, sea lions have external ear flaps and can rotate their hind flippers to walk on land. They are also very social and often gather in large groups.

5. **Walruses**: Recognizable by their long tusks and whiskers, walruses are large marine mammals that are found in Arctic regions. They use their tusks to help them climb out of the water and to break through ice.

6. **Manatees and Dugongs**: These gentle giants are often referred to as sea cows. Manatees are found in the Atlantic Ocean, while dugongs are found in the Indian and Pacific Oceans. They are herbivores and spend most of their time grazing on underwater vegetation.

7. **Sea Otters**: Known for their playful behavior and use of tools, sea otters are an important part of the marine ecosystem. They have thick fur to keep them warm in cold waters and are often seen floating on their backs.

8. **Polar Bears**: While not exclusively marine, polar bears spend a significant amount of time in the ocean, particularly in Arctic regions. They are excellent swimmers and rely on sea ice to hunt seals, their primary food source.

9. **Narwhals**: Often called the "unicorns of the sea," narwhals have a long, spiral tusk that is actually an elongated tooth. They are found in Arctic waters and are known for their deep diving abilities.

10. **Orcas (Killer Whales)**: Though they are technically a type of dolphin, orcas are often considered separately due to their size and distinctive black-and-white coloring. They are apex predators and have complex social structures.

Each of these ocean mammals has unique behaviors, adaptations, and ecological roles, making them fascinating subjects for study.
================================[1m Human Message [0m=================================
Name: Lance

Tell me more about Narwhals!
```

```python
# Invoke, using message filtering
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

```python
==================================[1m Ai Message [0m==================================
Name: Bot

Hi.
================================[1m Human Message [0m=================================
Name: Lance

Hi.
==================================[1m Ai Message [0m==================================
Name: Bot

So you said you were researching ocean mammals?
================================[1m Human Message [0m=================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
==================================[1m Ai Message [0m==================================

That's great that you know about whales! There are many other fascinating ocean mammals you can learn about. Here are a few:

1. **Dolphins**: Highly intelligent and social animals, dolphins are known for their playful behavior and sophisticated communication skills. There are many species of dolphins, including the well-known bottlenose dolphin.

2. **Porpoises**: Often confused with dolphins, porpoises are smaller and have different body shapes and teeth. They are generally more reclusive and less acrobatic than dolphins.

3. **Seals**: Seals are part of the pinniped family, which also includes sea lions and walruses. They have streamlined bodies and flippers, making them excellent swimmers. Common types of seals include harbor seals and elephant seals.

4. **Sea Lions**: Similar to seals but with some key differences, sea lions have external ear flaps and can rotate their hind flippers to walk on land. They are also very social and often gather in large groups.

5. **Walruses**: Recognizable by their long tusks and whiskers, walruses are large marine mammals that are found in Arctic regions. They use their tusks to help them climb out of the water and to break through ice.

6. **Manatees and Dugongs**: These gentle giants are often referred to as sea cows. Manatees are found in the Atlantic Ocean, while dugongs are found in the Indian and Pacific Oceans. They are herbivores and spend most of their time grazing on underwater vegetation.

7. **Sea Otters**: Known for their playful behavior and use of tools, sea otters are an important part of the marine ecosystem. They have thick fur to keep them warm in cold waters and are often seen floating on their backs.

8. **Polar Bears**: While not exclusively marine, polar bears spend a significant amount of time in the ocean, particularly in Arctic regions. They are excellent swimmers and rely on sea ice to hunt seals, their primary food source.

9. **Narwhals**: Often called the "unicorns of the sea," narwhals have a long, spiral tusk that is actually an elongated tooth. They are found in Arctic waters and are known for their deep diving abilities.

10. **Orcas (Killer Whales)**: Though they are technically a type of dolphin, orcas are often considered separately due to their size and distinctive black-and-white coloring. They are apex predators and have complex social structures.

Each of these ocean mammals has unique behaviors, adaptations, and ecological roles, making them fascinating subjects for study.
================================[1m Human Message [0m=================================
Name: Lance

Tell me more about Narwhals!
==================================[1m Ai Message [0m==================================

Of course! Narwhals (Monodon monoceros) are fascinating marine mammals that belong to the family Monodontidae, which also includes the beluga whale. They are best known for the long, spiral tusk that protrudes from the head of the males, which has earned them the nickname "unicorns of the sea."

Here are some key facts about narwhals:

### Physical Characteristics
- **Tusk**: The most distinctive feature of the narwhal is the tusk, which is actually an elongated tooth. It can grow up to 10 feet (3 meters) long and is usually found in males, though some females may also develop smaller tusks. The tusk grows in a spiral pattern and is thought to have sensory capabilities, with millions of nerve endings.
- **Body**: Narwhals have a stocky body with a mottled black and white skin pattern. They lack a dorsal fin, which is thought to be an adaptation to swimming under ice.
- **Size**: Adult narwhals typically range from 13 to 20 feet (4 to 6 meters) in length, with males generally being larger than females.
- **Weight**: They can weigh between 1,760 to 3,530 pounds (800 to 1,600 kilograms).

### Habitat and Distribution
- Narwhals are native to the Arctic waters of Canada, Greenland, Norway, and Russia. They are especially common in the Baffin Bay and the waters surrounding Greenland.
- They prefer deep waters and are often found in areas with heavy sea ice.

### Behavior and Diet
- **Diving**: Narwhals are deep divers and can reach depths of up to 5,000 feet (1,500 meters) in search of food. They can hold their breath for up to 25 minutes.
- **Diet**: Their diet primarily consists of fish such as Arctic cod and Greenland halibut, as well as squid and shrimp.
- **Social Structure**: Narwhals are social animals and are often found in groups called pods, which typically consist of 5 to 10 individuals but can sometimes number in the hundreds.

### Reproduction and Lifespan
- Females give birth to a single calf after a gestation period of about 14 to 15 months. Calves are usually born in the spring or early summer.
- Narwhals have a long lifespan and can live up to 50 years, although some individuals may live even longer.

### Conservation Status
- Narwhals are currently classified as "Near Threatened" by the International Union for Conservation of Nature (IUCN). Their main threats include climate change, which affects their sea ice habitat, and human activities such as shipping and oil exploration.
- Indigenous communities in the Arctic have traditionally hunted narwhals for their meat, blubber, and tusks, which are used for various purposes, including art and tools.

### Cultural Significance
- The narwhal's tusk has fascinated humans for centuries and was often sold as a "unicorn horn" in medieval Europe, believed to possess magical properties.
- Narwhals hold significant cultural and economic value for indigenous Arctic communities.

Overall, narwhals are remarkable creatures with unique adaptations that allow them to thrive in some of the planet's harshest environments.
```

O estado contém todas as mensagens.

Mas, vamos analisar o rastreamento do LangSmith para ver que a invocação do modelo usa apenas a última mensagem:

https://smith.langchain.com/public/75aca3ce-ef19-4b92-94be-0178c7a660d9/r

## Reduzir mensagens

Outra abordagem é [reduzir mensagens](https://python.langchain.com/v0.2/docs/how_to/trim_messages/#getting-the-last-max_tokens-tokens) com base em um número fixo de tokens.

Isso limita o histórico de mensagens a um número específico de tokens.

Enquanto a filtragem apenas retorna um subconjunto post-hoc das mensagens entre agentes, a redução limita o número de tokens que um modelo de chat pode usar para responder.

Veja o `trim_messages` abaixo.

```python
from langchain_core.messages import trim_messages

# Node
def chat_model_node(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me where Orcas live!", name="Lance"))
```

```python
# Example of trimming messages
trim_messages(
            messages,
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False
        )
output:
[HumanMessage(content='Tell me where Orcas live!', name='Lance')]
```

```python
# Invoke, using message trimming in the chat_model_node 
messages_out_trim = graph.invoke({'messages': messages})
```

---

### ✅ Outra técnica: usar apenas a última mensagem

Se você **não quiser modificar o estado do grafo**, pode simplesmente:

- Invocar o modelo de linguagem (LLM) com **apenas a última mensagem** da lista.
- O estado continua contendo todas as mensagens, mas o modelo **vê só a mais recente**.

Exemplo:

1. Adicionamos várias mensagens à lista.
2. No nó, usamos só a última para o LLM.
3. A saída contém a resposta baseada apenas nela.

---

### 🔍 Verificação com LangSmith

LangSmith é uma plataforma de observabilidade para fluxos de LLMs (como o LangGraph). Usamos ela para:

- **Visualizar a entrada real do modelo**;
- Garantir que ele viu **apenas a última mensagem**, não o histórico todo.

Resultado:

✅ O modelo recebeu apenas a mensagem “me fale mais sobre narvais”.

🔍 Confirmamos no LangSmith que o restante do histórico **não foi incluído** — exatamente como queríamos.

---

### 🧵 Outra abordagem: **trimming por tokens**

Além de apagar ou ignorar mensagens antigas, podemos **"truncar"** a conversa com base em **limites de tokens**. Isso é útil porque os LLMs têm uma **janela de contexto limitada**, geralmente entre 4.000 e 128.000 tokens.

Usamos a função `truncate_messages` do `langchain_core`, onde você pode:

- Definir o **número máximo de tokens**;
- Escolher a estratégia `last` (começa pelas mensagens mais recentes);
- Permitir truncamentos **parciais** (corta mensagens no meio, se necessário).

---

### ✂️ Testando o trimming

1. Usamos o `truncate_messages` com a opção `allow_partial = False`.
    - Resultado: só a última mensagem inteira é mantida.
2. Com `allow_partial = True`, conseguimos manter **parte da penúltima** também.

Rodamos o grafo, verificamos no LangSmith:

✅ O modelo recebeu apenas a mensagem “me diga onde vivem as orcas”.

🚀 A execução foi rápida, porque o número de tokens era pequeno.

---

### 📌 Conclusão

Essa aula mostrou **3 estratégias diferentes** para lidar com mensagens em chatbots com memória:

| Estratégia | O que faz |
| --- | --- |
| `removeMessages` | Remove mensagens antigas com base nos IDs |
| Usar só a última | Mantém todo o estado, mas o LLM vê só a última mensagem |
| `truncate_messages` | Trunca mensagens com base no número total de tokens |

Essas técnicas são **essenciais** para construir chatbots eficientes e econômicos com LangGraph.

---

Se quiser, posso gerar um exemplo prático com código Python ou LangGraph + LangChain baseado nesses conceitos. Quer seguir por esse caminho?