# Lição 1 - Motivação

Bem-vindo ao Módulo 1. Antes de começarmos a mergulhar no código, quero apresentar brevemente as motivações por trás do LangGraph e também fornecer um roteiro geral do curso, para que você tenha uma noção do que esperar.

Primeiramente, um modelo de linguagem sozinho é algo limitado. Ele não tem acesso a ferramentas. Ele não tem acesso a contexto externo, como documentação. Ele não consegue, sozinho, realizar fluxos de trabalho com várias etapas.

![alt text](https://img.notionusercontent.com/s3/prod-files-secure%2F43a77b18-9895-4da3-a6fa-c2bf840e8ecd%2Fcf4560be-5e77-4b7c-b898-bb3ad959c98a%2Fimage.png/size/w=2000?exp=1747947839&sig=0gHOuAg1GfTJ6ufSeeTW3RpzwwS7xYZpFV_NIpm31jA&id=1f9bd7fa-42ba-80c7-bf1f-dbd0a4a45e2f&table=block&userId=9bacf5ce-08bc-45e1-b61a-bd4328577f69)


Por isso, muitas aplicações com LLMs (modelos de linguagem grandes) utilizam algum tipo de **fluxo de controle** com etapas antes e depois das chamadas ao LLM. Essas etapas podem incluir chamadas a ferramentas, recuperação de informações, entre outras. Esse fluxo de controle forma o que chamamos de **cadeia (chain)**.

![image.png](https://img.notionusercontent.com/s3/prod-files-secure%2F43a77b18-9895-4da3-a6fa-c2bf840e8ecd%2Fec968913-a8b4-4611-b5b1-d5d1e42d4062%2Fimage.png/size/w=2000?exp=1747947878&sig=nUao33gjxWTY-0AqN08OTSksR62s97oVcSp8rCYc4T8&id=1f9bd7fa-42ba-8089-8816-d19a5096d751&table=block&userId=9bacf5ce-08bc-45e1-b61a-bd4328577f69)

Você provavelmente já ouviu esse termo “chain”. Pode pensar nisso como um conjunto de etapas que ocorrem antes e depois da chamada ao LLM. A vantagem das chains é que são muito **confiáveis** – o mesmo conjunto de etapas acontece toda vez que a chain é executada.

![image.png](https://img.notionusercontent.com/s3/prod-files-secure%2F43a77b18-9895-4da3-a6fa-c2bf840e8ecd%2F104a27ad-c13c-4a22-bdce-0f0c382201bc%2Fimage.png/size/w=2000?exp=1747947895&sig=c4GP4WfUGI9x0QOo0w8KYH2brW7YIvW2rxm_8eIjdA4&id=1f9bd7fa-42ba-8015-ac4b-f94a18295794&table=block&userId=9bacf5ce-08bc-45e1-b61a-bd4328577f69)

Mas queremos também criar sistemas com LLMs que possam escolher seu próprio fluxo de controle, dependendo do problema enfrentado. E isso é o que chamamos de **agente**. Essa é uma definição simples de agente: é um fluxo de controle **definido pelo próprio LLM**.

![image.png](https://img.notionusercontent.com/s3/prod-files-secure%2F43a77b18-9895-4da3-a6fa-c2bf840e8ecd%2F59931926-727f-45e4-b250-bd6ca4a9717c%2Fimage.png/size/w=2000?exp=1747947922&sig=0nUxr744WNy1G6CW7pCBWtV3umsLoQOOR9NCy1z7sMU&id=1f9bd7fa-42ba-803d-afc1-eb9bd02a3ddf&table=block&userId=9bacf5ce-08bc-45e1-b61a-bd4328577f69)

Então, temos:

- **Chains**: fluxo fixo, definido pelo desenvolvedor.
- **Agentes:** fluxo dinâmico, definido pelo LLM.

Agora, o interessante é que existem **vários tipos de agentes**. Você pode pensar nisso como um **controle gradual** – do menor para o maior nível de liberdade dado ao LLM. Por exemplo:

- Em um nível baixo, temos os **roteadores (routers)**. Aqui, o LLM escolhe apenas entre algumas opções fixas.
    - Exemplo: do passo 1, ele pode seguir para o passo 2 ou 3, com base em uma decisão simples.
- No outro extremo, temos **agentes totalmente autônomos**, que podem escolher qualquer sequência de etapas, ou até **gerar suas próprias etapas** com base nos recursos disponíveis.

Agora vem o desafio: à medida que **aumentamos o controle dado ao LLM**, a **confiabilidade** do sistema **tende a cair**. Ou seja, um roteador simples é mais confiável que um agente autônomo.

É aí que entra a **motivação do LangGraph**: ele foi criado para **melhorar essa curva de confiabilidade**, permitindo que você crie agentes mais flexíveis **sem perder a confiabilidade**.

Uma intuição importante: em muitas aplicações, queremos **combinar a intuição do desenvolvedor com o controle do LLM**. Por exemplo, você pode definir passos fixos no fluxo (começa no passo 1 e termina no passo 2) e deixar que o LLM decida o que acontece no meio. Isso é feito com **grafos (graphs)**.

Grafos são muito úteis:

- **Nós (nodes)** representam os passos da aplicação (ex: chamada a ferramenta, busca de dados).
- **Arestas (edges)** representam a conexão entre esses passos.

O LangGraph permite flexibilidade total para organizar esses nós e arestas, e vamos explorar isso com mais detalhes ao longo do curso.

Existem quatro **pilares do LangGraph**:

1. **Persistência** – manter estado e histórico.
2. **Transmissão em tempo real (streaming)** – receber respostas do LLM em tempo real.
3. **Human-in-the-loop** – incluir humanos no processo para validar ou modificar ações.
4. **Alta controlabilidade** – ajustar de forma precisa o que o LLM pode ou não fazer.

Esses pilares serão aprofundados nos módulos do curso.

O LangGraph também vem com um **IDE visual (Studio)**, que é um ambiente para você visualizar e depurar seus agentes. Vamos usá-lo bastante, e ele será uma ferramenta essencial para testes e observação.

Além disso, o LangGraph **funciona muito bem com o LangChain** – uma biblioteca open-source que oferece várias integrações, como com vetores e diferentes LLMs. Em muitos casos, vamos usar componentes do LangChain dentro dos fluxos do LangGraph.

**Exemplo simples:**

Um sistema de RAG (geração aumentada por recuperação) que:

1. Recupera documentos de uma base vetorial.
2. Usa o LLM para responder com base nesses documentos.

A recuperação pode usar um **vetor do LangChain**, e a chamada ao LLM também pode usar uma integração do LangChain — **mas isso não é obrigatório**.

O LangChain oferece **uma interface comum** para vários modelos de LLMs, facilitando a troca entre eles. Mas você pode usar o LangGraph **sem o LangChain** se quiser.

---

### 🔸 **Roteiro do Curso:**

### **Módulo 1 – Fundamentos**

- Introdução ao LangGraph Studio.
- Apresentação das abstrações principais do LangGraph.
- Criação de **dois agentes**:
    - Um **roteador (router)**.
    - Um agente genérico que **chama ferramentas**.

### **Módulo 2 – Memória**

- Criação de um **chatbot com memória** para manter o contexto em conversas longas.
- Uso de **persistência** e **memória** no LangGraph.

### **Módulo 3 – Human-in-the-loop**

- Como incluir humanos no fluxo para aprovar ou editar o estado do agente.
- Adição de human-in-the-loop no agente do módulo 1.
- Introdução ao **streaming**.

### **Módulo 4 – Projeto Final**

- Criação de um **assistente de pesquisa complexo e personalizável**.
- Uso de:
    - Human-in-the-loop.
    - Paralelismo (MapReduce).
    - Streaming.
    - Memória e persistência.

Módulos 1 a 3 são mais **fundamentais**. Se você já for um usuário avançado, pode pular direto para o Módulo 4. Mas, se quiser dominar os conceitos, é recomendável seguir do início.

---

### 🧠 **Explicações de termos técnicos:**