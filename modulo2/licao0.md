# Introdução

icas com **alta qualidade de experiência para o usuário**.

Isso acontece porque os usuários geralmente **esperam que essas aplicações se lembrem de interações anteriores**.

Um exemplo simples são os **chatbots**, que precisam relembrar mensagens anteriores de uma conversa para serem eficazes.

O **LangGraph** oferece bastante controle sobre como a memória funciona na sua aplicação.

Neste módulo, vamos explorar o conceito de **memória** e mostrar **como adicionar persistência ao seu grafo**.

---

### O que vamos ver:

1. Vamos reforçar alguns pontos do Módulo 1 sobre mensagens e sua importância como **estado do grafo**.
2. Vamos reconhecer que o **histórico de mensagens pode ficar longo**, o que gera **alto consumo de tokens**.
3. Vamos discutir algumas formas de **gerenciar esse histórico**, como:
    - **Filtragem**,
    - **Corte (trimming)**,
    - Ou **resumos automáticos** (summarization) para condensar interações longas em resumos mais compactos.

---

Uma das perguntas mais comuns que recebemos é sobre **como integrar diferentes bancos de dados** com a memória de agentes.

O LangGraph torna isso fácil, com suporte desde:

- **Armazenamentos simples em memória (key-value)**,
- Até **bancos de dados externos populares** como **PostgreSQL** ou **SQLite**.

---

Ao final deste módulo, você terá um entendimento completo sobre **memória no LangGraph**

e saberá **como implementar persistência para aprimorar a capacidade dos seus agentes**.

Vamos nessa!