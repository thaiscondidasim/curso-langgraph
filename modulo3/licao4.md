# Lição 4 - Breakpoint dinamico

### 🧠 **Contexto**

Breakpoints tradicionais (como `interrupt_before`) são úteis para:

- ✅ Aprovação humana (human-in-the-loop)
- 🐞 Depuração e repetição (debugging/replay)
- ✍️ Edição do estado do grafo

Mas... e se quisermos que o **próprio grafo decida interromper a execução com base em uma condição interna**?

> Isso é chamado de breakpoint dinâmico ou interrupção interna.
> 

---

## Revisão

Discutimos as motivações para a interação humana:

(1) `Aprovação` - Podemos interromper nosso agente, apresentar o estado a um usuário e permitir que ele aceite uma ação

(2) `Depuração` - Podemos retroceder o gráfico para reproduzir ou evitar problemas

(3) `Edição` - Você pode modificar o estado

Abordamos os pontos de interrupção como uma forma geral de interromper o gráfico em etapas específicas, o que permite casos de uso como `Aprovação`.

Também mostramos como editar o estado do gráfico e introduzir feedback humano.

## Objetivos

Os pontos de interrupção são definidos pelo desenvolvedor em um nó específico durante a compilação do gráfico.

Mas, às vezes, é útil permitir que o gráfico **se interrompa dinamicamente**!

Este é um ponto de interrupção interno e [pode ser alcançado usando `NodeInterrupt`](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/dynamic_breakpoints/#run-the-graph-with-dynamic-interrupt).

Isso tem alguns benefícios específicos:

(1) você pode fazer isso condicionalmente (de dentro de um nó com base na lógica definida pelo desenvolvedor).

(2) você pode comunicar ao usuário o motivo da interrupção (passando o que você quiser para `NodeInterrupt`).

Vamos criar um gráfico onde um `NodeInterrupt` é lançado com base no comprimento da entrada.

---

### 🚦 Introdução ao `node_interrupt`

Um **`node interrupt`** permite que um **nó do grafo interrompa a execução dinamicamente**, com base em alguma **condição do estado**.

---

### 🔧 Exemplo prático

Criamos um grafo simples com 3 etapas (passos):

1. `step1`: imprime o estado.
2. `step2`: verifica se a **entrada tem mais de 5 caracteres**.
    - Se sim, **lança um `node_interrupt`** e **interrompe a execução**.
3. `step3`: imprime o estado final.

Chamamos o grafo com entrada `"hello world"`.

💡 Como `"hello world"` tem mais de 5 caracteres, o grafo **para na `step2`**.

---

### 🔍 Verificando onde parou

Chamamos:

```python
graph.get_state(thread_id)

```

🔎 Resultado:

- O próximo nó é `step2`
- O estado contém a entrada `"hello world"`
- E há um registro no estado com a mensagem de `node_interrupt`

---

### 🛑 Estamos travados!

Se apenas chamarmos novamente:

```python
graph.stream(None, thread_id)

```

⚠️ Nada acontece. Por quê?

> O estado não mudou, então a condição de interrupção ainda é verdadeira.
> 
> 
> Continuamos presos em `step2`.
> 

---

### 🛠️ Atualizando o estado

Atualizamos o estado com uma entrada menor:

```python
graph.update_state(thread_id, {"input": "hi"})

```

Agora o estado é `"hi"` (2 caracteres).

Executamos novamente com:

```python
graph.stream(None, thread_id)

```

✅ Agora o grafo **continua**:

- Passa por `step2`
- Executa `step3`
- E finaliza com sucesso

---

### 🌐 Usando via LangGraph API (Studio)

1. O grafo com breakpoint dinâmico está rodando no Studio.
2. Conectamos via SDK, listamos os grafos, encontramos `dynamic_breakpoints`.
3. Criamos uma nova `thread` com entrada `"hello world"`.
4. Executamos e o grafo **para em `step2`** como esperado.
5. Usamos `get_state` e vemos a interrupção dinâmica registrada.

---

### ✍️ Atualizando o estado via API

Chamamos:

```python
client.update_state(thread_id, {"input": "hi"})

```

Executamos novamente:

```python
client.stream(None, thread_id)

```

✅ O grafo continua até o final (`step3`) e termina a execução.

---

### ✅ Conclusão

| Conceito | Descrição |
| --- | --- |
| `node_interrupt` | Interrompe dinamicamente o grafo de dentro de um nó |
| Condição de interrupção | Pode ser baseada em qualquer regra sobre o estado |
| `graph.update_state` | Permite modificar o estado para contornar a interrupção |
| `graph.stream(None, thread_id)` | Retoma a execução a partir do ponto em que o grafo parou |
| API `get_state` / `update_state` | Permite inspecionar e alterar o estado remotamente via SDK |

---

### 💡 Aplicações práticas

- Verificação de **condições de segurança** internas antes de continuar.
- Parar automaticamente se o **input estiver malformado**.
- Forçar **revisão manual** se um critério específico for atendido (ex: valor acima de X).

---

Se quiser, posso montar um **código exemplo em Python** com um `node_interrupt` baseado em um campo mais complexo (ex: score de um modelo ou aprovação de usuário), ou até simular isso com fluxo interativo. Deseja que eu crie esse exemplo?