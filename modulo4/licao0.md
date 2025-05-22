# Introdução

Até agora, você já aprendeu muitas bases úteis. Você já entende o que são *agents* (agentes), conhece as principais abstrações do Landgraph, sabe como usar *memory* (memória) com o LandGraph e também sabe como incorporar *human in the loop* (intervenção humana no processo). Com essas fundações estabelecidas, é hora de reunir tudo isso para criar seu assistente de IA que realmente pode fazer trabalhos úteis por você.

Neste módulo, vamos primeiro apresentar a ideia de *AI role play*, ou seja, criar assistentes pessoais com objetivos específicos relacionados às suas tarefas gerais. Vamos discutir funcionalidades de controle no LandGraph, como **paralelização** e **subgrafos**, que permitem que esses assistentes pessoais realizem coleta de informações usando diversas ferramentas externas, como buscas na web ou recuperação de documentos, para todas as subtarefas atribuídas a eles.

Também vamos abordar o conceito de **MapReduce** no LandGraph para **paralelizar** todo esse processo de coleta de informações entre todos os nossos assistentes, o que acelera bastante o processo. Mostraremos como destilar (resumir e organizar) esse conhecimento coletado em *outputs* (resultados) personalizáveis, como relatórios ou wikis. Em seguida, usaremos o **Hume in the Loop** para revisar esses resultados e, finalmente, conectaremos o Landgraf a ferramentas externas de relatórios, como o Slack.

No conjunto, isso vai resultar em um assistente de uso geral que você pode usar para várias tarefas diferentes do seu trabalho diário que são chatas ou repetitivas. Pode ser algo como resumir reuniões, coletar informações sobre parceiros com quem você vai se encontrar, escrever postagens para redes sociais ou transformar documentação interna em wikis. Na verdade, o objetivo aqui é apresentar essas ideias e fundamentos para que você possa personalizar o assistente e automatizar partes entediantes ou repetitivas do seu trabalho diário.

---

**Explicações dos termos técnicos:**

- **Agent (agente)**: uma entidade autônoma que executa tarefas com base em instruções ou objetivos. Em IA, um agente pode ser uma função, script ou modelo que age de forma "inteligente".
- LandGraph: provavelmente uma variação ou nome estilizado para *LangGraph*, uma biblioteca/framework para criar fluxos de agentes em IA generativa. Ele permite modelar agentes como grafos (nós e conexões).
- **Memory (memória)**: componente que armazena contexto ou histórico de interações para que o agente/IA possa lembrar de informações anteriores em uma conversa ou tarefa.
- **Human in the loop**: técnica onde humanos intervêm em algum ponto do processo automatizado (como revisão de respostas geradas por IA) para melhorar a qualidade ou corrigir erros.
- **AI role play**: criação de personagens ou assistentes com papéis específicos e comportamentos próprios para simular situações ou atender tarefas com personalização.
- **Paralelização**: técnica de executar várias tarefas ao mesmo tempo para ganhar velocidade, especialmente útil em coleta de dados ou processamento pesado.
- **Subgrafos**: partes menores de um grafo maior que executam funções específicas, permitindo modularidade e controle granular do fluxo de trabalho.
- **MapReduce**: modelo de processamento de dados em larga escala que divide tarefas (Map) e depois junta os resultados (Reduce), muito usado em computação distribuída.
- **Hume in the Loop**: parece ser uma ferramenta ou técnica específica para inserir humanos na revisão ou refinamento dos resultados gerados por IA.
- **Slack**: ferramenta de comunicação corporativa que pode ser integrada com sistemas automatizados para enviar relatórios, alertas ou mensagens.