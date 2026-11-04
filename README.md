# Assistente de IA para submissão do CAOL na PNIP

Projeto acadêmico de Machine Learning que propõe o uso do algoritmo **Random Forest** para prever a aprovação ou rejeição de pedidos de **CAOL** (Certificado de Acreditação de Origem Legal), complementando a arquitetura de agente inteligente com **MCP**, **RAG** e **CBR** apresentada por Casanova et al. (2026).

**Curso:** Desenvolvimento de Software Multiplataforma – FATEC Mauá

**Integrantes:**
- Luis Felipe Maia da Costa
- Matheus Nascimento de Holanda
- Miguel Martins Ferreira

---

## 1. Objetivo e problema a ser resolvido

### Contexto

O **CAOL** é o certificado eletrônico que comprova a origem legal dos produtos da pesca no Brasil. Ele é emitido pela **PNIP** (Plataforma Nacional da Indústria do Pescado), sistema oficial do Ministério da Pesca e Aquicultura.

O pedido envolve várias etapas (dados de produção, embarcações, indústria, mapas de bordo e documentos fiscais) e passa por uma aprovação sequencial com quatro papéis: **analista, coordenador, diretor e superintendente**. Isso torna o processo demorado, burocrático e sujeito a erros.

### Problema

O artigo de referência propõe um assistente com IA que:

- responde dúvidas com base nas normas oficiais (**RAG**);
- busca pedidos antigos parecidos e estima a chance de aprovação (**CBR**);
- se conecta à PNIP de forma controlada e auditável (**MCP**).

Os resultados foram bons no geral (**34% menos tempo** de preenchimento e **60% menos erros**), mas a **previsão de aprovação ou rejeição do CBR atingiu precisão de apenas 0,61**. Esse módulo funciona como um k-NN com k = 5: olha os 5 casos mais parecidos e calcula a proporção de aprovados.

Os próprios autores apontam como limitação a **falta de comparação com outros modelos** (baselines).

### Objetivo

Propor e avaliar o **Random Forest** como modelo de classificação para prever se um novo pedido de CAOL tende a ser **aprovado ou rejeitado**, buscando:

- superar a precisão de 0,61 do k-NN atual;
- servir como baseline de comparação, como sugerido pelos autores;
- explicar quais informações do pedido mais influenciam na rejeição;
- manter a decisão final sempre com um analista humano.

---

## 2. Referências

1. **CASANOVA, G. V.; MONTERO, P. B.; SILVA, G. R.; BOTH, T. P.; MACHADO, A.; MARAN, V.** Uma Arquitetura de Software para Integrar Raciocínio Baseado em Casos e Grandes Modelos de Linguagem para Certificação de Produtos Pesqueiros. In: *14º Latin American Symposium on Digital Government (LASDiGov 2026)*, 46º Congresso da Sociedade Brasileira de Computação (CSBC 2026), Gramado/RS, 2026.
   > Artigo base do projeto. Define o problema, a arquitetura MCP + RAG + CBR, a base de 250 casos e o resultado de 0,61 que buscamos melhorar.

2. **BREIMAN, L.** Random Forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.
   > Artigo que apresenta o algoritmo Random Forest, base teórica da solução proposta.

3. **AAMODT, A.; PLAZA, E.** Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches. *AI Communications*, v. 7, n. 1, p. 39–59, 1994.
   > Referência clássica sobre CBR, usada para entender o funcionamento do módulo atual e comparar com a nossa abordagem.

---

## 3. Solução prévia

### Ideia geral

O Random Forest funciona como **vários especialistas votando**. O algoritmo cria várias árvores de decisão, cada uma treinada com uma parte diferente dos dados. Cada árvore analisa o pedido de CAOL e vota em "aprovado" ou "rejeitado", e o resultado final é a **probabilidade de aprovação**.

O modelo **não substitui** o CBR nem o analista. Ele trabalha junto:

| Componente | Papel |
|---|---|
| **RAG** | Explica as normas e cita as fontes oficiais |
| **CBR** | Mostra os pedidos antigos mais parecidos |
| **Random Forest** | Estima a probabilidade de aprovação e mostra o que mais pesa na decisão |
| **Analista** | Toma a decisão final |

### Por que Random Forest

- **Ataca o ponto fraco:** tem potencial para superar a precisão de 0,61 do k-NN.
- **Serve de baseline:** atende à limitação apontada pelos próprios autores.
- **Aceita dados mistos:** lida bem com atributos categóricos (espécie, método de aquisição, região) e numéricos (quantidade produzida).
- **Funciona com poucos dados:** com 250 casos, combinar várias árvores reduz o risco de overfitting.
- **É explicável:** a *feature importance* mostra quais atributos mais levam à rejeição, algo essencial em contexto regulatório.
- **Trata desbalanceamento:** o parâmetro `class_weight` compensa se houver muito mais aprovados do que rejeitados.

### Etapas previstas

1. **Dados:** como a base de 250 CAOLs anonimizados do artigo não é pública, será gerada uma **base simulada** com os mesmos atributos descritos pelos autores: tipo de produto, espécie, método de aquisição, região de origem, quantidade produzida, período de captura e status final.
2. **Pré-processamento:** codificação dos atributos categóricos (One-Hot Encoding) e tratamento de registros incompletos.
3. **Treinamento:** `RandomForestClassifier` do scikit-learn com `class_weight="balanced"`.
4. **Validação:** validação cruzada estratificada (5 folds), adequada para uma base pequena.
5. **Comparação:** avaliação lado a lado com um **k-NN (k = 5)**, que reproduz o comportamento do CBR do artigo, usando acurácia, precisão, recall e F1-score.
6. **Explicabilidade:** análise da *feature importance* para identificar os fatores que mais influenciam na rejeição.

### Exemplo do modelo

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)  # baseline equivalente ao CBR do artigo

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome, modelo in [("Random Forest", rf), ("k-NN (k=5)", knn)]:
    scores = cross_val_score(modelo, X, y, cv=cv, scoring="f1")
    print(f"{nome}: F1 médio = {scores.mean():.2f}")
```

### Integração com a arquitetura

O modelo pode ser exposto como uma nova ferramenta MCP, seguindo o padrão do artigo (somente leitura e com registro de auditoria):

```
Ferramenta: predict_caol_outcome
Entrada:    dados do pedido (espécie, método de aquisição, região, quantidade...)
Saída:      probabilidade de aprovação + principais fatores que influenciaram
Permissão:  somente leitura
Auditoria:  log com hash da entrada e referência da saída
```

---

## Como executar o MVP (fase 1)

O MVP é um único notebook: [`notebooks/mvp_caol_random_forest.ipynb`](notebooks/mvp_caol_random_forest.ipynb).

1. Abra o arquivo no [Google Colab](https://colab.research.google.com) (*File → Upload notebook*) ou pelo botão do GitHub.
2. Rode todas as células (*Runtime → Run all*). Nenhuma instalação extra é necessária (pandas, numpy, scikit-learn e matplotlib já vêm no Colab).

O notebook gera a base simulada (250 registros, `random_state=42`, regras de rejeição documentadas no topo), faz a análise exploratória, treina o Random Forest e o k-NN (k = 5) nos mesmos 5 folds estratificados e mostra a tabela comparativa, a matriz de confusão e a *feature importance*.

### Resultados da primeira execução

Execução local do notebook (scikit-learn 1.9.1; no Colab os valores podem variar levemente), média ± desvio padrão dos 5 folds. A classe positiva é **rejeitado**.

| Modelo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0,628 ± 0,070 | 0,329 ± 0,110 | 0,262 ± 0,060 | 0,282 ± 0,057 |
| k-NN (k=5) | 0,676 ± 0,027 | 0,267 ± 0,165 | 0,129 ± 0,083 | 0,171 ± 0,107 |

Atributos de maior peso no Random Forest: `quantity_kg`, `haul_days` e `species`.

> **Cuidado ao interpretar:** a base é **simulada**, então estes números **não são comparáveis** ao 0,61 do artigo. A comparação válida é Random Forest x k-NN na nossa base. Os atributos mais importantes refletem as regras que nós mesmos usamos para gerar o rótulo, e com apenas 250 registros a diferença entre os modelos pode não ser significativa. Os resultados devem ser regerados no Colab antes de qualquer conclusão do grupo.

---

## Status

🚧 **Fase 1 (MVP no Colab):** notebook pronto, aguardando validação do grupo.
**Fase 2:** migração para `src/` com testes e, opcionalmente, a ferramenta MCP `predict_caol_outcome`. Só começa depois da aprovação dos resultados da fase 1.
