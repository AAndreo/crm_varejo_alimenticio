## 📁 Tabela: customers

| Variável               | Tipo        | Categoria            | Descrição                                                                 |
|------------------------|------------|---------------------|---------------------------------------------------------------------------|
| ID                     | Integer     | Identificação       | Identificador único do cliente                                            |
| Year_Birth             | Integer     | Demográfica         | Ano de nascimento do cliente                                              |
| Education              | Categorical | Demográfica         | Nível educacional do cliente                                              |
| Marital                | Categorical | Demográfica         | Estado civil do cliente                                                   |
| Income                 | Float       | Demográfica         | Renda anual do cliente                                                    |
| Kidhome                | Integer     | Familiar            | Número de crianças no domicílio                                           |
| Teenhome               | Integer     | Familiar            | Número de adolescentes no domicílio                                       |
| DtCustomer             | Date        | Relacionamento      | Data de cadastro do cliente                                               |
| Recency                | Integer     | Comportamento       | Dias desde a última compra                                                |
| Complain               | Binary      | Satisfação          | 1 se o cliente reclamou nos últimos 2 anos, 0 caso contrário               |

---

## 💰 Variáveis de Consumo (últimos 2 anos)

| Variável               | Tipo    | Categoria     | Descrição                                      |
|------------------------|--------|---------------|------------------------------------------------|
| MntWines              | Float  | Consumo       | Valor gasto com vinhos                         |
| MntFruits             | Float  | Consumo       | Valor gasto com frutas                         |
| MntMeatProducts       | Float  | Consumo       | Valor gasto com carnes                         |
| MntFishProducts       | Float  | Consumo       | Valor gasto com peixes                         |
| MntSweetProducts      | Float  | Consumo       | Valor gasto com doces                          |
| MntGoldProducts       | Float  | Consumo       | Valor gasto com produtos premium (gold)        |

---

## 🛒 Variáveis de Canal de Compra

| Variável               | Tipo    | Categoria     | Descrição                                      |
|------------------------|--------|---------------|------------------------------------------------|
| NumWebPurchases        | Integer| Canal         | Número de compras pelo site                    |
| NumCatalogPurchases    | Integer| Canal         | Número de compras por catálogo                 |
| NumStorePurchases      | Integer| Canal         | Número de compras em loja física               |
| NumDealsPurchases      | Integer| Promoção      | Número de compras com desconto                 |
| NumWebVisitsMonth      | Integer| Engajamento   | Número de visitas ao site no último mês        |

---

## 📣 Histórico de Campanhas

| Variável               | Tipo   | Categoria     | Descrição                                       |
|------------------------|--------|---------------|-------------------------------------------------|
| AcceptedCmp1          | Binary | Marketing     | Aceitou campanha 1 (1 = sim, 0 = não)          |
| AcceptedCmp2          | Binary | Marketing     | Aceitou campanha 2                              |
| AcceptedCmp3          | Binary | Marketing     | Aceitou campanha 3                              |
| AcceptedCmp4          | Binary | Marketing     | Aceitou campanha 4                              |
| AcceptedCmp5          | Binary | Marketing     | Aceitou campanha 5                              |

---

## 🎯 Variável Target

| Variável   | Tipo   | Categoria     | Descrição                                                |
|------------|--------|---------------|----------------------------------------------------------|
| Response   | Binary | Target        | 1 se aceitou a última campanha, 0 caso contrário         |

---

## ⚙️ Variáveis Técnicas

| Variável        | Tipo   | Categoria     | Descrição                              |
|-----------------|--------|---------------|----------------------------------------|
| Z_CostContact   | Float  | Técnica       | Custo de contato                       |
| Z_Revenue       | Float  | Técnica       | Receita associada                      |

---

## 🔧 Variáveis Derivadas (Feature Engineering)

| Variável              | Descrição |
|----------------------|----------|
| Total_Spend          | Soma de todos os gastos |
| Total_Purchases      | Soma de todas as compras |
| Avg_Ticket           | Ticket médio |
| Recency_Group        | Faixa de recência |
| Income_Group         | Faixa de renda |
