# Definição de Elementos Básicos do Projeto de TCC

**Curso de Extensão em Data Science – ITA**
**Prof. Dr. Johnny Marques**

---

## 1. Questão de Pesquisa

Variáveis macroeconômicas brasileiras, como por exemplo a taxa Selic, o índice de atividade econômica (IBC-Br), o câmbio USD/BRL, a taxa de desocupação e o rendimento médio real, são capazes de antecipar movimentos na taxa de inadimplência de pessoa física e melhorar a acurácia de previsão em comparação com modelos que utilizam apenas o histórico da própria inadimplência?

## 2. Objetivo Geral

Desenvolver um pipeline reprodutível de *nowcasting* da inadimplência de pessoa física no Brasil, integrando dados do Banco Central e do IBGE, e avaliar quantitativamente se a incorporação de variáveis macroeconômicas defasadas melhora a capacidade preditiva frente a modelos univariados.

## 3. Objetivos Específicos

1. **Ingestão automatizada de dados:** construir módulo que coleta séries temporais das APIs do BCB/SGS e do IBGE/SIDRA, armazenando-as em formato Parquet. As fontes de dados que serão utilizadas são:

   **BCB/SGS** (Banco Central do Brasil / Sistema Gerenciador de Séries Temporais):

   - **Selic**: taxa básica de juros da economia brasileira, definida pelo COPOM
   - **IBC-Br**: Índice de Atividade Econômica do Banco Central, usado como proxy mensal do PIB
   - **Câmbio PTAX**: taxa de câmbio oficial BRL/USD divulgada pelo Bacen (média das operações do dia)
   - **Indicadores de crédito**: volume de crédito, inadimplência, spreads bancários, entre outros

   **IBGE/SIDRA** (Sistema IBGE de Recuperação Automática):

   - **Taxa de desocupação**: percentual de pessoas desempregadas, oriundo da PNAD Contínua
   - **Rendimento médio real**: salário médio dos trabalhadores ajustado pela inflação, oriundo da PNAD Contínua

2. **Harmonização temporal:** padronizar todas as séries para frequência mensal — agregando séries diárias pela média e preenchendo séries trimestrais via *forward-fill* — e consolidá-las em um painel único.

3. **Engenharia de features com prevenção de *leakage*:** gerar features baseadas exclusivamente em informação disponível até *t−1*, incluindo defasagens (1, 3 e 6 meses), variações temporais (Δ1m, Δ3m) e termos autorregressivos do próprio alvo.

4. **Modelagem comparativa:** implementar e treinar modelos de complexidade crescente — persistência ingênua, ARIMA univariado, SARIMAX com variáveis exógenas, Elastic Net com regularização via CV, e XGBoost com seleção de hiperparâmetros por *TimeSeriesSplit*.

5. **Validação temporal rigorosa:** avaliar todos os modelos por *backtesting walk-forward* com janela expansível (mínimo de 36 observações iniciais), medindo MAE e RMSE.

6. **Análise de relevância das variáveis e resultados obtidos:** identificar, por meio de importância de features (XGBoost) e coeficientes (Elastic Net), quais variáveis macroeconômicas e quais defasagens são mais informativas para a previsão da inadimplência. Além disso, analisar os resultados obtidos por meio da geração de gráficos e análises estatísticas temporais, determinando qual modelo obteve melhor desempenho.

## 4. Orientador Proposto

**Elton Sbruzzi**
