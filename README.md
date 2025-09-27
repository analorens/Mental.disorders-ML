# Classificação de Transtornos Mentais com Rede Neural

## Objetivo do Projeto
O objetivo deste projeto é desenvolver um modelo de aprendizado de máquina capaz de **prever o tipo de transtorno mental de um indivíduo** com base em informações demográficas, histórico clínico e sintomas apresentados. A análise busca **identificar padrões ocultos nos dados** que possam auxiliar em diagnósticos precoces e mais assertivos, contribuindo para a compreensão e prevenção de transtornos mentais.

A escolha deste tema se deu pela **importância crescente da saúde mental** na sociedade atual. Transtornos mentais são frequentemente subdiagnosticados, e a aplicação de técnicas de ciência de dados pode oferecer insights valiosos para profissionais da saúde, políticas públicas e iniciativas de bem-estar. Além disso, trabalhar com esse tipo de dado permite explorar desafios reais de **limpeza, transformação e análise de informações sensíveis e heterogêneas**, tornando o projeto mais relevante e aplicável ao mundo real.

## Sobre o Modelo
Para este projeto, foi escolhida uma **rede neural do tipo Multi-Layer Perceptron (MLP)**, uma abordagem de aprendizado supervisionado que permite capturar **relações complexas e não lineares entre variáveis**. A arquitetura inicial consiste em duas camadas ocultas com 64 e 32 neurônios, utilizando a função de ativação **ReLU** para aprender padrões mais complexos nos dados.  

A rede neural foi escolhida por sua capacidade de lidar com múltiplas features e combinar informações de forma não linear, tornando-a ideal para classificação de transtornos mentais, que geralmente dependem de múltiplos fatores inter-relacionados.

## Base de Dados
- Fonte: [Kaggle - Mental Disorders Dataset](https://www.kaggle.com/datasets/mdsultanulislamovi/mental-disorders-dataset)  
- Contém dados históricos clínicos e informações sobre sintomas de pacientes com diferentes transtornos mentais.  

## Tecnologias Utilizadas
- Python 3.x  
- Pandas e Numpy (manipulação de dados)  
- Scikit-learn (pré-processamento e modelagem)  
- Matplotlib e Seaborn (visualização de dados)  

## Estrutura do Projeto
- `data/` → base de dados original e processada  
- `notebooks/` → notebook principal com ETL, modelagem e avaliação  
- `src/` → scripts Python organizados  
- `assets/` → gráficos e visualizações  
- `README.md` → documentação principal  
- `report.md` → relatório detalhado do projeto  
- `CITATION.cff` → informações de citação do projeto  

## Passos do Projeto
1. **Carregar e explorar dados**: entender a base, identificar padrões e valores ausentes.  
2. **Limpeza e pré-processamento (ETL)**: tratar valores nulos, codificar variáveis categóricas e normalizar features.  
3. **Divisão treino/teste**: separar os dados para avaliação justa do modelo.  
4. **Construção e treinamento da rede neural**: ajustar arquitetura e hiperparâmetros.  
5. **Avaliação do modelo**: métricas de desempenho, matriz de confusão e análise de resultados.  
6. **Ajustes e otimizações**: balanceamento de classes, melhoria da arquitetura e seleção de features.  
7. **Documentação e insights**: gerar relatórios e gráficos para apresentação dos resultados.  
