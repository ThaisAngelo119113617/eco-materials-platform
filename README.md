# eco-materials-platform
# EcoMaterials - Plataforma de Descoberta Sustentável de Materiais

## Sobre o Projeto

**EcoMaterials** é uma plataforma interativa desenvolvida em Python utilizando Streamlit, projetada para facilitar o gerenciamento, análise e descoberta de materiais sustentáveis. Este projeto tem como objetivo principal fornecer uma aplicação prática para **aprender e explorar conceitos de Machine Learning (ML)** e seus algoritmos, enquanto promove a sustentabilidade.

A ferramenta utiliza algoritmos de aprendizado de máquina para classificar e sugerir materiais com base em propriedades físicas, custo e impacto ambiental. O projeto serve como um exemplo real de como ML pode ser aplicado para resolver problemas do mundo real, como a escolha de materiais mais sustentáveis.

---

### Algoritmos de Machine Learning Utilizados

1. **K-Means Clustering**
   - **Descrição**:
     - Algoritmo de aprendizado não supervisionado utilizado para agrupar dados com base na similaridade entre os pontos de dados.
   - **Funcionamento**:
     - O K-Means particiona os dados em `k` clusters:
       - Inicialmente escolhe `k` centroides aleatórios.
       - Atribui os dados ao cluster mais próximo (com base na distância Euclidiana).
       - Recalcula os centroides iterativamente até atingir um ponto de convergência.
   - **Uso no projeto**:
     - Identificar padrões nos materiais agrupando-os com base em densidade, resistência, custo e impacto ambiental.

2. **Random Forest Classifier**
   - **Descrição**:
     - Algoritmo supervisionado que utiliza múltiplas árvores de decisão para realizar classificações com alta precisão.
   - **Funcionamento**:
     - O modelo cria várias árvores de decisão independentes em subconjuntos aleatórios dos dados.
     - Cada árvore fornece uma previsão, e o resultado final é definido pela votação da maioria (classificação) ou pela média (regressão).
   - **Uso no projeto**:
     - Classificar materiais como "sustentáveis" ou "não sustentáveis" com base em atributos como custo, emissão de CO₂ e propriedades físicas.
     - Avaliar a qualidade do modelo por meio da métrica de acurácia.

---

### Benefícios Educacionais

Este projeto foi desenvolvido com foco em aprendizado prático, permitindo que desenvolvedores e entusiastas:

- **Entendam conceitos de aprendizado supervisionado e não supervisionado.**
- **Explorem algoritmos como K-Means e Random Forest em cenários do mundo real.**
- **Visualizem clusters e classificações** em um contexto prático, facilitando o entendimento.
- **Experimentem com filtros, ajustes e modelos** para observar como diferentes escolhas de dados e configurações impactam os resultados.

---

## Funcionalidades

- **Adicionar Materiais**: Insira dados sobre materiais, incluindo densidade, resistência, custo e impacto ambiental.
- **Consultar Materiais**: Filtre e visualize os materiais cadastrados com base em critérios como densidade, resistência, custo e sustentabilidade.
- **Análise Sustentável**: Utilize algoritmos como K-Means para identificar clusters de materiais e Random Forest para classificação de sustentabilidade.
- **Importar CSV**: Carregue dados de materiais diretamente de arquivos CSV.
- **Limpar Banco de Dados**: Restaure o banco de dados removendo todos os materiais cadastrados.
- **Método do Cotovelo**: Ajude a determinar o número ideal de clusters com base na inércia.

## Tecnologias Utilizadas

- **Python** (3.7 ou superior)
- **Bibliotecas**:
  - [Streamlit](https://streamlit.io/)
  - [Pandas](https://pandas.pydata.org/)
  - [Scikit-learn](https://scikit-learn.org/)
  - [Matplotlib](https://matplotlib.org/)
- **Banco de Dados**: SQLite

## Pré-requisitos

Antes de começar, certifique-se de ter o Python instalado. Você pode instalá-lo [aqui](https://www.python.org/).
Instale também as dependências : pandas , scikit-learn , matplotlib , sqlite3 , streamlit.

## Execução
```bash
streamlit run eco_materials_app.py
```
Acesse o aplicativo no navegador: http://localhost:8501
