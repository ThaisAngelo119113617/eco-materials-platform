import sqlite3
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --- Funções auxiliares ---
def create_database():
    """Cria a tabela no banco de dados SQLite se não existir."""
    try:
        conn = sqlite3.connect("eco_materials.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY,
                name TEXT,
                density REAL,
                strength REAL,
                co2_emission REAL,
                cost_per_kg REAL,
                sustainability INTEGER
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Erro ao criar o banco de dados: {e}")
    finally:
        conn.close()

def insert_material(name, density, strength, co2_emission, cost_per_kg, sustainability):
    """Insere um novo material no banco de dados."""
    try:
        conn = sqlite3.connect("eco_materials.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO materials (name, density, strength, co2_emission, cost_per_kg, sustainability)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, density, strength, co2_emission, cost_per_kg, sustainability))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Erro ao inserir material: {e}")
    finally:
        conn.close()

def fetch_materials():
    """Obtém todos os materiais do banco de dados."""
    try:
        conn = sqlite3.connect("eco_materials.db")
        df = pd.read_sql_query("SELECT * FROM materials", conn)
        return df
    except sqlite3.Error as e:
        print(f"Erro ao buscar materiais: {e}")
        return pd.DataFrame()  # Retorna DataFrame vazio em caso de erro
    finally:
        conn.close()

def clear_database():
    """Remove todos os dados da tabela materials."""
    try:
        conn = sqlite3.connect("eco_materials.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM materials")
        conn.commit()
        return "Banco de dados limpo com sucesso!"
    except sqlite3.Error as e:
        return f"Erro ao limpar banco de dados: {e}"
    finally:
        conn.close()

def import_csv(file):
    """Importa dados de um arquivo CSV para o banco de dados."""
    try:
        df = pd.read_csv(file)
        conn = sqlite3.connect("eco_materials.db")
        cursor = conn.cursor()
        with conn:  # Transações em lote para maior eficiência
            cursor.executemany("""
                INSERT INTO materials (name, density, strength, co2_emission, cost_per_kg, sustainability)
                VALUES (?, ?, ?, ?, ?, ?)
            """, df.values.tolist())
        return "Dados importados com sucesso!"
    except Exception as e:
        return f"Erro ao importar CSV: {e}"

def clean_and_prepare_data():
    """Limpa e prepara os dados do banco de dados."""
    df = fetch_materials()
    if df.empty:
        return df  # Retorna vazio se não houver dados

    # Remover valores ausentes e duplicatas
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Normalizar colunas numéricas
    scaler = MinMaxScaler()
    df[['density', 'strength', 'co2_emission', 'cost_per_kg']] = scaler.fit_transform(
        df[['density', 'strength', 'co2_emission', 'cost_per_kg']]
    )
    return df

def determine_optimal_clusters(X_scaled):
    """Determina o número ideal de clusters usando o método do cotovelo."""
    silhouette_scores = []
    k_values = range(2, 10)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Exibir gráfico do método do cotovelo
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Método do Cotovelo")
    plt.grid(True)
    st.pyplot(plt)

def classify_materials():
    """Classifica materiais com KMeans e Random Forest."""
    data = clean_and_prepare_data()
    if data.empty:
        return "Nenhum material disponível para análise. Adicione materiais ao banco de dados."
    
    # Selecionar atributos para clustering
    X = data[['density', 'strength', 'co2_emission', 'cost_per_kg']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determinar o número ideal de clusters
    # st.subheader("Determinar Número Ideal de Clusters")
    # determine_optimal_clusters(X_scaled)

    # Clusterizar com KMeans (fixando k=3 como exemplo)
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Classificar sustentabilidade com Random Forest
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, data['sustainability'], test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)

    st.write(f"Acurácia do modelo Random Forest: {accuracy:.2f}")
    return data

def plot_materials_with_labels(data):
    """Gera um gráfico de impacto ambiental vs custo com legendas dos materiais."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        data['co2_emission'], 
        data['cost_per_kg'], 
        c=data['Cluster'], 
        cmap='viridis', 
        s=100
    )
    for i, row in data.iterrows():
        plt.text(row['co2_emission'], row['cost_per_kg'], row['name'], fontsize=9, ha='right')
    plt.xlabel("Impacto Ambiental (CO₂)")
    plt.ylabel("Custo ($)")
    plt.title("Impacto Ambiental vs Custo com Clusters")
    plt.colorbar(scatter, label='Clusters')
    plt.grid(True)
    st.pyplot(plt)

# --- Interface Streamlit ---
st.title("EcoMaterials - Plataforma de Descoberta Sustentável de Materiais")
create_database()

menu = st.sidebar.selectbox("Menu", [
    "Início", "Adicionar Material", "Consultar Materiais", 
    "Análise Sustentável", "Importar CSV", "Limpar Banco de Dados"
])

if menu == "Início":
    st.write("Bem-vindo à plataforma EcoMaterials!")

elif menu == "Adicionar Material":
    st.subheader("Adicionar Material ao Banco de Dados")
    name = st.text_input("Nome do Material")
    density = st.number_input("Densidade (kg/m³)")
    strength = st.number_input("Resistência (MPa)")
    co2_emission = st.number_input("Emissão de CO₂ (kg)")
    cost_per_kg = st.number_input("Custo por Kg ($)")
    sustainability = st.selectbox("Sustentabilidade (0=Não, 1=Sim)", [0, 1])
    if st.button("Adicionar"):
        insert_material(name, density, strength, co2_emission, cost_per_kg, sustainability)
        st.success(f"Material {name} adicionado com sucesso!")

elif menu == "Consultar Materiais":
    st.subheader("Consulta de Materiais")
    data = fetch_materials()
    if not data.empty:
        st.write("Filtros para consulta:")
        
        # Filtros de densidade, resistência, custo e sustentabilidade
        min_density = st.slider("Mínimo de Densidade (kg/m³)", float(data['density'].min()), float(data['density'].max()), float(data['density'].min()))
        max_density = st.slider("Máximo de Densidade (kg/m³)", float(data['density'].min()), float(data['density'].max()), float(data['density'].max()))
        
        min_strength = st.slider("Mínimo de Resistência (MPa)", float(data['strength'].min()), float(data['strength'].max()), float(data['strength'].min()))
        max_strength = st.slider("Máximo de Resistência (MPa)", float(data['strength'].min()), float(data['strength'].max()), float(data['strength'].max()))
        
        min_cost = st.slider("Mínimo de Custo ($)", float(data['cost_per_kg'].min()), float(data['cost_per_kg'].max()), float(data['cost_per_kg'].min()))
        max_cost = st.slider("Máximo de Custo ($)", float(data['cost_per_kg'].min()), float(data['cost_per_kg'].max()), float(data['cost_per_kg'].max()))
        
        sustainability_filter = st.selectbox("Sustentabilidade", options=["Todos", "Sustentável", "Não Sustentável"])

        # Aplicar os filtros
        filtered_data = data[
            (data['density'] >= min_density) & (data['density'] <= max_density) &
            (data['strength'] >= min_strength) & (data['strength'] <= max_strength) &
            (data['cost_per_kg'] >= min_cost) & (data['cost_per_kg'] <= max_cost)
        ]
        if sustainability_filter != "Todos":
            sustainability_value = 1 if sustainability_filter == "Sustentável" else 0
            filtered_data = filtered_data[filtered_data['sustainability'] == sustainability_value]

        st.write("Materiais filtrados:")
        st.dataframe(filtered_data)
    else:
        st.write("Nenhum material cadastrado no momento.")

elif menu == "Análise Sustentável":
    st.subheader("Análise de Sustentabilidade")
    classified_data = classify_materials()
    if isinstance(classified_data, str):
        st.write(classified_data)
    else:
        # Filtros para análise
        st.write("Filtros para os materiais:")
        max_co2 = st.slider("Máximo de Impacto Ambiental (CO₂)", 0.0, 1.0, 0.5)
        max_cost = st.slider("Máximo de Custo ($)", 0.0, 1.0, 0.5)

        # Aplicar filtros
        filtered_data = classified_data[
            (classified_data['co2_emission'] <= max_co2) &
            (classified_data['cost_per_kg'] <= max_cost)
        ]

        # # Gráfico do Método do Cotovelo
        # st.write("Gráfico do Método do Cotovelo")
        # inertia = []
        # X_scaled = StandardScaler().fit_transform(classified_data[['density', 'strength', 'co2_emission', 'cost_per_kg']])
        # for i in range(1, 11):
        #     kmeans = KMeans(n_clusters=i, random_state=42)
        #     kmeans.fit(X_scaled)
        #     inertia.append(kmeans.inertia_)

        # plt.figure(figsize=(8, 5))
        # plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
        # plt.title("Método do Cotovelo")
        # plt.xlabel("Número de Clusters")
        # plt.ylabel("Inércia")
        # st.pyplot(plt)

        if filtered_data.empty:
            st.write("Nenhum material encontrado para os filtros aplicados.")
        else:
            st.dataframe(filtered_data)
            st.write("Gráfico de Impacto Ambiental vs Custo com Identificação de Materiais")
            plot_materials_with_labels(filtered_data)


elif menu == "Importar CSV":
    st.subheader("Importar Dados de um Arquivo CSV")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
    if uploaded_file is not None:
        message = import_csv(uploaded_file)
        st.success(message)

elif menu == "Limpar Banco de Dados":
    st.subheader("Limpar Banco de Dados")
    if st.button("Limpar Todos os Dados"):
        message = clear_database()
        st.success(message)
