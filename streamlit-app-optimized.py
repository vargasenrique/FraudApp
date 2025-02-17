import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os

# Configuración de la página
st.set_page_config(
    page_title="Detector de Fraudes",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilo personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .fraud-warning {
        background-color: #ff4b4b;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .safe-transaction {
        background-color: #00cc44;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    """Carga el modelo guardado"""
    try:
        # Buscar el archivo .joblib en la carpeta models
        model_path = os.path.join('models', [f for f in os.listdir('models') if f.endswith('.joblib')][0])
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def crear_campos_formulario():
    """Crea los campos del formulario de entrada"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input("Monto de la Transacción ($)", 
                               min_value=0.0, 
                               step=0.01,
                               help="Ingrese el monto de la transacción en dólares")
        
        category = st.selectbox("Categoría", 
                              ["grocery_pos", "shopping_pos", "entertainment", 
                               "food_dining", "health_fitness", "otros"],
                              help="Seleccione la categoría de la transacción")
        
        merchant = st.text_input("Comerciante",
                               help="Ingrese el nombre del comerciante")

    with col2:
        state = st.selectbox("Estado", 
                           ["NY", "CA", "TX", "FL", "IL", "PA", "otros"],
                           help="Seleccione el estado donde se realizó la transacción")
        
        city = st.text_input("Ciudad",
                           help="Ingrese la ciudad donde se realizó la transacción")
        
        zip_code = st.text_input("Código Postal",
                               help="Ingrese el código postal de la ubicación")

    with col3:
        lat = st.number_input("Latitud", 
                            min_value=-90.0, 
                            max_value=90.0, 
                            value=0.0,
                            help="Ingrese la latitud de la transacción")
        
        long = st.number_input("Longitud", 
                             min_value=-180.0, 
                             max_value=180.0, 
                             value=0.0,
                             help="Ingrese la longitud de la transacción")

    return {
        'amount': amount,
        'merchant': merchant,
        'category': category,
        'state': state,
        'city': city,
        'zip': zip_code,
        'lat': lat,
        'long': long,
        'merch_lat': lat,
        'merch_long': long
    }

def mostrar_resultado(prediccion, datos):
    """Muestra el resultado de la predicción"""
    st.header("Resultado del Análisis")
    
    if prediccion == 1:
        st.markdown("""
            <div class="fraud-warning">
                <h3>⚠️ ALERTA: POSIBLE FRAUDE DETECTADO</h3>
                <p>Esta transacción muestra patrones similares a transacciones fraudulentas.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="safe-transaction">
                <h3>✅ TRANSACCIÓN SEGURA</h3>
                <p>Esta transacción parece ser legítima.</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("Ver Detalles de la Transacción"):
        st.json(datos)
        st.write(f"Evaluación realizada el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    st.title("🔍 Sistema de Detección de Fraudes")
    st.write("Complete el formulario con los datos de la transacción para evaluar si es fraudulenta")

    # Cargar el modelo
    modelo_components = cargar_modelo()
    if modelo_components is None:
        return

    # Extraer componentes del modelo
    modelo = modelo_components['modelo']
    scaler = modelo_components['scaler']
    encoders = modelo_components['encoders']
    selected_features = modelo_components['selected_features']

    # Crear formulario
    with st.form("transaction_form"):
        datos = crear_campos_formulario()
        submitted = st.form_submit_button("Evaluar Transacción")

    if submitted:
        try:
            # Crear DataFrame
            nueva_transaccion = pd.DataFrame([datos])
            
            # Preparar datos
            transaccion_prep = nueva_transaccion.copy()
            
            # Aplicar encoders
            for columna, encoder in encoders.items():
                if columna in transaccion_prep.columns:
                    transaccion_prep[columna] = encoder.transform(transaccion_prep[columna])
            
            # Seleccionar características importantes
            transaccion_prep = transaccion_prep[selected_features]
            
            # Escalar datos
            transaccion_scaled = scaler.transform(transaccion_prep)
            
            # Realizar predicción
            prediccion = modelo.predict(transaccion_scaled)[0]
            
            # Mostrar resultado
            mostrar_resultado(prediccion, datos)

        except Exception as e:
            st.error(f"Error al procesar la transacción: {str(e)}")

if __name__ == "__main__":
    main()
