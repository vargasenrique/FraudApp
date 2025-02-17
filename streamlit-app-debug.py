import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import logging
import requests
from io import BytesIO

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .fraud-warning {
        background-color: #ff4b4b;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .safe-transaction {
        background-color: #00cc44;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metrics-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .debug-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    """Carga el modelo desde Google Drive y muestra información de debug"""
    try:
        # URL de Google Drive (reemplaza con tu ID)
        GDRIVE_FILE_ID = "1FuCvBzGOvN2q8AX_vEBc1vdbcuCj8j4i"
        download_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        
        st.info("📥 Descargando modelo...")
        
        # Descargar el archivo
        response = requests.get(download_url)
        if response.status_code != 200:
            st.error("❌ Error al descargar el modelo de Google Drive")
            return None
            
        # Cargar el modelo desde los bytes descargados
        model_bytes = BytesIO(response.content)
        modelo_components = joblib.load(model_bytes)
        
        # Información de debug
        with st.expander("🔍 Debug: Información del Modelo"):
            st.write("Características requeridas:", modelo_components['selected_features'])
            st.write("Columnas con encoders:", list(modelo_components['encoders'].keys()))
        
        st.success("✅ Modelo cargado exitosamente")
        return modelo_components
        
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

def preparar_datos_para_modelo(datos, selected_features):
    """Prepara los datos del formulario para el modelo"""
    # Crear un DataFrame base con todas las columnas posibles
    df_base = pd.DataFrame({
        'Unnamed: 0': [0],
        'trans_date_trans_time': [datetime.now()],
        'amt': [datos['amount']],
        'first': ['unknown'],
        'gender': ['unknown'],
        'city_pop': [0],
        'dob': ['1970-01-01'],
        'unix_time': [int(datetime.now().timestamp())],
        'merchant': [datos['merchant']],
        'category': [datos['category']],
        'state': [datos['state']],
        'city': [datos['city']],
        'zip': [datos['zip']],
        'lat': [datos['lat']],
        'long': [datos['long']],
        'merch_lat': [datos['merch_lat']],
        'merch_long': [datos['merch_long']]
    })
    
    # Mostrar información de debug
    with st.expander("🔍 Debug: Preparación de Datos"):
        st.write("Columnas en DataFrame creado:", df_base.columns.tolist())
        st.write("Columnas requeridas por el modelo:", selected_features)
        
        # Verificar columnas faltantes
        missing_cols = [col for col in selected_features if col not in df_base.columns]
        if missing_cols:
            st.warning(f"⚠️ Columnas faltantes: {missing_cols}")
    
    return df_base

def validar_datos_entrada(datos):
    """Valida los datos de entrada del formulario"""
    errores = []
    
    if datos['amount'] <= 0:
        errores.append("El monto debe ser mayor que 0")
    
    if not datos['merchant'].strip():
        errores.append("El nombre del comerciante es requerido")
        
    if not datos['city'].strip():
        errores.append("La ciudad es requerida")
        
    if not datos['zip'].strip():
        errores.append("El código postal es requerido")
    
    return errores

def crear_campos_formulario():
    """Crea los campos del formulario de entrada"""
    st.markdown("### Datos de la Transacción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Información Básica")
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
        st.markdown("#### 📍 Ubicación")
        state = st.selectbox("Estado", 
                           ["NY", "CA", "TX", "FL", "IL", "PA", "otros"],
                           help="Seleccione el estado donde se realizó la transacción")
        
        city = st.text_input("Ciudad",
                           help="Ingrese la ciudad donde se realizó la transacción")
        
        zip_code = st.text_input("Código Postal",
                               help="Ingrese el código postal de la ubicación")

    with col3:
        st.markdown("#### 🌎 Coordenadas")
        lat = st.number_input("Latitud", 
                            min_value=-90.0, 
                            max_value=90.0, 
                            value=0.0,
                            format="%.6f",
                            help="Ingrese la latitud de la transacción")
        
        long = st.number_input("Longitud", 
                             min_value=-180.0, 
                             max_value=180.0, 
                             value=0.0,
                             format="%.6f",
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

def mostrar_resultado(prediccion, datos, probabilidad=None):
    """Muestra el resultado de la predicción"""
    st.header("📊 Resultado del Análisis")
    
    if prediccion == 1:
        st.markdown("""
            <div class="fraud-warning">
                <h3>⚠️ ALERTA: POSIBLE FRAUDE DETECTADO</h3>
                <p>Esta transacción muestra patrones similares a transacciones fraudulentas.</p>
                <p>Se recomienda una revisión manual detallada antes de proceder.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="safe-transaction">
                <h3>✅ TRANSACCIÓN SEGURA</h3>
                <p>Esta transacción parece ser legítima según nuestro análisis.</p>
                <p>Puede proceder con la operación normalmente.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Monto de la Transacción",
            value=f"${datos['amount']:,.2f}"
        )
    
    with col2:
        st.metric(
            label="Categoría",
            value=datos['category'].replace('_', ' ').title()
        )
    
    with col3:
        if probabilidad is not None:
            st.metric(
                label="Probabilidad de Fraude",
                value=f"{probabilidad:.1%}"
            )
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📝 Ver Detalles Completos"):
        st.json(datos)
        st.write(f"🕒 Evaluación realizada el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
        submitted = st.form_submit_button("🔍 Evaluar Transacción")

    if submitted:
        # Validar datos
        errores = validar_datos_entrada(datos)
        if errores:
            for error in errores:
                st.error(f"❌ {error}")
            return

        try:
            # Crear DataFrame con datos preparados
            nueva_transaccion = preparar_datos_para_modelo(datos, selected_features)
            
            # Preparar datos
            transaccion_prep = nueva_transaccion.copy()
            
            # Debug: Mostrar estado de los datos antes de transformaciones
            with st.expander("🔍 Debug: Transformación de Datos"):
                st.write("Datos antes de encoding:", transaccion_prep.head())
            
            # Aplicar encoders
            for columna, encoder in encoders.items():
                if columna in transaccion_prep.columns:
                    transaccion_prep[columna] = encoder.transform(transaccion_prep[columna])
            
            # Seleccionar características importantes
            transaccion_prep = transaccion_prep[selected_features]
            
            # Debug: Mostrar estado final de los datos
            with st.expander("🔍 Debug: Datos Finales"):
                st.write("Datos preparados para predicción:", transaccion_prep.head())
            
            # Escalar datos
            transaccion_scaled = scaler.transform(transaccion_prep)
            
            # Realizar predicción
            prediccion = modelo.predict(transaccion_scaled)[0]
            
            # Obtener probabilidad si el modelo lo soporta
            probabilidad = None
            if hasattr(modelo, 'predict_proba'):
                probabilidad = modelo.predict_proba(transaccion_scaled)[0][1]
            
            # Mostrar resultado
            mostrar_resultado(prediccion, datos, probabilidad)
            
            # Logging
            logging.info(f"Predicción realizada: {prediccion} para transacción de ${datos['amount']}")

        except Exception as e:
            logging.error(f"Error al procesar la transacción: {str(e)}")
            st.error(f"❌ Error al procesar la transacción: {str(e)}")
            
            # Mostrar detalles completos del error
            with st.expander("🔍 Debug: Detalles del Error"):
                st.write("Tipo de error:", type(e).__name__)
                st.write("Mensaje:", str(e))
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
