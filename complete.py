import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
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
    """Carga el modelo desde Google Drive"""
    try:
        # URL de Google Drive (reemplazar con tu ID)
        GDRIVE_FILE_ID = "1FuCvBzGOvN2q8AX_vEBc1vdbcuCj8j4i"
        download_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        
        st.info("📥 Descargando modelo...")
        
        response = requests.get(download_url)
        if response.status_code != 200:
            st.error("❌ Error al descargar el modelo de Google Drive")
            return None
            
        model_bytes = BytesIO(response.content)
        modelo_components = joblib.load(model_bytes)
        
        # Debug: Información del modelo
        with st.expander("🔍 Debug: Información del Modelo"):
            st.write("Características requeridas:", modelo_components['selected_features'])
            st.write("Columnas con encoders:", list(modelo_components['encoders'].keys()))
            st.write("Información del Scaler:", {
                'mean': dict(zip(modelo_components['selected_features'], 
                               modelo_components['scaler'].mean_)),
                'scale': dict(zip(modelo_components['selected_features'], 
                                modelo_components['scaler'].scale_))
            })
        
        st.success("✅ Modelo cargado exitosamente")
        return modelo_components
        
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

def crear_campos_formulario():
    """Crea los campos del formulario de entrada con los campos requeridos por el modelo"""
    st.markdown("### Datos de la Transacción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Información de la Transacción")
        amount = st.number_input("Monto de la Transacción ($)", 
                               min_value=0.0, 
                               step=0.01)
        
        category = st.selectbox("Categoría", 
                              ["grocery_pos", "shopping_pos", "entertainment", 
                               "food_dining", "health_fitness", "otros"])
        
        transaction_date = st.date_input("Fecha de Transacción", 
                                       value=datetime.now())
        
        transaction_time = st.time_input("Hora de Transacción", 
                                       value=datetime.now().time())

    with col2:
        st.markdown("#### 👤 Información Personal")
        first_name = st.selectbox("Nombre", 
                                ["John", "Jane", "Michael", "Sarah", "Other"])
        
        gender = st.selectbox("Género",
                            ["M", "F", "Other"])
        
        dob = st.date_input("Fecha de Nacimiento",
                           value=datetime.now() - pd.Timedelta(days=365*30))
        
        zip_code = st.text_input("Código Postal")

    with col3:
        st.markdown("#### 📍 Información Demográfica")
        city_pop = st.number_input("Población de la Ciudad",
                                 min_value=0,
                                 value=100000)

    return {
        'amount': amount,
        'category': category,
        'transaction_date': transaction_date,
        'transaction_time': transaction_time,
        'first_name': first_name,
        'gender': gender,
        'dob': dob,
        'zip': zip_code,
        'city_pop': city_pop
    }

def preparar_datos_para_modelo(datos, selected_features):
    """Prepara los datos del formulario según las características requeridas por el modelo"""
    # Combinar fecha y hora de transacción
    trans_datetime = datetime.combine(datos['transaction_date'], 
                                    datos['transaction_time'])
    
    # Convertir fechas a valores numéricos
    unix_time = int(trans_datetime.timestamp())
    
    # Para la fecha de nacimiento, convertir a días desde una fecha de referencia
    fecha_referencia = datetime(1970, 1, 1)
    dias_desde_nacimiento = (datos['dob'] - fecha_referencia.date()).days
    
    # Crear DataFrame con las características exactas del modelo
    df_base = pd.DataFrame({
        'Unnamed: 0': [0],  # Valor por defecto
        'trans_date_trans_time': [unix_time],  # Usar timestamp en lugar de string
        'category': [datos['category']],
        'amt': [float(datos['amount'])],
        'first': [datos['first_name']],
        'gender': [datos['gender']],
        'zip': [datos['zip']],
        'city_pop': [int(datos['city_pop'])],
        'dob': [dias_desde_nacimiento],  # Usar días desde fecha referencia
        'unix_time': [unix_time]
    })
    
    # Debug: Mostrar información de preparación
    with st.expander("🔍 Debug: Preparación de Datos"):
        st.write("1. Características del modelo:", selected_features)
        st.write("2. Columnas en DataFrame:", df_base.columns.tolist())
        st.write("3. Valores antes de transformación:", df_base.iloc[0].to_dict())
        st.write("4. Valores temporales convertidos:")
        st.write(f"   - Unix Time: {unix_time}")
        st.write(f"   - Días desde nacimiento: {dias_desde_nacimiento}")
        
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
    
    if not datos['zip'].strip():
        errores.append("El código postal es requerido")
    
    if datos['city_pop'] <= 0:
        errores.append("La población de la ciudad debe ser mayor que 0")
    
    # Validar fecha de nacimiento
    if datos['dob'] >= datetime.now().date():
        errores.append("La fecha de nacimiento debe ser en el pasado")
    
    return errores

def procesar_prediccion(df_preparado, modelo_components):
    """Procesa la predicción con el modelo"""
    try:
        # Extraer componentes
        modelo = modelo_components['modelo']
        scaler = modelo_components['scaler']
        encoders = modelo_components['encoders']
        
        # Aplicar encoders solo a las columnas categóricas que no son fechas
        df_encoded = df_preparado.copy()
        columnas_categoricas = ['category', 'first', 'gender']
        
        for columna in columnas_categoricas:
            if columna in df_encoded.columns and columna in encoders:
                try:
                    df_encoded[columna] = encoders[columna].transform(df_encoded[columna])
                except Exception as e:
                    st.warning(f"⚠️ No se pudo codificar la columna {columna}: {str(e)}")
        
        # Debug: Mostrar datos después del encoding
        with st.expander("🔍 Debug: Datos Codificados"):
            st.write("Valores después de encoding:", df_encoded.iloc[0].to_dict())
        
        # Asegurar que todas las columnas son numéricas
        for col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='raise')
        
        # Escalar datos
        datos_scaled = scaler.transform(df_encoded)
        
        # Realizar predicción
        prediccion = modelo.predict(datos_scaled)[0]
        
        # Obtener probabilidad si está disponible
        probabilidad = None
        if hasattr(modelo, 'predict_proba'):
            probabilidad = modelo.predict_proba(datos_scaled)[0][1]
        
        return prediccion, probabilidad
        
    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {str(e)}")
        with st.expander("🔍 Debug: Error Detallado"):
            st.write("Tipo de error:", type(e).__name__)
            st.write("Mensaje:", str(e))
            import traceback
            st.code(traceback.format_exc())
        return None, None

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

    # Mostrar detalles adicionales
    with st.expander("📝 Detalles de la Transacción"):
        st.json({
            'Monto': datos['amount'],
            'Categoría': datos['category'],
            'Fecha': datos['transaction_date'].strftime('%Y-%m-%d'),
            'Hora': datos['transaction_time'].strftime('%H:%M:%S'),
            'Código Postal': datos['zip'],
            'Población Ciudad': datos['city_pop'],
            'Predicción': 'Fraudulenta' if prediccion == 1 else 'Legítima',
            'Probabilidad de Fraude': f"{probabilidad:.1%}" if probabilidad is not None else "No disponible"
        })
        st.write(f"🕒 Evaluación realizada el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    st.title("🔍 Sistema de Detección de Fraudes")
    st.write("Complete el formulario con los datos de la transacción para evaluar si es fraudulenta")

    # Cargar el modelo
    modelo_components = cargar_modelo()
    if modelo_components is None:
        return

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
            # Preparar datos
            df_preparado = preparar_datos_para_modelo(
                datos, 
                modelo_components['selected_features']
            )
            
            # Procesar predicción
            prediccion, probabilidad = procesar_prediccion(
                df_preparado, 
                modelo_components
            )
            
            if prediccion is not None:
                # Mostrar resultado
                mostrar_resultado(prediccion, datos, probabilidad)
                
                # Logging
                logging.info(f"Predicción realizada: {prediccion} para transacción de ${datos['amount']}")
            
        except Exception as e:
            st.error(f"❌ Error al procesar la transacción: {str(e)}")
            with st.expander("🔍 Debug: Error Detallado"):
                st.write("Tipo de error:", type(e).__name__)
                st.write("Mensaje:", str(e))
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
