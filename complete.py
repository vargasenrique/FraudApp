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
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    try:
        GDRIVE_FILE_ID = "1FuCvBzGOvN2q8AX_vEBc1vdbcuCj8j4i"
        download_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        
        response = requests.get(download_url)
        if response.status_code != 200:
            st.error("❌ Error al descargar el modelo")
            return None
            
        model_bytes = BytesIO(response.content)
        modelo_components = joblib.load(model_bytes)
        return modelo_components
        
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

def preparar_datos_para_modelo(datos, selected_features):
    """Prepara los datos del formulario para el modelo"""
    # Crear un DataFrame con todas las características necesarias
    current_time = datetime.now()
    
    df_base = pd.DataFrame({
        'amt': [datos['amount']],
        'zip': [int(datos['zip'])],
        'lat': [datos['lat']],
        'long': [datos['long']],
        'city_pop': [datos['city_pop']],
        'unix_time': [int(current_time.timestamp())],
        'merch_lat': [datos['merch_lat']],
        'merch_long': [datos['merch_long']],
        'category': [datos['category']],
        'merchant': [datos['merchant']]
    })
    
    # Asegurar que tenemos todas las columnas necesarias
    for feature in selected_features:
        if feature not in df_base.columns:
            df_base[feature] = 0  # Valor por defecto para columnas faltantes
    
    # Seleccionar solo las características necesarias en el orden correcto
    return df_base[selected_features]

def crear_campos_formulario():
    """Crea los campos del formulario con todos los datos necesarios"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Información Básica")
        amount = st.number_input("Monto ($)", min_value=0.0, step=0.01)
        category = st.selectbox(
            "Categoría",
            ["grocery_pos", "shopping_pos", "entertainment", "food_dining", 
             "health_fitness", "shopping_net", "kids_pets", "personal_care",
             "home", "gas_transport", "misc_pos", "misc_net"]
        )
        merchant = st.text_input("Comerciante")

    with col2:
        st.markdown("#### 📍 Ubicación del Cliente")
        zip_code = st.text_input("Código Postal")
        city_pop = st.number_input("Población de la Ciudad", min_value=0)
        lat = st.number_input("Latitud", min_value=-90.0, max_value=90.0, value=0.0)
        long = st.number_input("Longitud", min_value=-180.0, max_value=180.0, value=0.0)

    with col3:
        st.markdown("#### 🏪 Ubicación del Comercio")
        merch_lat = st.number_input("Latitud del Comercio", min_value=-90.0, max_value=90.0, value=0.0)
        merch_long = st.number_input("Longitud del Comercio", min_value=-180.0, max_value=180.0, value=0.0)

    return {
        'amount': amount,
        'merchant': merchant,
        'category': category,
        'zip': zip_code,
        'city_pop': city_pop,
        'lat': lat,
        'long': long,
        'merch_lat': merch_lat,
        'merch_long': merch_long
    }

def validar_datos_entrada(datos):
    """Valida los datos de entrada del formulario"""
    errores = []
    
    if datos['amount'] <= 0:
        errores.append("El monto debe ser mayor que 0")
    
    if not datos['merchant'].strip():
        errores.append("El nombre del comerciante es requerido")
        
    try:
        int(datos['zip'])
    except ValueError:
        errores.append("El código postal debe ser un número")
    
    if datos['city_pop'] <= 0:
        errores.append("La población de la ciudad debe ser mayor que 0")
    
    return errores

def mostrar_resultado(prediccion, datos, probabilidad=None):
    """Muestra el resultado de la predicción"""
    st.header("📊 Resultado del Análisis")
    
    if prediccion == 1:
        st.error("⚠️ ALERTA: POSIBLE FRAUDE DETECTADO")
        st.write("Esta transacción muestra patrones similares a transacciones fraudulentas.")
    else:
        st.success("✅ TRANSACCIÓN SEGURA")
        st.write("Esta transacción parece ser legítima según nuestro análisis.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monto", f"${datos['amount']:,.2f}")
    with col2:
        st.metric("Categoría", datos['category'].replace('_', ' ').title())
    with col3:
        if probabilidad is not None:
            st.metric("Probabilidad de Fraude", f"{probabilidad:.1%}")

def main():
    st.title("🔍 Sistema de Detección de Fraudes")
    
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
            # Preparar datos para el modelo
            nueva_transaccion = preparar_datos_para_modelo(
                datos, 
                modelo_components['selected_features']
            )
            
            # Aplicar encoders
            for columna, encoder in modelo_components['encoders'].items():
                if columna in nueva_transaccion.columns:
                    nueva_transaccion[columna] = encoder.transform(nueva_transaccion[columna])
            
            # Escalar datos
            transaccion_scaled = modelo_components['scaler'].transform(nueva_transaccion)
            
            # Realizar predicción
            prediccion = modelo_components['modelo'].predict(transaccion_scaled)[0]
            
            # Obtener probabilidad si está disponible
            probabilidad = None
            if hasattr(modelo_components['modelo'], 'predict_proba'):
                probabilidad = modelo_components['modelo'].predict_proba(transaccion_scaled)[0][1]
            
            # Mostrar resultado
            mostrar_resultado(prediccion, datos, probabilidad)
            
        except Exception as e:
            st.error(f"❌ Error al procesar la transacción: {str(e)}")
            st.expander("🔍 Detalles del Error").write(str(e))

if __name__ == "__main__":
    main()
