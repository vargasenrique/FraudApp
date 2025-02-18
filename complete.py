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

# [... Mantener la configuración de la página y estilos existentes ...]

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
    
    # Crear un DataFrame con las características exactas que necesita el modelo
    df_base = pd.DataFrame({
        'Unnamed: 0': [0],  # Se usará un valor por defecto
        'trans_date_trans_time': [trans_datetime.strftime('%Y-%m-%d %H:%M:%S')],
        'category': [datos['category']],
        'amt': [datos['amount']],
        'first': [datos['first_name']],
        'gender': [datos['gender']],
        'zip': [datos['zip']],
        'city_pop': [datos['city_pop']],
        'dob': [datos['dob'].strftime('%Y-%m-%d')],
        'unix_time': [int(trans_datetime.timestamp())]
    })
    
    # Debug: Mostrar información de preparación
    with st.expander("🔍 Debug: Preparación de Datos"):
        st.write("1. Características del modelo:", selected_features)
        st.write("2. Columnas en DataFrame:", df_base.columns.tolist())
        st.write("3. Valores antes de transformación:", df_base.iloc[0].to_dict())
    
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
    
    return errores

def procesar_prediccion(df_preparado, modelo_components):
    """Procesa la predicción con el modelo"""
    try:
        # Extraer componentes
        modelo = modelo_components['modelo']
        scaler = modelo_components['scaler']
        encoders = modelo_components['encoders']
        
        # Aplicar encoders a las columnas categóricas
        df_encoded = df_preparado.copy()
        for columna, encoder in encoders.items():
            if columna in df_encoded.columns:
                try:
                    df_encoded[columna] = encoder.transform(df_encoded[columna])
                except:
                    st.warning(f"⚠️ No se pudo codificar la columna {columna}")
        
        # Debug: Mostrar datos después del encoding
        with st.expander("🔍 Debug: Datos Codificados"):
            st.write("Valores después de encoding:", df_encoded.iloc[0].to_dict())
        
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
