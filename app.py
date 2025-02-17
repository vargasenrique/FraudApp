# app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Detector de Fraudes",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def cargar_modelo(filename):
    """Carga el modelo guardado"""
    return joblib.load(filename)

def main():
    st.title("🔍 Sistema de Detección de Fraudes")
    st.write("Ingrese los datos de la transacción para evaluar si es fraudulenta")

    # Cargar el modelo
    try:
        modelo_components = cargar_modelo('modelo_fraude_[TIMESTAMP].joblib')  # Reemplazar [TIMESTAMP] con el nombre real del archivo
        modelo = modelo_components['modelo']
        scaler = modelo_components['scaler']
        encoders = modelo_components['encoders']
        selected_features = modelo_components['selected_features']
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return

    # Crear formulario para entrada de datos
    with st.form("transaction_form"):
        st.subheader("Datos de la Transacción")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Monto ($)", min_value=0.0, step=0.01)
            merchant = st.text_input("Comerciante")
            category = st.selectbox("Categoría", 
                ["grocery_pos", "shopping_pos", "entertainment", "food_dining", "health_fitness"])

        with col2:
            state = st.selectbox("Estado", ["NY", "CA", "TX", "FL"])  # Añadir más estados según necesidad
            city = st.text_input("Ciudad")
            zip_code = st.text_input("Código Postal")

        with col3:
            lat = st.number_input("Latitud", -90.0, 90.0, 0.0)
            long = st.number_input("Longitud", -180.0, 180.0, 0.0)

        submitted = st.form_submit_button("Evaluar Transacción")

    if submitted:
        # Crear DataFrame con los datos ingresados
        data = {
            'amount': [amount],
            'merchant': [merchant],
            'category': [category],
            'state': [state],
            'city': [city],
            'zip': [zip_code],
            'lat': [lat],
            'long': [long],
            'merch_lat': [lat],  # Usando los mismos valores para ejemplo
            'merch_long': [long]
        }
        
        nueva_transaccion = pd.DataFrame(data)

        try:
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
            st.header("Resultado del Análisis")
            if prediccion == 1:
                st.error("⚠️ POSIBLE FRAUDE DETECTADO")
                st.write("Esta transacción muestra patrones similares a transacciones fraudulentas.")
            else:
                st.success("✅ TRANSACCIÓN NORMAL")
                st.write("Esta transacción parece ser legítima.")

            # Mostrar detalles adicionales
            st.subheader("Detalles de la Transacción")
            st.json(data)
            
            # Guardar registro
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"Evaluación realizada el: {now}")

        except Exception as e:
            st.error(f"Error al procesar la transacción: {str(e)}")

if __name__ == "__main__":
    main()
