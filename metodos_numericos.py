
# Calculadora de Métodos Numéricos

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify, exp

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Calculadora de Métodos Numéricos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ESTILO CSS PERSONALIZADO
st.markdown("""
<style>
body {
    background-color: #6F457D;
    color: #2e2e2e;
}
h1, h2, h3 {
    color: #6F457D;
}
.stButton>button {
    background-color: #6F457D;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stNumberInput>div>div>input, .stTextInput>div>input {
    border-radius: 8px;
    height: 2.5em;
}
.stSidebar {
    background-color: #6F457D;
}
</style>
""", unsafe_allow_html=True)

# FUNCIONES AUXILIARES
def crear_funcion(funcion_str):
    """
    Convierte un string a función numérica usando sympy y lambdify.
    Acepta 'exp()' en lugar de 'e^'
    """
    x = symbols('x')
    funcion_simbolica = sympify(funcion_str)
    return lambdify(x, funcion_simbolica, modules=["numpy"])

def derivar_funcion(funcion_str):
    """
    Devuelve la derivada de la función como función numérica
    """
    x = symbols('x')
    funcion_simbolica = sympify(funcion_str)
    derivada_simbolica = funcion_simbolica.diff(x)
    return lambdify(x, derivada_simbolica, modules=["numpy"])

# MÉTODOS NUMÉRICOS

def metodo_biseccion(funcion, xi, xu, error_max=0.001, iter_max=100):
    iteracion = 0
    xr = (xi + xu) / 2
    tabla_datos = []

    while iteracion < iter_max:
        fxi = float(funcion(xi))
        fxu = float(funcion(xu))
        xr_anterior = xr
        xr = (xi + xu) / 2
        fxr = float(funcion(xr))

        error = 100.0 if iteracion == 0 else abs((xr - xr_anterior) / xr) * 100
        signo_valor = np.sign(fxi * fxr)
        regla = "Xu → Xr" if signo_valor < 0 else "Xi → Xr"

        tabla_datos.append({
            "Iteración": iteracion + 1,
            "Xi": xi,
            "Xu": xu,
            "f(Xi)": fxi,
            "f(Xu)": fxu,
            "Xr": xr,
            "f(Xr)": fxr,
            "Error (%)": error,
            "Signo": "+" if signo_valor > 0 else "-",
            "Regla": regla
        })

        if signo_valor < 0:
            xu = xr
        else:
            xi = xr

        if error < error_max and iteracion != 0:
            break
        iteracion += 1

    return xr, pd.DataFrame(tabla_datos)

def metodo_falsa_posicion(funcion, xi, xu, error_max=0.001, iter_max=100):
    iteracion = 0
    xr = xi
    tabla_datos = []

    while iteracion < iter_max:
        fxi = float(funcion(xi))
        fxu = float(funcion(xu))
        xr_anterior = xr
        xr = xu - (fxu * (xi - xu)) / (fxi - fxu)
        fxr = float(funcion(xr))

        error = 100.0 if iteracion == 0 else abs((xr - xr_anterior) / xr) * 100
        signo_valor = np.sign(fxi * fxr)
        regla = "Xu ← Xr" if signo_valor < 0 else "Xi ← Xr"

        tabla_datos.append({
            "Iteración": iteracion + 1,
            "Xi": xi,
            "Xu": xu,
            "f(Xi)": fxi,
            "f(Xu)": fxu,
            "Xr": xr,
            "f(Xr)": fxr,
            "Error (%)": error,
            "Signo": "+" if signo_valor > 0 else "-",
            "Regla": regla
        })

        if signo_valor < 0:
            xu = xr
        else:
            xi = xr

        if error < error_max and iteracion != 0:
            break
        iteracion += 1

    return xr, pd.DataFrame(tabla_datos)

def metodo_punto_fijo(funcion, g_funcion, x0, error_max=0.001, iter_max=100):
    iteracion = 0
    xi = x0
    tabla_datos = []

    while iteracion < iter_max:
        xi_anterior = xi
        xi = float(g_funcion(xi_anterior))
        fxi = float(funcion(xi))

        error = 100.0 if iteracion == 0 else abs((xi - xi_anterior) / xi) * 100

        tabla_datos.append({
            "Iteración": iteracion + 1,
            "Xi": xi_anterior,
            "g(Xi)": xi,
            "f(g(Xi))": fxi,
            "Error (%)": error
        })

        if error < error_max and iteracion != 0:
            break
        iteracion += 1

    return xi, pd.DataFrame(tabla_datos)

def metodo_newton_raphson(funcion, derivada, x0, error_max=0.001, iter_max=100):
    iteracion = 0
    xi = x0
    tabla_datos = []

    while iteracion < iter_max:
        fxi = float(funcion(xi))
        fpxi = float(derivada(xi))
        xi_siguiente = xi - fxi / fpxi

        error = 100.0 if iteracion == 0 else abs((xi_siguiente - xi) / xi_siguiente) * 100

        tabla_datos.append({
            "Iteración": iteracion + 1,
            "Xi": xi,
            "f(Xi)": fxi,
            "f'(Xi)": fpxi,
            "Xi+1": xi_siguiente,
            "Error (%)": error
        })

        if error < error_max and iteracion != 0:
            break

        xi = xi_siguiente
        iteracion += 1

    return xi, pd.DataFrame(tabla_datos)

# INTERFAZ DE USUARIO
st.title("Calculadora de Métodos Numéricos")
st.write("Castillo Cruz Sofia 24212705 SC4A")
st.write("Encuentra la raíz de tu función con diferentes métodos")

# Sidebar para entradas
with st.sidebar:
    st.header("Configurar Calculadora")
    funcion_str = st.text_input("Función f(x):", "-0.5*x^2+2.5*x+4.5")
    funcion_lambda = crear_funcion(funcion_str)
    derivada_lambda = derivar_funcion(funcion_str)

    metodo = st.selectbox("Método:", ["Bisección", "Falsa Posición", "Punto Fijo", "Newton-Raphson"])
    criterio = st.radio("Criterio de parada:", ["Error", "Iteraciones"])

    if criterio == "Error":
        error_max = st.number_input("Error máximo (%)", value=0.01, format="%.2f")
        iter_max = 1000
    else:
        iter_max = st.number_input("Número máximo de iteraciones", value=10, format="%d")
        error_max = 0.0

    if metodo in ["Bisección", "Falsa Posición"]:
        xi = st.number_input("Xi (límite inferior):", value=5, format="%d")
        xu = st.number_input("Xu (límite superior):", value=10, format="%d")
    else:
        x0 = st.number_input("Valor inicial Xi:", value=0, format="%d")
        if metodo == "Punto Fijo":
            g_str = st.text_input("g(x) para Punto Fijo:", "x")
            g_lambda = crear_funcion(g_str)

# Botón de cálculo
if st.button("Calcular raíz"):
    if metodo == "Bisección":
        raiz, tabla = metodo_biseccion(funcion_lambda, xi, xu, error_max, iter_max)
    elif metodo == "Falsa Posición":
        raiz, tabla = metodo_falsa_posicion(funcion_lambda, xi, xu, error_max, iter_max)
    elif metodo == "Punto Fijo":
        raiz, tabla = metodo_punto_fijo(funcion_lambda, g_lambda, x0, error_max, iter_max)
    elif metodo == "Newton-Raphson":
        raiz, tabla = metodo_newton_raphson(funcion_lambda, derivada_lambda, x0, error_max, iter_max)

    # Caja destacada con la raíz
    st.markdown(f"<div style='background-color:#6F457D;padding:15px;border-radius:10px;color:white;font-size:18px;text-align:center'>Raíz aproximada: {raiz:.6f}</div>", unsafe_allow_html=True)

    # Tabla y gráfica lado a lado
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tabla de iteraciones")
        st.dataframe(tabla)

    with col2:
        st.subheader("Gráfica")
        x_vals = np.linspace(raiz - 5, raiz + 5, 200)
        y_vals = funcion_lambda(x_vals)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_vals, y_vals, label="f(x)", color="#24043b", linewidth=2)
        ax.axvline(raiz, color="#3820a3", linestyle="--", label="Raíz aproximada")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(f"f(x) = {funcion_str}", color="#24043b")
        ax.legend()
        st.pyplot(fig)