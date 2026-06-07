# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.graph_objects as go
 
 # =========================================================================
 # PLATAFORMA DE OPTIMIZACIÓN WEB - GRUPO VMA OPTIMA (STREAMLIT)
 # =========================================================================

# Configuración de página en modo ancho para aprovechar la pantalla completa
st.set_page_config(layout="wide")
 
st.markdown("""
<style>

.stApp {
    background-color: #081c3a;
    color: white;
}

/* Títulos */
h1, h2, h3 {
    color: #d4af37 !important;
}

/* Texto normal y etiquetas */
p, label {
    color: white !important;
}

/* CONFIGURACIÓN DE LAS PESTAÑAS PARA QUE ACTÚEN COMO GRANDES BOTONES */
button[data-baseweb="tab"] {
    background-color: #10284d !important;
    color: white !important;
    border: 2px solid #d4af37 !important;
    border-radius: 10px 10px 0px 0px;
    padding: 12px 24px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    margin-right: 8px !important;
    transition: all 0.3s ease;
    text-transform: uppercase;
}

/* Botón/Pestaña Seleccionada (Activa) */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #d4af37 !important;
    color: #081c3a !important;
    border-color: #d4af37 !important;
}

/* Efecto Hover al pasar el mouse por los botones/pestañas */
button[data-baseweb="tab"]:hover {
    background-color: #f0c94a !important;
    color: #081c3a !important;
    cursor: pointer;
}

/* El recuadro o "pestañita" contenedora de los datos de entrada */
div[data-testid="stTabPanel"] {
    background-color: #06152d !important;
    border: 2px solid #d4af37 !important;
    border-radius: 0px 10px 10px 10px;
    padding: 25px !important;
    margin-top: -2px !important;
    margin-bottom: 25px !important;
}

/* Inputs (Cajas de texto y números) - Alta legibilidad */
.stTextInput input,
.stNumberInput input {
    background-color: #10284d !important;
    color: white !important;
    border: 2px solid #d4af37 !important;
    border-radius: 8px;
}

/* Foco en inputs al escribir */
.stTextInput input:focus,
.stNumberInput input:focus {
    background-color: #10284d !important;
    color: white !important;
    border: 2px solid #f0c94a !important;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] {
    background-color: #10284d !important;
    color: white !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #10284d;
}

/* Botón principal de Ejecutar */
.stButton button {
    background-color: #d4af37;
    color: #081c3a;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

.stButton button:hover {
    background-color: #f0c94a;
}

/* Métricas */
[data-testid="metric-container"] {
    background-color: #10284d;
    border: 1px solid #d4af37;
    padding: 10px;
    border-radius: 10px;
}

/* Selectbox corrección de contraste */
[data-baseweb="select"] span {
    color: white !important;
}

div[data-baseweb="select"] * {
    color: white !important;
}

ul[role="listbox"] {
    background-color: #10284d !important;
}

li[role="option"] {
    color: white !important;
    background-color: #10284d !important;
}

li[role="option"]:hover {
    background-color: #d4af37 !important;
    color: #081c3a !important;
}

</style>
""", unsafe_allow_html=True)

st.title("Plataforma Web de Optimización - Grupo: VMA Optima")
st.markdown("Selecciona el método de optimización mediante los botones superiores para configurar sus parámetros correspondientes:")

# --- FUNCIÓN CENTRALIZADA DE CÓMPUTO Y RESOLUCIÓN MATEMÁTICA ---
def resolver_y_graficar(metodo, n_vars, funcion_str_input, x0_str, max_iter, tol, tipo_busqueda, alpha_0, c1, c2, rho):
    try:
        # Pre-procesamiento sintaxis MATLAB
        funcion_str = funcion_str_input.replace('^', '**')
        x_vals = [float(i) for i in x0_str.split(",")]
        
        if len(x_vals) != n_vars:
            st.error(f"❌ El punto de partida debe tener exactamente {n_vars} valores separados por comas.")
            return

        xk = np.array(x_vals, dtype=float)
        var_names = [f"x{i+1}" for i in range(n_vars)]
        vars_sym = tuple(sp.symbols(" ".join(var_names)) if n_vars > 1 else [sp.Symbol("x1")])
        
        f_sym = sp.sympify(funcion_str)
        grad_sym = [sp.diff(f_sym, var) for var in vars_sym]
        hess_sym = [[sp.diff(g, var) for var in vars_sym] for g in grad_sym]

        f_num = sp.lambdify(vars_sym, f_sym, "numpy")
        grad_num = sp.lambdify(vars_sym, grad_sym, "numpy")
        hess_num = sp.lambdify(vars_sym, hess_sym, "numpy")
        
        def f(v): return float(f_num(*v))
        def grad(v): return np.array(grad_num(*v), dtype=float)
        def hess(v): return np.array(hess_num(*v), dtype=float)

        history = [xk.copy()]
        f_history = [f(xk)]
        err_history = [np.linalg.norm(grad(xk))]
        status = "Número máximo de iteraciones alcanzado"
        dk_old = None
        g_old = None

        for k in range(1, max_iter + 1):
            g = grad(xk)
            err = np.linalg.norm(g)

            if err < tol:
                status = f"Convergencia exitosa: ||grad|| < {tol}"
                break

            if metodo == 'Gradiente':
                dk = -g
            elif metodo == 'Newton':
                H = hess(xk)
                H = np.atleast_2d(H)
                if np.linalg.cond(H) > 1e10 or np.isnan(H).any():
                    H += 1e-2 * np.eye(n_vars)
                try:
                    dk = np.linalg.solve(H, -g)
                except:
                    dk = -g
            elif metodo == 'Gradiente Conjugado (FR)':
                if k == 1 or dk_old is None:
                    dk = -g
                else:
                    beta_cg = np.dot(g, g) / (np.dot(g_old, g_old) + 1e-12)
                    dk = -g + beta_cg * dk_old
                dk_old, g_old = dk, g

            if np.dot(g, dk) > 0: dk = -dk
                
            alpha = alpha_0
            alpha_min = 0.0
            alpha_max = float('inf')
            fk = f(xk)
            g_d = np.dot(g, dk)

            for w_iter in range(50):
                x_next = xk + alpha * dk
                fk_next = f(x_next)
                g_next = grad(x_next)

                cond1 = fk_next <= fk + c1 * alpha * g_d
                cond2 = np.dot(g_next, dk) >= c2 * g_d
                if tipo_busqueda == 'Solo Armijo':
                    if cond1: break
                    else: alpha = rho * alpha
                elif tipo_busqueda == 'Wolfe Completo':
                    if cond1 and cond2: break
                    if not cond1:
                        alpha_max = alpha
                        alpha = rho * alpha
                    else:
                        alpha_min = alpha
                        alpha = 1.5 * alpha if np.isinf(alpha_max) else 0.5 * (alpha_min + alpha_max)
                else:
                    if cond1 and cond2: break
                    if not cond1:
                        alpha_max = alpha
                        alpha = 0.5 * (alpha_min + alpha_max)
                    else:
                        alpha_min = alpha
                        alpha = 2.0 * alpha if np.isinf(alpha_max) else 0.5 * (alpha_min + alpha_max)
                if alpha < 1e-12: break

            xk = xk + alpha * dk
            history.append(xk.copy())
            f_history.append(f(xk))
            err_history.append(np.linalg.norm(grad(xk)))

        st.success(f"**Criterio de Parada Alcanzado:** {status}")

        # --- RESUMEN FINAL ---
        st.subheader("📊 Resumen Final")
        col1, col2, col3 = st.columns(3)
        col1.metric("Punto Mínimo Encontrado (x*)", str(np.round(xk, 5)))
        col2.metric("Valor de la Función f(x*)", f"{f(xk):.8f}")
        col3.metric("Iteraciones Realizadas", str(k if err >= tol else k-1))

        # --- TABLA HISTORIAL ---
        st.subheader("📝 Historial Detallado (Paso a Paso)")
        df_hist = pd.DataFrame({
            "Iteración (k)": range(len(history)),
            "Coordenadas (x_k)": [str(np.round(x, 5)) for x in history],
            "Valor f(x_k)": [f"{fv:.6f}" for fv in f_history],
            "Error ||∇f||": [f"{ev:.6e}" for ev in err_history]
        })
        st.dataframe(df_hist, use_container_width=True)
        st.divider()

        # --- GRÁFICOS ---
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.subheader("📉 Gráfico de Convergencia")
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(
                x=list(range(len(err_history))), y=err_history,
                mode='lines+markers', name='Error',
                text=[f"Iteración: {i}<br>xk = {np.round(history[i],5)}<br>Error = {err_history[i]:.3e}" for i in range(len(history))],
                hoverinfo='text'
            ))
            fig_err.update_layout(title='Convergencia del Error', xaxis_title='Iteración', yaxis_title='||∇f||', yaxis_type='log', height=450)
            st.plotly_chart(fig_err, use_container_width=True)

        with g_col2:
            st.subheader("🗺️ Trayectoria Realizada")
            if n_vars == 2:
                history_np = np.array(history)
                x_pts, y_pts = history_np[:, 0], history_np[:, 1]
                mx = max(abs(max(x_pts)-min(x_pts))*0.6, 2.0)
                my = max(abs(max(y_pts)-min(y_pts))*0.6, 2.0)
                x_grid = np.linspace(min(x_pts)-mx, max(x_pts)+mx, 80)
                y_grid = np.linspace(min(y_pts)-my, max(y_pts)+my, 80)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = np.zeros_like(X)
                for r in range(X.shape[0]):
                    for c in range(X.shape[1]): Z[r,c] = f([X[r,c], Y[r,c]])
                     
                fig_traj = go.Figure()
                fig_traj.add_trace(go.Contour(x=x_grid, y=y_grid, z=Z, colorscale='Viridis', contours=dict(showlabels=False), opacity=0.6, showscale=False))
                fig_traj.add_trace(go.Scatter(x=history_np[:,0], y=history_np[:,1], mode='lines+markers', name='Trayectoria', text=[f"Iteración: {i}<br>x = {history_np[i,0]:.6f}<br>y = {history_np[i,1]:.6f}<br>f(x) = {f_history[i]:.6f}" for i in range(len(history_np))], hoverinfo='text'))
                fig_traj.add_trace(go.Scatter(x=[history_np[0,0]], y=[history_np[0,1]], mode='markers', name='Inicio', marker=dict(size=12)))
                fig_traj.add_trace(go.Scatter(x=[xk[0]], y=[xk[1]], mode='markers', name='Mínimo', marker=dict(size=14)))
                fig_traj.update_layout(title='Trayectoria del Método', xaxis_title='X', yaxis_title='Y', height=500)
                st.plotly_chart(fig_traj, use_container_width=True)
            elif n_vars == 1:
                history_np = np.array(history).flatten()
                x_min, x_max = min(history_np) - 2, max(history_np) + 2
                x_curve = np.linspace(x_min, x_max, 400)
                y_curve = [f([x]) for x in x_curve]
                fig_1d = go.Figure()
                fig_1d.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='f(x)'))
                fig_1d.add_trace(go.Scatter(x=history_np, y=[f([x]) for x in history_np], mode='markers+lines', name='Iteraciones', text=[f"Iteración: {i}<br>x = {history_np[i]:.6f}<br>f(x) = {f([history_np[i]]):.6f}" for i in range(len(history_np))], hoverinfo='text'))
                fig_1d.update_layout(title='Trayectoria de Optimización en 1D', xaxis_title='x', yaxis_title='f(x)', height=500)
                st.plotly_chart(fig_1d, use_container_width=True)
            else:
                st.info("ℹ️ La visualización gráfica de trayectoria solo está disponible para problemas de 1 o 2 variables.")
    except Exception as e:
        st.error(f"❌ Ocurrió un error matemático o de sintaxis: {str(e)}")

# --- CREACIÓN DE LOS TRES GRANDES BOTONES DE PESTAÑA ---
tab_gradiente, tab_conjugado, tab_newton = st.tabs([
    "📈 Método del Gradiente", 
    "🚀 Gradiente Conjugado (FR)", 
    "🧮 Método de Newton"
])

# =========================================================================
# PESTAÑA 1: MÉTODO DEL GRADIENTE
# =========================================================================
with tab_gradiente:
    st.subheader("🔧 Configuración - Descenso de Gradiente Estándar")
    
    c_grad1, c_grad2 = st.columns(2)
    with c_grad1:
        n_vars_g = st.number_input("Número de Variables", min_value=1, max_value=50, value=2, step=1, key="n_vars_g")
        var_names_g = [f"x{i+1}" for i in range(n_vars_g)]
        default_function_g = " + ".join([f"x{i+1}^2" for i in range(min(n_vars_g,3))])
        funcion_str_g = st.text_input("Función Objetivo", value=default_function_g, key="func_g")
        st.caption(f"Escribir variables como {', '.join(var_names_g)}  
Ejemplo: x1^2 + x2^2")
        
        default_x0_g = ", ".join(["0"] * n_vars_g)
        x0_str_g = st.text_input("Punto de Partida (x₀)", value=default_x0_g, key="x0_g")
        
    with c_grad2:
        max_iter_g = st.number_input("Número de Iteraciones", min_value=1, value=100, key="max_iter_g")
        tol_g = st.number_input("Tolerancia de Convergencia", value=1e-6, format="%.1e", key="tol_g")
        tipo_busqueda_g = st.selectbox("Criterio de Búsqueda de Línea", ['Solo Armijo', 'Wolfe Completo', 'Wolfe Completo sin Backtracking'], key="busqueda_g")

    st.markdown("#### ⚙️ Parámetros Algorítmicos")
    p_grad1, p_grad2, p_grad3 = st.columns(3)
    with p_grad1:
        alpha_0_g = st.number_input("Paso Inicial Alpha (α)", value=1.0, key="alpha_0_g")
    with p_grad2:
        c1_g = st.number_input("Parámetro Armijo (β)", value=0.1, key="c1_g")
    with p_grad3:
        # Lógica condicional adaptativa para Curvatura (σ)
        if tipo_busqueda_g in ['Wolfe Completo', 'Wolfe Completo sin Backtracking']:
            c2_g = st.number_input("Parámetro Curvatura (σ)", min_value=0.01, max_value=0.999, value=0.90, step=0.01, help="Condición de Wolfe. Valores comunes: 0.1 = muy estricto, 0.5 = moderado, 0.9 = estándar, 0.99 = muy permisivo.", key="c2_g")
        else:
            c2_g = 0.90
            
        # Lógica condicional adaptativa para Contracción (ρ)
        if tipo_busqueda_g != 'Wolfe Completo sin Backtracking':
            rho_g = st.number_input("Contracción Backtracking (ρ)", value=0.5, key="rho_g")
        else:
            rho_g = 0.5

    ejecutar_g = st.button("🚀 EJECUTAR OPTIMIZACIÓN - GRADIENTE", use_container_width=True, key="btn_ej_g")
    if ejecutar_g:
        resolver_y_graficar('Gradiente', n_vars_g, funcion_str_g, x0_str_g, max_iter_g, tol_g, tipo_busqueda_g, alpha_0_g, c1_g, c2_g, rho_g)

# =========================================================================
# PESTAÑA 2: GRADIENTE CONJUGADO (FLETCHER-REEVES)
# =========================================================================
with tab_conjugado:
    st.subheader("🔧 Configuración - Gradiente Conjugado (FR)")
    st.info("💡 **Nota Metodológica:** Este método utiliza de forma automática una búsqueda de línea bajo la condición de *Wolfe Completo sin Backtracking* fijando el Paso Inicial $\\alpha_0 = 1.0$ para preservar la conjugación de direcciones.")
    
    c_cg1, c_cg2 = st.columns(2)
    with c_cg1:
        n_vars_cg = st.number_input("Número de Variables", min_value=1, max_value=50, value=2, step=1, key="n_vars_cg")
        var_names_cg = [f"x{i+1}" for i in range(n_vars_cg)]
        default_function_cg = " + ".join([f"x{i+1}^2" for i in range(min(n_vars_cg,3))])
        funcion_str_cg = st.text_input("Función Objetivo", value=default_function_cg, key="func_cg")
        st.caption(f"Variables disponibles: {', '.join(var_names_cg)}")
        
        default_x0_cg = ", ".join(["0"] * n_vars_cg)
        x0_str_cg = st.text_input("Punto de Partida (x₀)", value=default_x0_cg, key="x0_cg")
        
    with c_cg2:
        max_iter_cg = st.number_input("Número de Iteraciones", min_value=1, value=100, key="max_iter_cg")
        tol_cg = st.number_input("Tolerancia de Convergencia", value=1e-6, format="%.1e", key="tol_cg")

    st.markdown("#### ⚙️ Parámetros de Búsqueda de Línea (Wolfe)")
    p_cg1, p_cg2 = st.columns(2)
    with p_cg1:
        c1_cg = st.number_input("Parámetro Armijo (β)", value=0.1, key="c1_cg")
    with p_cg2:
        c2_cg = st.number_input("Parámetro Curvatura (σ)", min_value=0.01, max_value=0.999, value=0.90, step=0.01, help="Condición de Wolfe. Valores comunes: 0.1 = muy estricto, 0.5 = moderado, 0.9 = estándar, 0.99 = muy permisivo.", key="c2_cg")

    ejecutar_cg = st.button("🚀 EJECUTAR OPTIMIZACIÓN - GRADIENTE CONJUGADO", use_container_width=True, key="btn_ej_cg")
    if ejecutar_cg:
        resolver_y_graficar('Gradiente Conjugado (FR)', n_vars_cg, funcion_str_cg, x0_str_cg, max_iter_cg, tol_cg, 'Wolfe Completo sin Backtracking', 1.0, c1_cg, c2_cg, 0.5)

# =========================================================================
# PESTAÑA 3: MÉTODO DE NEWTON
# =========================================================================
with tab_newton:
    st.subheader("🔧 Configuración - Método de Newton de Segundo Orden")
    st.info("💡 **Nota Metodológica:** El método de Newton calcula de forma analítica la matriz Hessiana. Se autoconfigura con búsqueda de línea de *Wolfe Completo sin Backtracking* y $\\alpha_0 = 1.0$ para garantizar la tasa de convergencia cuadrática cerca del mínimo.")
    
    c_nw1, c_nw2 = st.columns(2)
    with c_nw1:
        n_vars_nw = st.number_input("Número de Variables", min_value=1, max_value=50, value=2, step=1, key="n_vars_nw")
        var_names_nw = [f"x{i+1}" for i in range(n_vars_nw)]
        default_function_nw = " + ".join([f"x{i+1}^2" for i in range(min(n_vars_nw,3))])
        funcion_str_nw = st.text_input("Función Objetivo", value=default_function_nw, key="func_nw")
        st.caption(f"Variables disponibles: {', '.join(var_names_nw)}")
        
        default_x0_nw = ", ".join(["0"] * n_vars_nw)
        x0_str_nw = st.text_input("Punto de Partida (x₀)", value=default_x0_nw, key="x0_nw")
        
    with c_nw2:
        max_iter_nw = st.number_input("Número de Iteraciones", min_value=1, value=100, key="max_iter_nw")
        tol_nw = st.number_input("Tolerancia de Convergencia", value=1e-6, format="%.1e", key="tol_nw")

    st.markdown("#### ⚙️ Parámetros de Búsqueda de Línea (Wolfe)")
    p_nw1, p_nw2 = st.columns(2)
    with p_nw1:
        c1_nw = st.number_input("Parámetro Armijo (β)", value=0.1, key="c1_nw")
    with p_nw2:
        c2_nw = st.number_input("Parámetro Curvatura (σ)", min_value=0.01, max_value=0.999, value=0.90, step=0.01, help="Condición de Wolfe. Valores comunes: 0.1 = muy estricto, 0.5 = moderado, 0.9 = estándar, 0.99 = muy permisivo.", key="c2_nw")

    ejecutar_nw = st.button("🚀 EJECUTAR OPTIMIZACIÓN - NEWTON", use_container_width=True, key="btn_ej_nw")
    if ejecutar_nw:
        resolver_y_graficar('Newton', n_vars_nw, funcion_str_nw, x0_str_nw, max_iter_nw, tol_nw, 'Wolfe Completo sin Backtracking', 1.0, c1_nw, c2_nw, 0.5)
