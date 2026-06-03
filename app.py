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

/* Texto normal */
p, label {
    color: white !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #06152d;
}

/* Inputs (Cajas de texto y números) - Corregido para legibilidad total al escribir */
.stTextInput input,
.stNumberInput input {
    background-color: #10284d !important;
    color: white !important;
    border: 2px solid #d4af37 !important;
    border-radius: 8px;
}

/* Forzar que el texto y fondo se mantengan legibles al hacer clic o escribir (Focus) */
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

/* Botón principal */
.stButton button {
    background-color: #d4af37;
    color: #081c3a;
    font-weight: bold;
    border-radius: 10px;
    border: none;
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

/* Selectbox cerrado y abierto - Corrección de Contraste */
[data-baseweb="select"] span {
    color: white !important;
}

div[data-baseweb="select"] * {
    color: white !important;
}

/* Menú desplegable completo */
ul[role="listbox"] {
    background-color: #10284d !important;
}

/* Opciones dentro del menú desplegable */
li[role="option"] {
    color: white !important;
    background-color: #10284d !important;
}

/* Efecto hover al pasar el mouse sobre las opciones */
li[role="option"]:hover {
    background-color: #d4af37 !important;
    color: #081c3a !important;
}

</style>
""", unsafe_allow_html=True)
st.title("Plataforma Web de Optimización - Grupo: VMA Optima")
 
# --- PANEL LATERAL (DATOS DE ENTRADA) ---
with st.sidebar:
    st.header("🔧 Datos de Entrada")

    n_vars = st.number_input(
        "Número de Variables",
        min_value=1,
        max_value=50,
        value=2,
        step=1
    )

    var_names = [f"x{i+1}" for i in range(n_vars)]

    metodo = st.selectbox(
        "Método de Optimización",
        ['Gradiente', 'Gradiente Conjugado (FR)', 'Newton']
    )
     # Nota: Internamente el código convertirá los "^" a "**" para que funcione como MATLAB
    default_function = " + ".join(
    [f"x{i+1}^2" for i in range(min(n_vars,3))]
    )
    
    funcion_str_input = st.text_input(
        "Función Objetivo",
        value=default_function
    )
    
    st.caption(
        f"Variables disponibles: {', '.join(var_names)}"
    )

    default_x0 = ", ".join(["0"] * n_vars)

    x0_str = st.text_input(
        "Punto de Partida (x₀)",
        value=default_x0
    )
    max_iter = st.number_input("Número de Iteraciones", min_value=1, value=100)
    tol = st.number_input("Tolerancia de Convergencia", value=1e-6, format="%.1e")
 
    st.markdown("### ⚙️ Parámetros de Búsqueda")
    
    # --- MENÚ DINÁMICO ADAPTATIVO SEGÚN EL MÉTODO ---
    if metodo == 'Gradiente':
        tipo_busqueda = st.selectbox(
            "Criterio de Búsqueda", 
            ['Solo Armijo', 'Wolfe Completo', 'Wolfe Completo sin Backtracking']
        )
        alpha_0 = st.number_input("Paso Inicial Alpha (α)", value=1.0)
        c1 = st.number_input("Parámetro Armijo (β)", value=0.1)

        # Mostrar parámetro Curvatura (c2) solo si es Wolfe
        if tipo_busqueda in ['Wolfe Completo', 'Wolfe Completo sin Backtracking']:
            c2 = st.number_input(
                "Parámetro Curvatura (σ)",
                min_value=0.01,
                max_value=0.999,
                value=0.90,
                step=0.01,
                help="Condición de Wolfe. Valores comunes: 0.1 = muy estricto, 0.5 = moderado, 0.9 = estándar, 0.99 = muy permisivo."
            )
        else:
            c2 = 0.90

        # Mostrar parámetro Contracción (rho) solo si incluye Backtracking
        if tipo_busqueda != 'Wolfe Completo sin Backtracking':
            rho = st.number_input("Contracción Backtracking (ρ)", value=0.5)
        else:
            rho = 0.5

    else:
        # Interfaz simplificada para Newton y Gradiente Conjugado
        st.info(f"💡 **Nota:** El método **{metodo}** utiliza una búsqueda de línea de *Wolfe Completo sin Backtracking* con $\\alpha_0 = 1.0$ fijado de manera automática.")
        
        # Parámetros lógicos internos ocultos al usuario pero asignados para el algoritmo
        tipo_busqueda = 'Wolfe Completo sin Backtracking'
        alpha_0 = 1.0
        rho = 0.5
        
        # Únicos parámetros necesarios de ingresar para estos métodos
        c1 = st.number_input("Parámetro Armijo (β)", value=0.1)
        c2 = st.number_input(
            "Parámetro Curvatura (σ)",
            min_value=0.01,
            max_value=0.999,
            value=0.90,
            step=0.01,
            help="Condición de Wolfe. Valores comunes: 0.1 = muy estricto, 0.5 = moderado, 0.9 = estándar, 0.99 = muy permisivo."
        )

    ejecutar = st.button("🚀 EJECUTAR OPTIMIZACIÓN", use_container_width=True, type="primary")

 # --- LÓGICA MATEMÁTICA Y EJECUCIÓN ---
if ejecutar:
    try:
         # Pre-procesamiento para admitir sintaxis de MATLAB en Python
        funcion_str = funcion_str_input.replace('^', '**')
 
        x_vals = [float(i) for i in x0_str.split(",")]
        if len(x_vals) != n_vars:
            st.error(f"El punto de partida debe tener {n_vars} valores separados por comas.")
            st.stop()
        xk = np.array(x_vals, dtype=float)
        vars_sym = tuple(
            sp.symbols(" ".join(var_names))
            if n_vars > 1
            else [sp.Symbol("x1")]
        )
        f_sym = sp.sympify(funcion_str)
        grad_sym = [sp.diff(f_sym, var) for var in vars_sym]
        hess_sym = [[sp.diff(g, var) for var in vars_sym] for g in grad_sym]

        f_num = sp.lambdify(vars_sym, f_sym, "numpy")
        grad_num = sp.lambdify(vars_sym, grad_sym, "numpy")
        hess_num = sp.lambdify(vars_sym, hess_sym, "numpy")
        
        def f(v):
            return float(f_num(*v))
        
        def grad(v):
            return np.array(grad_num(*v), dtype=float)
        
        def hess(v):
            return np.array(hess_num(*v), dtype=float)

        history = [xk.copy()]
        f_history = [f(xk)]
        err_history = [np.linalg.norm(grad(xk))]
        status = "Número máximo de iteraciones alcanzado"
        dk_old = None
        g_old = None

        # Sincronización lógica interna redundante de seguridad
        if metodo != 'Gradiente':
            tipo_busqueda = 'Wolfe Completo sin Backtracking'
            alpha_0 = 1.0

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
                x=list(range(len(err_history))),
                y=err_history,
                mode='lines+markers',
                name='Error',
                text=[
                    f"""
                    Iteración: {i}<br>
                    xk = {np.round(history[i],5)}<br>
                    Error = {err_history[i]:.3e}
                    """
                    for i in range(len(history))
                ],
                hoverinfo='text'
            ))

            fig_err.update_layout(
                title='Convergencia del Error',
                xaxis_title='Iteración',
                yaxis_title='||∇f||',
                yaxis_type='log',
                height=450
            )

            st.plotly_chart(fig_err, use_container_width=True)

        with g_col2:
            st.subheader("🗺️ Trayectoria Realizada")
            if n_vars == 2:
                
                history_np = np.array(history)
                x_pts = history_np[:, 0]
                y_pts = history_np[:, 1]

                mx = max(abs(max(x_pts)-min(x_pts))*0.6, 2.0)
                my = max(abs(max(y_pts)-min(y_pts))*0.6, 2.0)
               
                x_grid = np.linspace(min(x_pts)-mx, max(x_pts)+mx, 80)
                y_grid = np.linspace(min(y_pts)-my, max(y_pts)+my, 80)
               
                X, Y = np.meshgrid(x_grid, y_grid)
                
                Z = np.zeros_like(X)
                
                for r in range(X.shape[0]):
                    for c in range(X.shape[1]):
                        Z[r,c] = f([X[r,c], Y[r,c]])
                     
                fig_traj = go.Figure()

                # Curvas de nivel
                fig_traj.add_trace(go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=Z,
                    colorscale='Viridis',
                    contours=dict(showlabels=False),
                    opacity=0.6,
                    showscale=False
                ))

                # Trayectoria
                fig_traj.add_trace(go.Scatter(
                    x=history_np[:,0],
                    y=history_np[:,1],
                    mode='lines+markers',
                    name='Trayectoria',
                    text=[
                        f"""
                        Iteración: {i}<br>
                        x = {history_np[i,0]:.6f}<br>
                        y = {history_np[i,1]:.6f}<br>
                        f(x) = {f_history[i]:.6f}
                        """
                        for i in range(len(history_np))
                    ],
                   hoverinfo='text'
                ))
 
                # Punto inicial
                fig_traj.add_trace(go.Scatter(
                    x=[history_np[0,0]],
                    y=[history_np[0,1]],
                    mode='markers',
                    name='Inicio',
                    marker=dict(size=12)
                ))

                # Punto final
                fig_traj.add_trace(go.Scatter(
                    x=[xk[0]],
                    y=[xk[1]],
                    mode='markers',
                    name='Mínimo',
                    marker=dict(size=14)
               ))

                fig_traj.update_layout(
                    title='Trayectoria del Método',
                    xaxis_title='X',
                    yaxis_title='Y',
                    height=500
                )

                st.plotly_chart(fig_traj, use_container_width=True)
            elif n_vars == 1:

                history_np = np.array(history).flatten()
            
                x_min = min(history_np) - 2
                x_max = max(history_np) + 2
            
                x_curve = np.linspace(x_min, x_max, 400)
                y_curve = [f([x]) for x in x_curve]
            
                fig_1d = go.Figure()
            
                # Curva de la función
                fig_1d.add_trace(go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode='lines',
                    name='f(x)'
                ))
            
                # Puntos iterativos
                fig_1d.add_trace(go.Scatter(
                    x=history_np,
                    y=[f([x]) for x in history_np],
                    mode='markers+lines',
                    name='Iteraciones',
                    text=[
                        f"""
                        Iteración: {i}<br>
                        x = {history_np[i]:.6f}<br>
                        f(x) = {f([history_np[i]]):.6f}
                        """
                        for i in range(len(history_np))
                    ],
                    hoverinfo='text'
                ))
            
                fig_1d.update_layout(
                    title='Trayectoria de Optimización en 1D',
                    xaxis_title='x',
                    yaxis_title='f(x)',
                    height=500
                )
            
                st.plotly_chart(fig_1d, use_container_width=True)
            
            else:
                st.info("La visualización gráfica solo está disponible para 1 o 2 variables.")

    except Exception as e:
        st.error(f"Ocurrió un error matemático o de sintaxis: {str(e)}")
