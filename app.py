import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.graph_objects as go

# =========================================================================
# PLATAFORMA DE OPTIMIZACIÓN WEB - GRUPO VMA OPTIMA (STREAMLIT)
# =========================================================================

st.set_page_config(page_title="Plataforma de Optimización - VMA Optima", layout="wide")
st.title("Plataforma Web de Optimización - Grupo: VMA Optima")

# --- PANEL LATERAL (DATOS DE ENTRADA) ---
with st.sidebar:
    st.header("🔧 Datos de Entrada")
    
    n_vars = st.number_input("Número de Variables", min_value=1, max_value=3, value=2)
    metodo = st.selectbox("Método de Optimización", ['Gradiente', 'Gradiente Conjugado (FR)', 'Newton'])
    
    # Nota: Internamente el código convertirá los "^" a "**" para que funcione como MATLAB
    funcion_str_input = st.text_input("Función Objetivo f(x,y,z)", value="2*x^2 - 4*x*y + y^4 + 5*y^2 - 10*y")
    st.caption("Use: x, y, z (Ej: x^2 + y^2)")
    
    x0_str = st.text_input("Punto de Partida (x_0)", value="0, 0")
    max_iter = st.number_input("Número de Iteraciones", min_value=1, value=100)
    tol = st.number_input("Tolerancia de Convergencia", value=1e-6, format="%.1e")
    
    st.markdown("### Parámetros de Búsqueda")
    alpha_0 = st.number_input("Paso Inicial Alpha (α)", value=1.0)
    tipo_busqueda = st.selectbox("Criterio de Búsqueda", ['Solo Armijo', 'Wolfe Completo', 'Wolfe Completo sin Backtracking'])
    
    c1 = st.number_input("Parámetro Armijo (β)", value=0.1)
    
    disabled_c2 = (tipo_busqueda == 'Solo Armijo')
    c2 = st.number_input("Parámetro Curvatura (σ)", value=0.9, disabled=disabled_c2)
    
    disabled_rho = (tipo_busqueda == 'Wolfe Completo sin Backtracking')
    rho = st.number_input("Contracción Backtracking (ρ)", value=0.5, disabled=disabled_rho)
    
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
        
        vars_sym = sp.symbols('x y z')[:n_vars]
        f_sym = sp.sympify(funcion_str)
        grad_sym = [sp.diff(f_sym, var) for var in vars_sym]
        hess_sym = [[sp.diff(g, var) for var in vars_sym] for g in grad_sym]
        
        f_num = sp.lambdify([vars_sym], f_sym, "numpy")
        grad_num = sp.lambdify([vars_sym], grad_sym, "numpy")
        hess_num = sp.lambdify([vars_sym], hess_sym, "numpy")
        
        def f(v): return float(f_num(list(v)))
        def grad(v): return np.array(grad_num(list(v)), dtype=float)
        def hess(v): return np.array(hess_num(list(v)), dtype=float)

        history = [xk.copy()]
        f_history = [f(xk)]
        err_history = [np.linalg.norm(grad(xk))]
        status = "Número máximo de iteraciones alcanzado"
        dk_old = None
        g_old = None
        
        # Bloqueo lógico interno
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
        # --- GRÁFICOS INTERACTIVOS (PLOTLY) ---

        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            st.subheader("📉 Gráfico de Convergencia")
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(
                x=list(range(len(err_history))), 
                y=err_history, 
                mode='lines+markers', 
                name='Error ||∇f||',
                line=dict(color='#0066cc', width=2),
                marker=dict(size=6)
            ))
            fig_err.update_layout(
                xaxis_title="Número de Iteraciones",
                yaxis_title="Error log(||∇f||)",
                yaxis_type="log",
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig_err, use_container_width=True)
            
        with g_col2:
            st.subheader("🗺️ Trayectoria Espacial (Interactiva)")
            if n_vars == 2:
                history = np.array(history)
                x_pts, y_pts = history[:, 0], history[:, 1]
                
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
                
                # Capa 1: Curvas de nivel de fondo
                fig_traj.add_trace(go.Contour(
                    x=x_grid, y=y_grid, z=Z, 
                    colorscale='Viridis', 
                    opacity=0.5, 
                    name='f(x,y)',
                    hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>f(x,y): %{z:.3f}<extra></extra>'
                ))
                
                # Capa 2: Línea de trayectoria
                fig_traj.add_trace(go.Scatter(
                    x=x_pts, y=y_pts, 
                    mode='lines+markers', 
                    name='Trayectoria',
                    line=dict(color='red', width=2),
                    marker=dict(symbol='circle', size=6, color='red'),
                    hovertemplate='Punto: (%{x:.4f}, %{y:.4f})<extra></extra>'
                ))
                
                # Capa 3: Punto de Inicio
                fig_traj.add_trace(go.Scatter(
                    x=[history[0,0]], y=[history[0,1]], 
                    mode='markers', 
                    name='Inicio',
                    marker=dict(color='blue', size=12, line=dict(color='white', width=2)),
                    hovertemplate='Inicio: (%{x:.4f}, %{y:.4f})<extra></extra>'
                ))
                
                # Capa 4: Punto Mínimo Encontrado
                fig_traj.add_trace(go.Scatter(
                    x=[xk[0]], y=[xk[1]], 
                    mode='markers', 
                    name='Mínimo (x*)',
                    marker=dict(color='green', symbol='star', size=16, line=dict(color='black', width=1)),
                    hovertemplate='Mínimo: (%{x:.4f}, %{y:.4f})<extra></extra>'
                ))
                
                fig_traj.update_layout(
                    xaxis_title="Eje X",
                    yaxis_title="Eje Y",
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig_traj, use_container_width=True)
            else:
                st.info("El gráfico de trayectoria espacial interactiva solo está disponible para N=2 variables.")

    except Exception as e:
        st.error(f"Ocurrió un error matemático o de sintaxis: {str(e)}")