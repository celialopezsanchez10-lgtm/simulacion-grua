"""
Mecánica y Ondas I - Pluma de grúa en caída libre con gancho.

Integra numéricamente las ecuaciones de Euler-Lagrange del sistema
(sin usar la solución analítica), comprueba las constantes del movimiento,
dibuja las gráficas y genera las animaciones de varios casos (distintas
condiciones iniciales, casos límite y distintos parámetros del sistema).

Coordenadas generalizadas (4 grados de libertad):
    x, y  : posición del centro de masas de la pluma (y = altura sobre el suelo)
    alpha : ángulo de la pluma respecto a la horizontal (antihorario)
    phi   : ángulo del cable respecto a la vertical descendente (positivo hacia +x)

Lagrangiana:
    L = 1/2 (M+m)(xd^2 + yd^2) + 1/24 M L^2 alphad^2 + 1/2 m l^2 phid^2
        + m l phid (xd cos(phi) + yd sin(phi)) - (M+m) g y + m g l cos(phi)

Uso:
    python simulacion.py

Salida:
    animaciones/<caso>.gif   animación de cada caso (con la torre de la grúa)
    graficas/<caso>.png      las tres primeras gráficas de cada caso
    resultados.png           las seis gráficas del caso principal
    animacion.gif            copia de la animación del caso principal
"""
import os
import shutil

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# 1. PARÁMETROS DEL SISTEMA (valores base; cada caso puede cambiar alguno)

PARAMS_BASE = dict(
    M=500.0,   # masa de la pluma [kg]
    m=100.0,   # masa del gancho [kg]
    L=10.0,    # longitud de la pluma [m]
    l=3.0,     # longitud del cable [m]  (la "l" minúscula del enunciado)
    H=40.0,    # altura inicial de la pluma sobre el suelo [m]
    g=9.81,    # gravedad [m/s^2]
)

# 2. CONDICIONES INICIALES Y CASOS
#    CI_BASE son las condiciones iniciales por defecto. Cada caso indica solo
#    lo que cambia:
#       ci     -> condiciones iniciales distintas de CI_BASE
#       params -> parámetros distintos de PARAMS_BASE
#    Para añadir un caso nuevo, copia una entrada y cambia lo que quieras.

CI_BASE = dict(phi0=np.radians(20), dphi0=0.0, dalpha0=0.0, vx0=0.0, vy0=0.0)

# Condiciones iniciales del caso "giro" (se reutilizan al variar los parámetros)
GIRO = dict(dphi0=1.0, dalpha0=0.4, vx0=0.5)

CASOS = {
    # --- Distintas condiciones iniciales (con los parámetros base) ---------
    "enunciado": dict(
        desc="Todo en reposo; solo el gancho está desviado 20° de la vertical",
        ci={}, params={}),
    "giro": dict(
        desc="Pluma y cable giran y hay algo de velocidad horizontal (caso principal)",
        ci=GIRO, params={}),
    "cable_girando": dict(
        desc="Solo gira el cable al empezar (dphi0 = 3 rad/s)",
        ci=dict(dphi0=3.0), params={}),
    "pluma_girando": dict(
        desc="Solo gira la pluma al empezar (dalpha0 = 1.5 rad/s)",
        ci=dict(dalpha0=1.5), params={}),
    "tiro_oblicuo": dict(
        desc="La pluma sale lanzada hacia arriba y hacia +x (vx0 = 6 m/s, vy0 = 8 m/s)",
        ci=dict(vx0=6.0, vy0=8.0, dphi0=1.0), params={}),
    # --- Casos límite -------------------------------------------------------
    "caida_libre": dict(
        desc="Límite: cable vertical y todo en reposo; cae como un solo cuerpo (T = 0)",
        ci=dict(phi0=0.0), params={}),
    "giro_rapido": dict(
        desc="Límite: el cable gira muy deprisa (dphi0 = 8 rad/s), tensión muy alta",
        ci=dict(dphi0=8.0), params={}),
    # --- Distintos parámetros (con las condiciones iniciales de "giro") ----
    "pluma_pesada": dict(
        desc="Pluma 10 veces más pesada (M = 5000 kg)",
        ci=GIRO, params=dict(M=5000.0)),
    "gancho_pesado": dict(
        desc="Gancho muy pesado (m = 2000 kg)",
        ci=GIRO, params=dict(m=2000.0)),
    "gancho_ligero": dict(
        desc="Gancho muy ligero (m = 5 kg)",
        ci=GIRO, params=dict(m=5.0)),
    "cable_largo": dict(
        desc="Cable largo (l = 15 m)",
        ci=GIRO, params=dict(l=15.0)),
    "pluma_larga": dict(
        desc="Pluma larga (L = 30 m)",
        ci=GIRO, params=dict(L=30.0)),
    "altura_baja": dict(
        desc="Caída desde poca altura (H = 10 m)",
        ci=GIRO, params=dict(H=10.0)),
    "luna": dict(
        desc="Caída en la Luna (g = 1.62 m/s²)",
        ci=GIRO, params=dict(g=1.62)),
}

CASO_PRINCIPAL = "giro"           # el que da resultados.png y animacion.gif
CASOS_A_GENERAR = list(CASOS)     # quita nombres de esta lista para generar menos

# Torre de la grúa que se dibuja en las animaciones (solo decorativa)
TORRE_LADO = "izquierda"          # "izquierda" o "derecha" de la pluma
TORRE_ANCHO = 1.0                 # [m]


def completar(caso):
    """Devuelve (condiciones iniciales, parámetros) completos de un caso."""
    ci = {**CI_BASE, **caso.get("ci", {})}
    P = {**PARAMS_BASE, **caso.get("params", {})}
    return ci, P


# 3. ECUACIONES DE MOVIMIENTO (Euler-Lagrange)
#    Estado s = [x, y, alpha, phi, xd, yd, alphad, phid]
#    Las ecuaciones de x, y, phi forman un sistema lineal en las aceleraciones:
#      (M+m) xdd + m l cos(phi) phidd = m l phid^2 sin(phi)
#      (M+m) ydd + m l sin(phi) phidd = -(M+m) g - m l phid^2 cos(phi)
#      cos(phi) xdd + sin(phi) ydd + l phidd = -g sin(phi)
#    y la de alpha es: alphadd = 0

def aceleraciones(s, P):
    M, m, l, g = P["M"], P["m"], P["l"], P["g"]
    phi, dphi = s[3], s[7]
    c, sn = np.cos(phi), np.sin(phi)
    A = np.array([[M + m, 0.0, m * l * c],
                  [0.0, M + m, m * l * sn],
                  [c, sn, l]])
    b = np.array([m * l * dphi**2 * sn,
                  -(M + m) * g - m * l * dphi**2 * c,
                  -g * sn])
    return np.linalg.solve(A, b)  # xdd, ydd, phidd


def rhs(t, s, P):
    xdd, ydd, phidd = aceleraciones(s, P)
    return [s[4], s[5], s[6], s[7], xdd, ydd, 0.0, phidd]


def simular(ci, P):
    L, l, H = P["L"], P["l"], P["H"]

    def choca_con_el_suelo(t, s):
        """Se anula cuando el punto más bajo (extremo de la pluma o gancho) llega al suelo."""
        y, alpha, phi = s[1], s[2], s[3]
        return min(y - 0.5 * L * abs(np.sin(alpha)), y - l * np.cos(phi))

    choca_con_el_suelo.terminal = True
    choca_con_el_suelo.direction = -1

    s0 = [0.0, H, 0.0, ci["phi0"], ci["vx0"], ci["vy0"], ci["dalpha0"], ci["dphi0"]]
    return solve_ivp(lambda t, s: rhs(t, s, P), (0.0, 20.0), s0,
                     events=choca_con_el_suelo,
                     rtol=1e-10, atol=1e-12, dense_output=True)


# 4. MAGNITUDES DERIVADAS Y COMPROBACIONES

def constantes(s, P):
    """Devuelve p_x, p_alpha y la energía E para cada instante (s: array 8 x N)."""
    M, m, L, l, g = P["M"], P["m"], P["L"], P["l"], P["g"]
    x, y, al, phi, xd, yd, ad, pd = s
    px = (M + m) * xd + m * l * pd * np.cos(phi)
    pa = M * L**2 / 12.0 * ad
    E = (0.5 * (M + m) * (xd**2 + yd**2) + M * L**2 / 24.0 * ad**2
         + 0.5 * m * l**2 * pd**2
         + m * l * pd * (xd * np.cos(phi) + yd * np.sin(phi))
         + (M + m) * g * y - m * g * l * np.cos(phi))
    return px, pa, E


def tension(s, P):
    """Tensión del cable, calculada a partir de la 2ª ley de Newton del gancho."""
    m, l, g = P["m"], P["l"], P["g"]
    T = np.empty(s.shape[1])
    for k in range(s.shape[1]):
        xdd, ydd, _ = aceleraciones(s[:, k], P)
        phi, pd = s[3, k], s[7, k]
        T[k] = m * (-xdd * np.sin(phi) + (ydd + g) * np.cos(phi) + l * pd**2)
    return T


def posiciones(s, P):
    L, l = P["L"], P["l"]
    x, y, al, phi = s[0], s[1], s[2], s[3]
    ext1 = (x - 0.5 * L * np.cos(al), y - 0.5 * L * np.sin(al))  # extremo izquierdo
    ext2 = (x + 0.5 * L * np.cos(al), y + 0.5 * L * np.sin(al))  # extremo derecho
    gancho = (x + l * np.sin(phi), y - l * np.cos(phi))
    return ext1, ext2, gancho


def analizar(sol, P):
    """Errores de las constantes del movimiento y comparación con la solución analítica."""
    M, m, l, g = P["M"], P["m"], P["l"], P["g"]
    t = np.linspace(0, sol.t[-1], 2000)
    s = sol.sol(t)

    def err_rel(q):
        ref = abs(q[0]) if abs(q[0]) > 1e-12 else 1.0
        return np.max(np.abs(q - q[0])) / ref

    px, pa, E = constantes(s, P)

    # Solución analítica del apartado c):
    #   CM del sistema (X, Y): movimiento uniforme en X, caída libre en Y
    #   phi: rotación uniforme
    mu = m * l / (M + m)
    X = s[0] + mu * np.sin(s[3])
    Y = s[1] - mu * np.cos(s[3])
    X_an = X[0] + (s[4, 0] + mu * np.cos(s[3, 0]) * s[7, 0]) * t
    Y_an = Y[0] + (s[5, 0] + mu * np.sin(s[3, 0]) * s[7, 0]) * t - 0.5 * g * t**2
    phi_an = s[3, 0] + s[7, 0] * t

    T = tension(s, P)
    T_an = M * m / (M + m) * l * s[7]**2

    return dict(
        t_f=sol.t[-1],
        err_px=err_rel(px), err_pa=err_rel(pa), err_E=err_rel(E),
        dif_an=max(np.max(np.abs(X - X_an)), np.max(np.abs(Y - Y_an)),
                   np.max(np.abs(s[3] - phi_an))),
        T_min=float(np.round(T.min(), 6)) + 0.0,   # "+ 0.0" evita escribir -0.000
        dif_T=np.max(np.abs(T - T_an)),
    )


# 5. GRÁFICAS

def paneles_basicos(ax0, ax1, ax2, t, s, P):
    """Las tres primeras gráficas: centro de masas, ángulos y trayectorias."""
    ext1, ext2, gancho = posiciones(s, P)

    ax0.plot(t, s[0], label="x (CM)")
    ax0.plot(t, s[1], label="y (CM)")
    ax0.set(xlabel="t [s]", ylabel="posición [m]", title="Centro de masas de la pluma")
    ax0.legend()

    ax1.plot(t, np.degrees(s[2]), label="α (pluma)")
    ax1.plot(t, np.degrees(s[3]), label="φ (cable)")
    ax1.set(xlabel="t [s]", ylabel="ángulo [°]", title="Ángulos")
    ax1.legend()

    ax2.plot(ext1[0], ext1[1], label="extremo izq.")
    ax2.plot(ext2[0], ext2[1], label="extremo der.")
    ax2.plot(gancho[0], gancho[1], label="gancho")
    ax2.axhline(0, color="k", lw=1)
    ax2.set(xlabel="x [m]", ylabel="y [m]", title="Trayectorias")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.legend()


def graficas_basicas(sol, P, nombre, desc, fichero):
    """Figura con las tres primeras gráficas de un caso."""
    t = np.linspace(0, sol.t[-1], 600)
    s = sol.sol(t)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    paneles_basicos(ax[0], ax[1], ax[2], t, s, P)
    fig.suptitle(f"Caso «{nombre}»: {desc}")
    fig.tight_layout()
    fig.savefig(fichero, dpi=110)
    return fig


def graficas_completas(sol, P, fichero="resultados.png"):
    """Las seis gráficas (se usan para el caso principal)."""
    t = np.linspace(0, sol.t[-1], 600)
    s = sol.sol(t)
    px, pa, E = constantes(s, P)
    T = tension(s, P)

    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    paneles_basicos(ax[0, 0], ax[0, 1], ax[0, 2], t, s, P)

    for q, nom in ((px, "p_x"), (pa, "p_α"), (E, "E")):
        ref = q[0] if abs(q[0]) > 1e-12 else 1.0
        ax[1, 0].semilogy(t, np.abs(q - q[0]) / abs(ref) + 1e-18, label=nom)
    ax[1, 0].set(xlabel="t [s]", ylabel="|q(t) − q(0)| / |q(0)|",
                 title="Conservación (error relativo)")
    ax[1, 0].legend()

    ax[1, 1].plot(t, T)
    ax[1, 1].axhline(0, color="k", lw=1)
    ax[1, 1].set(xlabel="t [s]", ylabel="T [N]", title="Tensión del cable")

    ax[1, 2].plot(t, s[3] - s[3, 0], label="φ(t) − φ(0)")
    ax[1, 2].plot(t, s[7, 0] * t, "--", label="ω₀·t (uniforme)")
    ax[1, 2].set(xlabel="t [s]", ylabel="[rad]", title="φ: comparación con rotación uniforme")
    ax[1, 2].legend()

    fig.tight_layout()
    fig.savefig(fichero, dpi=130)
    return fig


# 6. ANIMACIÓN (con la torre de la grúa)

def animar(sol, P, nombre, fichero, fps=30):
    L, H = P["L"], P["H"]
    n = max(2, int(sol.t[-1] * fps))
    t = np.linspace(0, sol.t[-1], n)
    s = sol.sol(t)
    ext1, ext2, gancho = posiciones(s, P)

    # Torre: una barra vertical pegada a un extremo de la pluma, desde el suelo
    # hasta un poco por encima de la altura inicial de la pluma.
    alto_torre = H + 1.5
    if TORRE_LADO == "izquierda":
        torre_x0 = -0.5 * L - TORRE_ANCHO
    else:
        torre_x0 = 0.5 * L
    torre_x1 = torre_x0 + TORRE_ANCHO

    fig, ax = plt.subplots(figsize=(6, 7))
    xmin = min(ext1[0].min(), ext2[0].min(), gancho[0].min(), torre_x0) - 3
    xmax = max(ext1[0].max(), ext2[0].max(), gancho[0].max(), torre_x1) + 3
    ymax = max(ext1[1].max(), ext2[1].max(), gancho[1].max(), alto_torre) + 4
    xc = 0.5 * (xmin + xmax)
    semi = max(0.5 * (xmax - xmin), 0.3 * (ymax + 2))
    ax.set_xlim(xc - semi, xc + semi)
    ax.set_ylim(-2, ymax)
    ax.set_aspect("equal")
    ax.axhline(0, color="saddlebrown", lw=3)
    ax.add_patch(Rectangle((torre_x0, 0), TORRE_ANCHO, alto_torre,
                           facecolor="dimgray", edgecolor="black", zorder=1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Pluma de grúa en caída libre (caso: {nombre})")

    pluma, = ax.plot([], [], "-", color="goldenrod", lw=6, solid_capstyle="butt")
    cable, = ax.plot([], [], "-", color="gray", lw=1.5)
    cm, = ax.plot([], [], "ko", ms=4)
    hook, = ax.plot([], [], "o", color="crimson", ms=9)
    traza, = ax.plot([], [], ":", color="crimson", lw=1)
    texto = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top")

    def init():
        for a in (pluma, cable, cm, hook, traza):
            a.set_data([], [])
        texto.set_text("")
        return pluma, cable, cm, hook, traza, texto

    def update(k):
        pluma.set_data([ext1[0][k], ext2[0][k]], [ext1[1][k], ext2[1][k]])
        cable.set_data([s[0, k], gancho[0][k]], [s[1, k], gancho[1][k]])
        cm.set_data([s[0, k]], [s[1, k]])
        hook.set_data([gancho[0][k]], [gancho[1][k]])
        traza.set_data(gancho[0][:k + 1], gancho[1][:k + 1])
        texto.set_text(f"t = {t[k]:.2f} s")
        return pluma, cable, cm, hook, traza, texto

    ani = FuncAnimation(fig, update, frames=n, init_func=init, blit=True, interval=1000 / fps)
    ani.save(fichero, writer=PillowWriter(fps=fps), dpi=70)
    return fig


# 7. PROGRAMA PRINCIPAL

if __name__ == "__main__":
    os.makedirs("animaciones", exist_ok=True)
    os.makedirs("graficas", exist_ok=True)

    resumen = []
    fig_principal = None
    for k, nombre in enumerate(CASOS_A_GENERAR, start=1):
        caso = CASOS[nombre]
        ci, P = completar(caso)
        print(f"[{k}/{len(CASOS_A_GENERAR)}] {nombre}: {caso['desc']}")

        sol = simular(ci, P)
        resumen.append((nombre, analizar(sol, P)))

        fig = graficas_basicas(sol, P, nombre, caso["desc"], f"graficas/{nombre}.png")
        plt.close(fig)
        fig = animar(sol, P, nombre, f"animaciones/{nombre}.gif")
        plt.close(fig)

        if nombre == CASO_PRINCIPAL:
            fig_principal = graficas_completas(sol, P, "resultados.png")
            shutil.copyfile(f"animaciones/{nombre}.gif", "animacion.gif")

    print("\nResumen de los casos (errores relativos máximos; T = 0 significa cable justo sin tensión):")
    cab = (f"{'caso':15s} {'t suelo [s]':>11s} {'err. E':>9s} {'err. p_x':>9s} "
           f"{'err. p_alpha':>12s} {'dif. analít.':>12s} {'T mín [N]':>10s}")
    print(cab)
    print("-" * len(cab))
    for nombre, r in resumen:
        print(f"{nombre:15s} {r['t_f']:11.3f} {r['err_E']:9.1e} {r['err_px']:9.1e} "
              f"{r['err_pa']:12.1e} {r['dif_an']:12.1e} {r['T_min']:10.3f}")

    print("\nGenerados: animaciones/, graficas/"
          + (", resultados.png y animacion.gif" if fig_principal is not None else ""))
    if fig_principal is not None:
        plt.show()
