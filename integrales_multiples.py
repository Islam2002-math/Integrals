#!/usr/bin/env python3
"""
Intégrales simples, doubles et triples : de l'aire au centre de masse.
 - Intégrale simple  : somme de Riemann de sin(x) sur [0, pi]            (exact = 2)
 - Intégrale double  : volume sous z = sin(x) sin(y) sur [0, pi]^2        (exact = 4)
 - Intégrale triple  : sphère de rayon 1, densité rho = 1 + z, Monte Carlo
       masse M = 4*pi/3, centre de masse z_G = 1/5, moment d'inertie Iz = 8*pi/15
Génère : vidéo MP4, figure d'ordres de convergence, tableau de résultats.

Usage : python integrales_multiples.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy import integrate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

PI = math.pi
EXACT_1D = 2.0
EXACT_2D = 4.0
EXACT_M = 4 * PI / 3
EXACT_ZG = 1 / 5
EXACT_IZ = 8 * PI / 15


def fr(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


# ----------------------------------------------------------------------------
# 1. Méthodes numériques
# ----------------------------------------------------------------------------
def left_riemann_1d(n):
    h = PI / n
    return h * np.sum(np.sin(np.arange(n) * h))


def midpoint_1d(n):
    h = PI / n
    return h * np.sum(np.sin((np.arange(n) + 0.5) * h))


def simpson_1d(n):  # n pair
    x = np.linspace(0, PI, n + 1)
    y = np.sin(x)
    h = PI / n
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


def midpoint_2d(n):
    h = PI / n
    m = (np.arange(n) + 0.5) * h
    return h * h * np.sum(np.sin(m)[:, None] * np.sin(m)[None, :])


def midpoint_3d(n):  # integrale de sin x sin y sin z sur [0, pi]^3, exact = 8
    h = PI / n
    s = np.sum(np.sin((np.arange(n) + 0.5) * h))
    return (h * s) ** 3  # le produit se factorise : on garde n^3 évaluations "en pensée"


def midpoint_3d_full(n):
    h = PI / n
    m = (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(m, m, m, indexing="ij")
    return h ** 3 * np.sum(np.sin(X) * np.sin(Y) * np.sin(Z))


def mc_sphere(rng, n):
    """Estimations Monte Carlo (cube [-1,1]^3, volume 8) pour la sphère de densité 1+z."""
    P = rng.uniform(-1, 1, (n, 3))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    inside = (x * x + y * y + z * z) <= 1.0
    rho = (1 + z) * inside
    M = 8 * rho.sum() / n
    zg = (rho * z).sum() / rho.sum()
    Iz = 8 * (rho * (x * x + y * y)).sum() / n
    return M, zg, Iz


# ----------------------------------------------------------------------------
# 2. Vérification des valeurs exactes avec scipy (tplquad / dblquad / quad)
# ----------------------------------------------------------------------------
def verify_exact():
    q1 = integrate.quad(np.sin, 0, PI)[0]
    q2 = integrate.dblquad(lambda y, x: math.sin(x) * math.sin(y), 0, PI, 0, PI)[0]
    lim_y = lambda x: math.sqrt(max(0.0, 1 - x * x))
    lim_z = lambda x, y: math.sqrt(max(0.0, 1 - x * x - y * y))
    kw = dict(epsabs=1e-8, epsrel=1e-8)
    M = integrate.tplquad(lambda z, y, x: 1 + z, -1, 1, lambda x: -lim_y(x), lim_y,
                          lambda x, y: -lim_z(x, y), lim_z, **kw)[0]
    Mz = integrate.tplquad(lambda z, y, x: z * (1 + z), -1, 1, lambda x: -lim_y(x), lim_y,
                           lambda x, y: -lim_z(x, y), lim_z, **kw)[0]
    Iz = integrate.tplquad(lambda z, y, x: (x * x + y * y) * (1 + z), -1, 1, lambda x: -lim_y(x), lim_y,
                           lambda x, y: -lim_z(x, y), lim_z, **kw)[0]
    return q1, q2, M, Mz / M, Iz


# ----------------------------------------------------------------------------
# 3. Étude des ordres de convergence (figure statique)
# ----------------------------------------------------------------------------
def slope(N, err, lo=0):
    N, err = np.array(N, float)[lo:], np.array(err, float)[lo:]
    ok = err > 1e-14
    return np.polyfit(np.log(N[ok]), np.log(err[ok]), 1)[0]


def convergence_study(path):
    ns = [2 ** k for k in range(1, 11)]  # 2 ... 1024
    e_left = [abs(left_riemann_1d(n) - EXACT_1D) for n in ns]
    e_mid = [abs(midpoint_1d(n) - EXACT_1D) for n in ns]
    e_simp = [abs(simpson_1d(n) - EXACT_1D) for n in ns]

    # erreur en fonction du nombre d'évaluations N, pour d = 1, 2, 3
    n1 = [2 ** k for k in range(2, 13)]
    n2 = [2 ** k for k in range(1, 7)]
    n3 = [2 ** k for k in range(1, 7)]
    N1 = n1
    E1 = [abs(midpoint_1d(n) - 2) for n in n1]
    N2 = [n * n for n in n2]
    E2 = [abs(midpoint_2d(n) - 4) for n in n2]
    N3 = [n ** 3 for n in n3]
    E3 = [abs(midpoint_3d_full(n) - 8) for n in n3]

    # Monte Carlo (d = 3, mêmes intégrande) : moyenne sur 40 tirages
    rng = np.random.default_rng(1)
    NM = [int(v) for v in np.logspace(2, 5.5, 10)]
    EM = []
    for n in NM:
        errs = []
        for _ in range(40):
            P = rng.uniform(0, PI, (n, 3))
            est = PI ** 3 * np.mean(np.sin(P[:, 0]) * np.sin(P[:, 1]) * np.sin(P[:, 2]))
            errs.append(abs(est - 8))
        EM.append(np.mean(errs))

    s = {
        "gauche (1D)": slope(ns, e_left, 2), "point milieu (1D)": slope(ns, e_mid, 2),
        "Simpson (1D)": slope(ns, e_simp, 2),
        "grille d=1": slope(N1, E1, 2), "grille d=2": slope(N2, E2, 1),
        "grille d=3": slope(N3, E3, 1), "Monte Carlo d=3": slope(NM, EM, 0),
    }

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)
    ax = axs[0]
    ax.loglog(ns, e_left, "o-", label=f"gauche  (pente {s['gauche (1D)']:.2f})", color="#e67e22")
    ax.loglog(ns, e_mid, "s-", label=f"point milieu  (pente {s['point milieu (1D)']:.2f})", color="#2980b9")
    ax.loglog(ns, e_simp, "^-", label=f"Simpson  (pente {s['Simpson (1D)']:.2f})", color="#27ae60")
    ax.set_xlabel("n (nombre de sous-intervalles)")
    ax.set_ylabel("erreur absolue")
    ax.set_title("Intégrale simple : ∫₀^π sin x dx")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axs[1]
    ax.loglog(N1, E1, "o-", label=f"grille d=1  (pente {s['grille d=1']:.2f})", color="#2980b9")
    ax.loglog(N2, E2, "s-", label=f"grille d=2  (pente {s['grille d=2']:.2f})", color="#8e44ad")
    ax.loglog(N3, E3, "^-", label=f"grille d=3  (pente {s['grille d=3']:.2f})", color="#c0392b")
    ax.loglog(NM, EM, "d--", label=f"Monte Carlo d=3  (pente {s['Monte Carlo d=3']:.2f})", color="#27ae60")
    ax.set_xlabel("N (nombre d'évaluations de la fonction)")
    ax.set_ylabel("erreur absolue")
    ax.set_title("Malédiction de la dimension (point milieu vs Monte Carlo)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return s


# ----------------------------------------------------------------------------
# 4. Vidéo
# ----------------------------------------------------------------------------
def make_video(path, only=None):
    rng = np.random.default_rng(3)
    NMAX = 200_000
    P = rng.uniform(-1, 1, (NMAX, 3))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    inside = (x * x + y * y + z * z) <= 1.0
    rho = (1 + z) * inside
    cM = np.cumsum(rho)
    cZ = np.cumsum(rho * z)
    cX = np.cumsum(rho * x)
    cY = np.cumsum(rho * y)
    cI = np.cumsum(rho * (x * x + y * y))

    # --- plan des images ---
    n1_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100]
    n2_list = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20]
    N3_list = sorted(set(int(v) for v in np.logspace(2, math.log10(NMAX), 70)))
    frames = []
    frames += [("a1", n, i) for i, n in enumerate(n1_list) for _ in range(3)]
    frames += [("a2", n, i) for i, n in enumerate(n2_list) for _ in range(6)]
    frames += [("a3", n, i) for i, n in enumerate(N3_list) for _ in range(2)]
    frames += [("out", k, k) for k in range(60)]
    if only:
        frames = [fr_ for fr_ in frames if fr_[0] in only]
    counters = {"a2": 0, "a3": 0}

    fig = plt.figure(figsize=(10, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], left=0.09, right=0.96,
                          top=0.91, bottom=0.08, hspace=0.28)
    ax2 = fig.add_subplot(gs[0])
    ax3 = fig.add_subplot(gs[0], projection="3d")
    axb = fig.add_subplot(gs[1])
    axt = fig.add_axes([0, 0, 1, 1])
    note3 = fig.text(0.96, 0.925, "✕ = centre de masse estimé", ha="right", fontsize=9)
    note3.set_visible(False)

    def bottom_curve(xs, ys_list, labels, colors, exact_list, xlabel, ylabel, ylim=None, logx=True, xlim=None):
        axb.clear()
        for ys, lb, c in zip(ys_list, labels, colors):
            axb.plot(xs, ys, "-o" if len(xs) < 30 else "-", color=c, lw=2, ms=3.5, label=lb)
        for e in exact_list:
            axb.axhline(e, color="k", ls="--", lw=1)
        if logx:
            axb.set_xscale("log")
        if ylim:
            axb.set_ylim(*ylim)
        if xlim:
            axb.set_xlim(*xlim)
        axb.set_xlabel(xlabel, fontsize=8)
        axb.set_ylabel(ylabel, fontsize=8)
        axb.tick_params(labelsize=7)
        axb.grid(alpha=0.25, which="both")
        if labels[0]:
            axb.legend(fontsize=7.5, loc="lower right", frameon=False)

    S1 = [left_riemann_1d(n) for n in n1_list]
    V2 = [midpoint_2d(n) for n in n2_list]

    def update(f):
        kind, a, idx = frames[f]
        ax2.set_visible(kind == "a1")
        ax3.set_visible(kind in ("a2", "a3"))
        axb.set_visible(kind != "out")
        axt.set_visible(kind == "out")
        note3.set_visible(kind == "a3")

        if kind == "a1":
            n = a
            h = PI / n
            ax2.clear()
            xx = np.linspace(0, PI, 400)
            xl = np.arange(n) * h
            ax2.bar(xl, np.sin(xl), width=h, align="edge", color="#4c78a8", alpha=0.65,
                    edgecolor="#1f3b5c", lw=0.8)
            ax2.plot(xx, np.sin(xx), "k", lw=2)
            ax2.set_xlim(-0.1, PI + 0.1)
            ax2.set_ylim(-0.05, 1.2)
            ax2.set_xlabel("x", fontsize=8)
            ax2.set_ylabel("sin x", fontsize=8)
            ax2.tick_params(labelsize=7)
            ax2.text(0.04, 0.88, r"$\int_0^{\pi}\sin x\,dx = 2$", transform=ax2.transAxes, fontsize=14)
            S = S1[idx]
            bottom_curve(n1_list[:idx + 1], [S1[:idx + 1]], ["somme de Riemann $S_n$"], ["#2980b9"], [2.0],
                         "n (nombre de rectangles)", "aire", ylim=(-0.1, 2.3), xlim=(0.9, 110))
            title = f"Intégrale simple  |  n = {n}  |  S_n = {fr(S, 4)}  |  exact = 2"
        elif kind == "a2":
            n = a
            c = counters["a2"]
            counters["a2"] += 1
            h = PI / n
            m = (np.arange(n) + 0.5) * h
            X, Y = np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing="ij")
            Zh = np.sin(m)[:, None] * np.sin(m)[None, :]
            ax3.clear()
            ax3.bar3d(X.ravel(), Y.ravel(), np.zeros(n * n), h, h, Zh.ravel(),
                      color=plt.cm.viridis(Zh.ravel()), edgecolor="k", linewidth=0.25, shade=True, alpha=0.95)
            ax3.set_xlim(0, PI)
            ax3.set_ylim(0, PI)
            ax3.set_zlim(0, 1.1)
            ax3.set_box_aspect((1, 1, 0.55), zoom=1.22)
            ax3.view_init(elev=28, azim=-62 + 0.45 * c)
            ax3.tick_params(labelsize=6)
            ax3.set_xlabel("x", fontsize=8)
            ax3.set_ylabel("y", fontsize=8)
            ax3.text2D(0.03, 0.9, r"$\iint\,\sin x\,\sin y\;dx\,dy = 4$", transform=ax3.transAxes, fontsize=14)
            V = V2[idx]
            bottom_curve([k * k for k in n2_list[:idx + 1]], [V2[:idx + 1]], ["volume $V_n$"], ["#8e44ad"], [4.0],
                         "nombre de colonnes (n²)", "volume", ylim=(0, 10.5), xlim=(0.9, 450))
            title = f"Intégrale double  |  {n}×{n} = {n * n} colonnes  |  V_n = {fr(V, 4)}  |  exact = 4"
        elif kind == "a3":
            N = a
            c = counters["a3"]
            counters["a3"] += 1
            show = min(N, 5000)
            xs, ys, zs, ins = x[:show], y[:show], z[:show], inside[:show]
            ax3.clear()
            ax3.scatter(xs[~ins], ys[~ins], zs[~ins], s=1, color="#999", alpha=0.12, depthshade=False)
            ax3.scatter(xs[ins], ys[ins], zs[ins], s=5, c=zs[ins], cmap="coolwarm", vmin=-1, vmax=1,
                        depthshade=True, alpha=0.9)
            Mn = 8 * cM[N - 1] / N
            zgn = cZ[N - 1] / cM[N - 1]
            xgn = cX[N - 1] / cM[N - 1]
            ygn = cY[N - 1] / cM[N - 1]
            In = 8 * cI[N - 1] / N
            ax3.scatter([xgn], [ygn], [zgn], s=170, color="#111", marker="X", edgecolors="white", linewidths=0.8, depthshade=False)
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
            ax3.set_zlim(-1, 1)
            ax3.set_box_aspect((1, 1, 1), zoom=1.1)
            ax3.view_init(elev=22, azim=-55 + 0.8 * c)
            ax3.tick_params(labelsize=6)
            ax3.set_xlabel("x", fontsize=8)
            ax3.set_ylabel("y", fontsize=8)
            ax3.set_zlabel("z", fontsize=8)
            ax3.text2D(0.03, 0.9, r"$\iiint\,\rho\,dV,\ \rho = 1+z$", transform=ax3.transAxes, fontsize=14)
            k = np.searchsorted(N3_list, N) + 1
            Ns = N3_list[:k]
            rM = [8 * cM[n - 1] / n / EXACT_M for n in Ns]
            rZ = [cZ[n - 1] / cM[n - 1] / EXACT_ZG for n in Ns]
            rI = [8 * cI[n - 1] / n / EXACT_IZ for n in Ns]
            bottom_curve(Ns, [rM, rZ, rI], ["masse M / exacte", "centre de masse z_G / exact",
                                            "moment d'inertie Iz / exact"],
                         ["#27ae60", "#c0392b", "#2980b9"], [1.0], "N (points tirés)",
                         "estimation / valeur exacte", ylim=(0.5, 1.5))
            axb.set_xlim(100, NMAX)
            title = (f"Intégrale triple (Monte Carlo)  |  N = {N:,}".replace(",", " ") +
                     f"  |  M = {fr(Mn, 3)}  |  z_G = {fr(zgn, 3)}  |  Iz = {fr(In, 3)}")
        else:  # outro
            k = idx
            axt.clear()
            axt.axis("off")
            axt.set_xlim(0, 1)
            axt.set_ylim(0, 1)
            axt.text(0.5, 0.92, "Que peut-on calculer avec une intégrale ?", ha="center", fontsize=17, weight="bold")
            cols = [
                (0.19, r"$\int$  simple", "#2980b9",
                 ["aire sous une courbe", "longueur d'une courbe", "travail d'une force",
                  "probabilité (densité)"],
                 r"$\int_0^{\pi}\sin x\,dx = 2$" + f"\nS₁₀₀ = {fr(S1[-1], 4)}", 0),
                (0.50, r"$\iint$  double", "#8e44ad",
                 ["volume sous une surface", "masse d'une plaque", "centre de gravité",
                  "moyenne d'un champ 2D"],
                 r"$\iint\sin x\,\sin y\,dx\,dy = 4$" + f"\nV₂₀ = {fr(V2[-1], 4)}", 12),
                (0.81, r"$\iiint$  triple", "#c0392b",
                 ["masse d'un solide", "centre de masse", "moment d'inertie",
                  "volume d'un solide"],
                 f"sphère, ρ = 1 + z\nM = 4π/3 ≈ {fr(EXACT_M, 3)}\nz_G = 1/5, Iz = 8π/15 ≈ {fr(EXACT_IZ, 3)}", 24),
            ]
            for cx, head, col, items, res, start in cols:
                if k < start:
                    continue
                a_ = min(1.0, (k - start + 1) / 6)
                axt.text(cx, 0.80, head, ha="center", fontsize=19, color=col, weight="bold", alpha=a_)
                for j, it in enumerate(items):
                    axt.text(cx, 0.70 - 0.075 * j, "• " + it, ha="center", fontsize=11.5, alpha=a_)
                axt.text(cx, 0.34, res, ha="center", va="top", fontsize=10.5, color=col, alpha=a_,
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=col, lw=1.2))
            if k >= 38:
                a_ = min(1.0, (k - 37) / 8)
                axt.text(0.5, 0.11, "Erreur en fonction de N évaluations :  1/N² (d=1)  →  1/N (d=2)  →  1/N^(2/3) (d=3)\n"
                                    "Monte Carlo : 1/√N, quelle que soit la dimension",
                         ha="center", fontsize=11, weight="bold", alpha=a_)
            title = ""

        fig.suptitle(title, fontsize=11, weight="bold", y=0.965)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=50)
    writer = FFMpegWriter(fps=20, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    anim.save(path, writer=writer)
    plt.close(fig)
    return S1[-1], V2[-1]


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
PARTS = ["a1", "a2", "a3", "out"]


def render_part(part):
    make_video(os.path.join(OUT, f"part_{part}.mp4"), only=[part])


def concat_parts():
    lst = os.path.join(OUT, "parts.txt")
    with open(lst, "w") as f:
        for p in PARTS:
            f.write(f"file 'part_{p}.mp4'\n")
    os.system(f"ffmpeg -v error -y -f concat -safe 0 -i {lst} -c copy "
              f"{os.path.join(OUT, 'integrales_multiples.mp4')}")
    for p in PARTS:
        os.remove(os.path.join(OUT, f"part_{p}.mp4"))
    os.remove(lst)


def finish():
    print("Vérification des valeurs exactes (scipy)...")
    q1, q2, M, zg, Iz = verify_exact()
    print(f"  ∫ = {q1:.6f}  ∬ = {q2:.6f}  M = {M:.6f}  zG = {zg:.6f}  Iz = {Iz:.6f}")

    print("Étude des ordres de convergence...")
    slopes = convergence_study(os.path.join(OUT, "ordres_de_convergence.png"))
    for k, v in slopes.items():
        print(f"  pente {k:20s}: {v:6.2f}")

    # Monte Carlo : précision à N = 200 000 (même graine que la vidéo)
    rng = np.random.default_rng(3)
    Pm = rng.uniform(-1, 1, (200_000, 3))
    xx, yy, zz = Pm.T
    ins = (xx ** 2 + yy ** 2 + zz ** 2) <= 1
    rr = (1 + zz) * ins
    Mmc = 8 * rr.sum() / 200_000
    zmc = (rr * zz).sum() / rr.sum()
    Imc = 8 * (rr * (xx ** 2 + yy ** 2)).sum() / 200_000
    s100 = left_riemann_1d(100)
    v20 = midpoint_2d(20)

    with open(os.path.join(OUT, "resultats.md"), "w", encoding="utf-8") as f:
        f.write("| Intégrale | Exact | Numérique (vidéo) | Erreur relative | Contrôle scipy |\n|---|---|---|---|---|\n")
        rows = [
            ("∫₀^π sin x dx (somme de Riemann, n=100)", 2.0, s100, q1),
            ("∬ sin x sin y (colonnes 20×20)", 4.0, v20, q2),
            ("Masse M (Monte Carlo, N=200 000)", EXACT_M, Mmc, M),
            ("Centre de masse z_G (Monte Carlo)", EXACT_ZG, zmc, zg),
            ("Moment d'inertie Iz (Monte Carlo)", EXACT_IZ, Imc, Iz),
        ]
        for name, ex, num, sc in rows:
            f.write(f"| {name} | {ex:.5f} | {num:.5f} | {100 * abs(num - ex) / ex:.3f} % | {sc:.5f} |\n")
        f.write("\nPentes de convergence (log-log, erreur vs N) :\n\n")
        for k, v in slopes.items():
            f.write(f"- {k} : {v:.2f}\n")
    print(open(os.path.join(OUT, "resultats.md"), encoding="utf-8").read())


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "part":
        render_part(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "finish":
        concat_parts()
        finish()
    else:
        for p in PARTS:
            print("Rendu de la partie", p)
            render_part(p)
        concat_parts()
        finish()
    print("Terminé :", OUT)
