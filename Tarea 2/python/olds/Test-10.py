import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from multiprocessing import Pool
import copy

seed_global = 12345
np.random.seed(seed_global)

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def generate_scale_free_network(N, alpha, seed=None):
    print(f'[{timestamp()}][alpha={alpha}] Generando red scale-free N={N}')
    start = time.time()
    k_min, k_max = 1, int(np.sqrt(N))
    degrees = []
    while len(degrees) < N:
        k = np.random.zipf(alpha)
        if k_min <= k <= k_max:
            degrees.append(k)
    if sum(degrees) % 2 == 1:
        degrees[0] += 1
    G = nx.configuration_model(degrees, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f'[{timestamp()}][alpha={alpha}] Red creada en {time.time()-start:.2f}s')
    return G

def strong_rewire(G, target_rkk, max_iter_factor=10, tol=0.01, alpha=None):
    iter_count = 0
    current_rkk = nx.degree_assortativity_coefficient(G)
    print(f'[{timestamp()}][alpha={alpha}] Iniciando rewiring rkk={current_rkk:.4f} target={target_rkk}')
    start = time.time()
    while abs(current_rkk - target_rkk) > tol and iter_count < max_iter_factor * G.number_of_edges():
        edges = list(G.edges())
        M = len(edges)
        if M < 2:
            break
        idx1, idx2 = np.random.choice(M, 2, replace=False)
        e1, e2 = edges[idx1], edges[idx2]
        if len(set(e1 + e2)) == 4:
            G_temp = G.copy()
            G_temp.remove_edge(*e1)
            G_temp.remove_edge(*e2)
            G_temp.add_edge(e1[0], e2[1])
            G_temp.add_edge(e2[0], e1[1])
            new_rkk = nx.degree_assortativity_coefficient(G_temp)
            if abs(new_rkk - target_rkk) < abs(current_rkk - target_rkk):
                G.remove_edge(*e1)
                G.remove_edge(*e2)
                G.add_edge(e1[0], e2[1])
                G.add_edge(e2[0], e1[1])
                current_rkk = new_rkk
        iter_count += 1
    print(f'[{timestamp()}][alpha={alpha}] Rewiring final rkk={current_rkk:.4f} en {iter_count} iteraciones y {time.time()-start:.2f}s')
    return G

def bisection_attribute_assignment(G, px1, target_rho_kx, max_iter=50, tol=0.01, alpha=None, rkk=None):
    print(f'[{timestamp()}][alpha={alpha}, rkk={rkk}] Ajustando atributos para rho_kx={target_rho_kx}')
    start = time.time()
    n = G.number_of_nodes()
    degrees = np.array([deg for _, deg in G.degree()])
    degrees_norm = (degrees - degrees.min()) / (degrees.max() - degrees.min())
    low, high = 0.0, 1.0
    best_attrs = None
    best_corr_diff = 1e6
    for i in range(max_iter):
        theta = (low + high) / 2
        prob_active = (1 - theta) * px1 + theta * degrees_norm
        prob_active /= prob_active.sum()
        n_active = int(px1 * n)
        attrs = np.zeros(n, dtype=int)
        indices = np.random.choice(n, n_active, p=prob_active, replace=False)
        attrs[indices] = 1
        corr = pearsonr(degrees, attrs)[0]
        corr_diff = abs(corr - target_rho_kx)
        if corr_diff < best_corr_diff:
            best_corr_diff = corr_diff
            best_attrs = attrs.copy()
        if corr > target_rho_kx:
            high = theta
        else:
            low = theta
        if best_corr_diff < tol:
            break
    print(f'[{timestamp()}][alpha={alpha}, rkk={rkk}] Asignación terminada en {time.time()-start:.2f}s con corr_diff={best_corr_diff:.4f}')
    return best_attrs

def majority_illusion_fraction(G, attributes, alpha=None, rkk=None, rho_kx=None):
    print(f'[{timestamp()}][alpha={alpha}, rkk={rkk}, rho_kx={rho_kx}] Calculando fracción espejismo')
    start = time.time()
    n = G.number_of_nodes()
    adj = nx.adjacency_matrix(G, nodelist=range(n))
    attrs = np.array(attributes)
    counts = adj.dot(attrs)
    degrees = np.array([d for _, d in G.degree()])
    illusion_frac = np.sum(counts > degrees / 2) / n
    print(f'[{timestamp()}][alpha={alpha}, rkk={rkk}, rho_kx={rho_kx}] Fracción: {illusion_frac:.3f}, tiempo {time.time()-start:.2f}s')
    return illusion_frac

def experiment(args):
    alpha, rkk_target, rho_kx, G_base, px1, max_rewire_factor, tol, seed = args
    np.random.seed(seed)
    print(f'\n[{timestamp()}][alpha={alpha}] Inicio experimento rkk={rkk_target}, rho_kx={rho_kx}')
    G = copy.deepcopy(G_base)
    G = strong_rewire(G, rkk_target, max_iter_factor=max_rewire_factor, tol=tol, alpha=alpha)
    attrs = bisection_attribute_assignment(G, px1, rho_kx, alpha=alpha, rkk=rkk_target)
    frac = majority_illusion_fraction(G, attrs, alpha=alpha, rkk=rkk_target, rho_kx=rho_kx)
    print(f'[{timestamp()}][alpha={alpha}] Resultado rkk={rkk_target}, rho_kx={rho_kx} --> frac={frac:.3f}')
    return (alpha, rkk_target, rho_kx, frac)

# Parámetros
N = 5000
px1 = 0.05
params = {
    2.1: [-0.4, -0.1, 0.1, 0.3],
    2.4: [-0.45, -0.15, 0.05, 0.15],
    3.1: [-0.6, -0.25, 0.05, 0.25]
}
rho_kxs = np.linspace(0, 0.6, 10)
max_rewire_factor = 10
tol = 0.02

# Crear redes base para cada alpha
print(f'[{timestamp()}] Generando redes base...')
base_networks = {}
for alpha in params.keys():
    base_networks[alpha] = generate_scale_free_network(N, alpha, seed=42+int(alpha*10))

# Crear lista para experimentos paralelos
param_list = []
base_seed = 2025
for alpha, rkk_list in params.items():
    G_base = base_networks[alpha]
    for rkk in rkk_list:
        for rho_kx in rho_kxs:
            param_list.append((alpha, rkk, rho_kx, G_base, px1, max_rewire_factor, tol, base_seed))
            base_seed += 1

print(f'[{timestamp()}] Iniciando experimentos en paralelo...')
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(experiment, param_list)
print(f'[{timestamp()}] Experimentos completados.')

results_dict = {alpha: {} for alpha in params.keys()}
for alpha, rkk, rho_kx, frac in results:
    results_dict[alpha].setdefault(rkk, []).append((rho_kx, frac))

plt.figure(figsize=(18, 5))
colors = ['blue', 'orange', 'green', 'red']
markers = ['o', 's', '^', 'D']

for idx, alpha in enumerate(sorted(params.keys())):
    plt.subplot(1, 3, idx+1)
    for cidx, rkk in enumerate(params[alpha]):
        data = sorted(results_dict[alpha][rkk], key=lambda x: x[0])
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        linestyle = '-' if alpha == 2.1 else '--' if alpha == 2.4 else ':'
        plt.plot(x, y, label=f'$r_{{kk}}={rkk}$', color=colors[cidx], marker=markers[cidx], linestyle=linestyle)
    plt.title(f'Alpha={alpha}')
    plt.xlabel('Degree-attribute correlation $\\rho_{kx}$')
    plt.ylabel('Fraction nodes majority active neighbors')
    plt.legend()
    plt.tight_layout()

plt.suptitle('Majority Illusion en Redes Scale-Free por Alpha')
plt.show()
