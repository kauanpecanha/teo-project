#!/usr/bin/env python3

import os
import glob
import csv
import math
import random
import statistics
import datetime
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

DATASET_DIR = "./dataset"
SOLUTIONS_DIR = "./solutions"
RESULTS_ROOT = "./results"

NUM_RUNS = 30
POP_SIZE = 50
GENERATIONS = 100
WORKERS = max(1, os.cpu_count() - 2)
SEED_BASE = 1000

# ============================================================
# CLASSES DO MODELO
# ============================================================

class Node:
    def __init__(self, id, x, y, demanda=0, is_depot=False):
        self.id = id
        self.x = x
        self.y = y
        self.demanda = demanda
        self.is_depot = is_depot

class MDVRP_Problem:
    def __init__(self):
        self.clientes = []
        self.depositos = []
        self.capacidade_veiculo = 0
        self.dist_matrix = {}

    def carregar_cordeau(self, caminho_arquivo):
        with open(caminho_arquivo, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        header = list(map(int, lines[0].split()))
        num_clientes = header[2]
        num_depositos = header[3]
        specs = list(map(int, lines[1].split()))
        self.capacidade_veiculo = specs[1]

        start_customers = 1 + num_depositos
        self.clientes = []
        for i in range(num_clientes):
            dados = list(map(int, lines[start_customers + i].split()))
            self.clientes.append(Node(dados[0], dados[1], dados[2], demanda=dados[4]))

        start_depots = start_customers + num_clientes
        self.depositos = []
        for i in range(num_depositos):
            if start_depots + i >= len(lines): break
            dados = list(map(int, lines[start_depots + i].split()))
            self.depositos.append(Node(dados[0], dados[1], dados[2], is_depot=True))

        todos_nos = self.clientes + self.depositos
        self.dist_matrix = {}
        for n1 in todos_nos:
            for n2 in todos_nos:
                d = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
                self.dist_matrix[(n1.id, n2.id)] = d

    def get_dist(self, id1, id2):
        return self.dist_matrix.get((id1, id2), float('inf'))

class SplitDecoderMDVRP:
    def __init__(self, problem):
        self.p = problem

    def split(self, giant_tour):
        n = len(giant_tour)
        V = [float('inf')] * (n + 1)
        P = [0] * (n + 1)
        D = [-1] * (n + 1)
        V[0] = 0

        for i in range(1, n + 1):
            carga = 0
            custo_interno = 0
            j = i
            while j <= n:
                curr = self.p.clientes[self.get_client_idx(giant_tour[j-1])]
                if carga + curr.demanda > self.p.capacidade_veiculo:
                    break
                carga += curr.demanda
                
                if j > i:
                    prev = self.p.clientes[self.get_client_idx(giant_tour[j-2])]
                    custo_interno += self.p.get_dist(prev.id, curr.id)

                first = self.p.clientes[self.get_client_idx(giant_tour[i-1])]
                
                melhor_custo = float('inf')
                melhor_dep = -1

                for idx_dep, dep in enumerate(self.p.depositos):
                    d_in = self.p.get_dist(dep.id, first.id)
                    d_out = self.p.get_dist(curr.id, dep.id)
                    total = d_in + custo_interno + d_out
                    if total < melhor_custo:
                        melhor_custo = total
                        melhor_dep = idx_dep

                if V[i-1] + melhor_custo < V[j]:
                    V[j] = V[i-1] + melhor_custo
                    P[j] = i - 1
                    D[j] = melhor_dep
                j += 1

        rotas = []
        curr = n
        while curr > 0:
            prev = P[curr]
            dep_idx = D[curr]
            segmento = giant_tour[prev:curr]
            rotas.append({
                'deposito': self.p.depositos[dep_idx],
                'clientes': [self.p.clientes[self.get_client_idx(cid)] for cid in segmento],
                'custo': V[curr] - V[prev]
            })
            curr = prev
        return rotas, V[n]

    def get_client_idx(self, cid):
        return cid - 1

class HGS_MDVRP:
    def __init__(self, problema, pop_size=50, geracoes=100):
        self.p = problema
        self.pop_size = pop_size
        self.geracoes = geracoes
        self.decoder = SplitDecoderMDVRP(problema)
        self.populacao = []

    def inicializar(self):
        ids = [c.id for c in self.p.clientes]
        for _ in range(self.pop_size):
            random.shuffle(ids)
            fit = self.avaliar(ids)
            self.populacao.append((list(ids), fit))

    def avaliar(self, tour):
        _, custo = self.decoder.split(tour)
        return custo

    def crossover_ox(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        filho = [-1]*size
        filho[a:b] = p1[a:b]
        holes = [x for x in p2 if x not in filho]
        idx_h = 0
        for i in range(size):
            if filho[i] == -1:
                filho[i] = holes[idx_h]
                idx_h += 1
        return filho

    def local_search_swap(self, tour):
        best_tour = list(tour)
        best_fit = self.avaliar(tour)
        for _ in range(30): 
            i, j = random.sample(range(len(tour)), 2)
            neighbor = list(best_tour)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            fit = self.avaliar(neighbor)
            if fit < best_fit:
                best_fit = fit
                best_tour = neighbor
        return best_tour, best_fit

    def run(self, run_id=None):
        self.inicializar()
        self.populacao.sort(key=lambda x: x[1])
        best_sol = self.populacao[0]
        
        for _ in range(self.geracoes):
            new_pop = self.populacao[:int(self.pop_size*0.2)]
            while len(new_pop) < self.pop_size:
                p1 = min(random.sample(self.populacao, 5), key=lambda x: x[1])[0]
                p2 = min(random.sample(self.populacao, 5), key=lambda x: x[1])[0]
                child = self.crossover_ox(p1, p2)
                if random.random() < 0.6: 
                    child, fit = self.local_search_swap(child)
                else:
                    fit = self.avaliar(child)
                new_pop.append((child, fit))
            self.populacao = sorted(new_pop, key=lambda x: x[1])
            if self.populacao[0][1] < best_sol[1]:
                best_sol = self.populacao[0]
        return best_sol

# ============================================================
# FUNÇÕES WORKER E PLOTAGEM
# ============================================================

def run_single_ga(args):
    """Worker isolado para processamento paralelo"""
    seed, dataset_path = args
    random.seed(seed)
    p = MDVRP_Problem()
    try:
        p.carregar_cordeau(dataset_path)
    except Exception as e:
        return None
    ga = HGS_MDVRP(p, pop_size=POP_SIZE, geracoes=GENERATIONS)
    seq, cost = ga.run(run_id=seed)
    return {"seed": seed, "cost": cost, "sequence": seq}

def load_reference(instance_name):
    """Busca o arquivo .res correspondente na pasta solutions"""
    res_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.res")
    if not os.path.exists(res_path):
        return None
    try:
        with open(res_path, 'r', errors='ignore') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            return float(lines[0])
    except:
        return None

def save_instance_plots(problem, results, output_dir, ref_cost):
    # 1. Scatter Plot
    costs = [r["cost"] for r in results]
    best_my_cost = min(costs)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(range(1, len(costs)+1), costs, label="Execuções GA", alpha=0.7)
    if ref_cost:
        plt.axhline(ref_cost, color='red', linestyle='--', label=f"Referência ({ref_cost:.2f})")
    plt.scatter([costs.index(best_my_cost)+1], [best_my_cost], color='green', s=100, label="Melhor GA")
    
    plt.xlabel("Execução")
    plt.ylabel("Custo")
    plt.title("Dispersão das Execuções")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "scatter_local.png"))
    plt.close()

    # 2. Melhor Rota Visual
    best_run = min(results, key=lambda x: x["cost"])
    decoder = SplitDecoderMDVRP(problem)
    rotas, _ = decoder.split(best_run["sequence"])
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', len(problem.depositos))
    
    for c in problem.clientes:
        plt.scatter(c.x, c.y, c='grey', s=15, alpha=0.3)
    
    for r in rotas:
        dep = r['deposito']
        d_idx = [d.id for d in problem.depositos].index(dep.id)
        path_x = [dep.x] + [c.x for c in r['clientes']] + [dep.x]
        path_y = [dep.y] + [c.y for c in r['clientes']] + [dep.y]
        plt.plot(path_x, path_y, color=colors(d_idx), linewidth=1, alpha=0.8)

    for i, d in enumerate(problem.depositos):
        plt.scatter(d.x, d.y, c=[colors(i)], marker='s', s=100, edgecolors='black', label=f"Dep {d.id}")

    plt.title(f"Melhor Solução (Custo: {best_run['cost']:.2f})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "melhor_rota.png"))
    plt.close()

def plot_global_comparison(global_stats, run_dir):
    """Gera gráfico de barras comparativo: Meu Melhor vs Referência"""
    instances = sorted(global_stats.keys())
    my_best = []
    refs = []
    
    for inst in instances:
        my_best.append(global_stats[inst]['best_ga'])
        # Se não houver referência, usamos 0 ou omitimos visualmente
        refs.append(global_stats[inst]['ref_cost'] if global_stats[inst]['ref_cost'] else 0)

    x = np.arange(len(instances))
    width = 0.35

    plt.figure(figsize=(12, 6))
    
    # Se houver referências válidas, plotamos. Caso contrário, só o GA.
    if any(r > 0 for r in refs):
        plt.bar(x - width/2, my_best, width, label='Meu GA (Melhor)', color='steelblue')
        plt.bar(x + width/2, refs, width, label='Referência (BKS)', color='orange')
    else:
        plt.bar(x, my_best, width, label='Meu GA (Melhor)', color='steelblue')

    plt.xlabel('Instância')
    plt.ylabel('Custo Total')
    plt.title('Comparativo Global: Algoritmo Genético vs Melhor Solução Conhecida')
    plt.xticks(x, instances, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "COMPARATIVO_GLOBAL.png"))
    plt.close()
    print(f"[INFO] Gráfico comparativo global salvo em {run_dir}")

# ============================================================
# LÓGICA PRINCIPAL (ITERAÇÃO)
# ============================================================

def process_instance(filepath, run_timestamp_dir):
    filename = os.path.basename(filepath)
    instance_name = os.path.splitext(filename)[0]
    
    # Criar pasta para esta instância
    instance_dir = os.path.join(run_timestamp_dir, instance_name)
    os.makedirs(instance_dir, exist_ok=True)
    
    print(f"\n>>> Processando instância: {instance_name}")
    
    # Carregar Referência
    ref_cost = load_reference(instance_name)
    
    # Executar GA Paralelo
    seeds = [SEED_BASE + i for i in range(NUM_RUNS)]
    results = []
    args_list = [(s, filepath) for s in seeds]
    
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_single_ga, arg): arg for arg in args_list}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    
    if not results:
        print(f"[ERRO] Falha ao processar {instance_name}")
        return None

    # Estatísticas
    costs = [r["cost"] for r in results]
    best_ga = min(costs)
    avg_ga = statistics.mean(costs)
    
    # Salvar CSV Local
    with open(os.path.join(instance_dir, "resultados.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "custo", "sequencia"])
        for r in results:
            writer.writerow([r["seed"], r["cost"], r["sequence"]])

    # Salvar Sumário Local
    with open(os.path.join(instance_dir, "sumario.txt"), "w") as f:
        gap = ((best_ga - ref_cost)/ref_cost * 100) if ref_cost else 0
        f.write(f"Instância: {instance_name}\n")
        f.write(f"Ref (BKS): {ref_cost if ref_cost else 'N/A'}\n")
        f.write(f"GA Melhor: {best_ga:.2f}\n")
        f.write(f"GA Média:  {avg_ga:.2f}\n")
        f.write(f"GAP:       {gap:.2f}%\n")

    # Gerar Plots Locais
    p = MDVRP_Problem()
    p.carregar_cordeau(filepath)
    save_instance_plots(p, results, instance_dir, ref_cost)

    return {
        "name": instance_name,
        "best_ga": best_ga,
        "avg_ga": avg_ga,
        "ref_cost": ref_cost
    }

def main():
    # Setup de diretórios
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Encontrar arquivos
    dataset_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mdvrp")))
    if not dataset_files:
        print(f"[ERRO] Nenhum arquivo .mdvrp encontrado em {DATASET_DIR}")
        return

    print(f"--- INICIANDO BATERIA DE TESTES ---")
    print(f"Diretório de saída: {run_dir}")
    print(f"Arquivos encontrados: {len(dataset_files)}\n")

    global_stats = {}

    for fpath in dataset_files:
        stats = process_instance(fpath, run_dir)
        if stats:
            global_stats[stats["name"]] = stats

    # Gerar Relatório Global CSV
    with open(os.path.join(run_dir, "resumo_geral.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["instancia", "ref_bks", "ga_melhor", "ga_media", "gap_percent"])
        for name in sorted(global_stats.keys()):
            s = global_stats[name]
            ref = s['ref_cost']
            gap = ((s['best_ga'] - ref)/ref * 100) if ref else 0
            writer.writerow([name, ref, f"{s['best_ga']:.2f}", f"{s['avg_ga']:.2f}", f"{gap:.2f}"])

    # Gerar Gráfico Global
    plot_global_comparison(global_stats, run_dir)
    print("\n--- BATERIA FINALIZADA COM SUCESSO ---")

if __name__ == "__main__":
    main()