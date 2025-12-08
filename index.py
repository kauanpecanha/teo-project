#!/usr/bin/env python3

import os
import glob
import csv
import random
import statistics
import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from classes import MDVRP_Problem, SplitDecoderMDVRP, HGS_MDVRP

# CONFIGURAÇÕES GERAIS
DATASET_DIR = "./dataset"
SOLUTIONS_DIR = "./solutions"
RESULTS_ROOT = "./results"

NUM_RUNS = 30
POP_SIZE = 50
GENERATIONS = 100
WORKERS = max(1, os.cpu_count() - 2)
SEED_BASE = 1000

# FUNÇÕES WORKER E PLOTAGEM
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
    # Scatter Plot
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

    # Melhor Rota Visual
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

# LÓGICA PRINCIPAL (ITERAÇÃO)
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

    print(f"Diretório de saída: {run_dir}")
    print(f"Arquivos encontrados: {len(dataset_files)}\n")

    global_stats = {}

    for fpath in dataset_files:
        stats = process_instance(fpath, run_dir)
        if stats:
            global_stats[stats["name"]] = stats

if __name__ == "__main__":
    main()