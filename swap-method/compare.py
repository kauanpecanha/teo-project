#!/usr/bin/env python3

import os
import re
import csv
import math
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import matplotlib.pyplot as plt

from index import MDVRP_Problem, HGS_MDVRP   # Import do seu código


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATASET_FILE = "../dataset/p01.mdvrp"
SOLUTIONS_DIR = "../solutions"

NUM_RUNS = 30
POP_SIZE = 50
GENERATIONS = 100
WORKERS = max(1, os.cpu_count() - 3)

OUTPUT_DIR = f"parallel_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED_BASE = 1000


# ============================================================
# FUNÇÃO: LER SOLUÇÕES DE REFERÊNCIA
# ============================================================

def load_reference_solutions(problem, folder=SOLUTIONS_DIR, dataset_file=DATASET_FILE):
    solutions = []

    if not os.path.isdir(folder):
        print(f"[AVISO] Pasta de soluções não encontrada: {folder}")
        return []

    # ====================================================
    # Descobrir nome base do dataset, por exemplo:
    # "./dataset/p01.mdvrp"  ->  "p01"
    # ====================================================
    base = os.path.basename(dataset_file).split(".")[0]  # p01

    # Arquivo esperado de solução:
    expected_file = f"{base}.res"

    fpath = os.path.join(folder, expected_file)
    if not os.path.isfile(fpath):
        print(f"[ERRO] Arquivo de solução esperado não encontrado: {expected_file}")
        return []

    # Agora lê somente p01.res
    try:
        with open(fpath, 'r', errors='ignore') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except:
        print(f"[ERRO] Não foi possível ler {expected_file}")
        return []

    # Primeira linha contém o custo total
    try:
        total_cost = float(lines[0])
    except:
        print(f"[ERRO] Primeira linha de {expected_file} não contém custo total.")
        return []

    # Extrair rotas apenas para referência visual
    routes = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 5:
            continue

        client_ids = [int(x) for x in parts[4:] if x != "0"]
        routes.append(client_ids)

    return [{
        "file": expected_file,
        "cost": total_cost,
        "routes": routes
    }]





# ============================================================
# EXECUTA UMA ÚNICA RUN DO GA
# ============================================================

def run_single(seed):
    random.seed(seed)

    problem = MDVRP_Problem()
    problem.carregar_cordeau(DATASET_FILE)

    ga = HGS_MDVRP(problem, pop_size=POP_SIZE, geracoes=GENERATIONS)
    sol_seq, sol_cost = ga.run(run_id=seed)

    return {"seed": seed, "cost": sol_cost, "sequence": sol_seq}


# ============================================================
# EXECUÇÕES PARALELAS
# ============================================================

def run_parallel_experiments():
    seeds = [SEED_BASE + i for i in range(NUM_RUNS)]
    results = []

    print(f"[INFO] Rodando {NUM_RUNS} execuções paralelas...")

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_single, s): s for s in seeds}

        for future in as_completed(futures):
            seed = futures[future]
            try:
                res = future.result()
                print(f"[OK] seed={seed} → custo={res['cost']:.2f}")
                results.append(res)
            except Exception as e:
                print(f"[ERRO] Execução seed {seed}: {e}")

    results.sort(key=lambda x: x["cost"])
    return results


# ============================================================
# GERA CSV
# ============================================================

def save_csv(results):
    csv_path = os.path.join(OUTPUT_DIR, "resultados.csv")

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["semente", "custo", "sequência_de_clientes"])

        for r in results:
            writer.writerow([r["seed"], r["cost"], " ".join(map(str, r["sequence"]))])

    print(f"[INFO] CSV salvo em {csv_path}")


# ============================================================
# SCATTER COMPARATIVO
# ============================================================

def plot_scatter(costs, best_ref_cost):
    path = os.path.join(OUTPUT_DIR, "scatter_comparacao.png")

    plt.figure(figsize=(8, 5))

    x = list(range(1, len(costs)+1))

    # Pontos do GA
    plt.scatter(x, costs, label="Custo obtido pelo algoritmo genético", s=50)

    # Linha da melhor solução
    if best_ref_cost is not None:
        plt.axhline(best_ref_cost, color='red', linestyle='--',
                    label=f"Melhor solução conhecida ({best_ref_cost:.2f})")

    # Melhor solução obtida
    best_cost = min(costs)
    best_index = costs.index(best_cost) + 1
    plt.scatter([best_index], [best_cost], color='green', s=120,
                label=f"Melhor custo obtido ({best_cost:.2f})")

    plt.xlabel("Número da execução")
    plt.ylabel("Custo da solução")
    plt.title("Comparação entre as execuções do GA e a melhor solução conhecida")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(path)
    plt.close()

    print(f"[INFO] Scatter salvo em {path}")



# ============================================================
# SUMÁRIO CLARO EM PORTUGUÊS
# ============================================================

def save_summary(results, best_ref):
    summary_path = os.path.join(OUTPUT_DIR, "sumario.txt")

    custos = [r["cost"] for r in results]
    melhor = min(custos)
    pior = max(custos)
    media = statistics.mean(custos)
    desvio = statistics.stdev(custos) if len(custos) > 1 else 0

    with open(summary_path, "w") as f:

        f.write("RESUMO DETALHADO DO EXPERIMENTO\n")
        f.write("=====================================\n\n")

        f.write(f"Total de execuções realizadas: {NUM_RUNS}\n")
        f.write(f"Gerações por execução: {GENERATIONS}\n")
        f.write(f"Tamanho da população: {POP_SIZE}\n\n")

        f.write("Resultados obtidos:\n")
        f.write(f"- Melhor custo encontrado pelo algoritmo: {melhor:.2f}\n")
        f.write(f"- Pior custo encontrado: {pior:.2f}\n")
        f.write(f"- Custo médio das execuções: {media:.2f}\n")
        f.write(f"- Desvio padrão dos custos: {desvio:.2f}\n\n")

        if best_ref:
            gap = (melhor - best_ref["cost"]) / best_ref["cost"] * 100
            f.write("Comparação com a melhor solução conhecida:\n")
            f.write(f"- Arquivo da solução de referência: {best_ref['file']}\n")
            f.write(f"- Melhor custo conhecido: {best_ref['cost']:.2f}\n")
            f.write(f"- Melhor custo obtido pelo algoritmo: {melhor:.2f}\n")
            f.write(f"- Diferença percentual (GAP): {gap:.2f}%\n\n")

            f.write("Interpretação do GAP:\n")
            f.write("- Um valor positivo indica que a solução obtida pelo GA é pior do que a referência.\n")
            f.write("- Quanto menor o GAP, mais próxima a solução do GA está da melhor solução conhecida.\n")
        else:
            f.write("Nenhuma solução de referência foi encontrada para comparação.\n")

    print(f"[INFO] Sumário salvo em {summary_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    # Carrega problema (para ler soluções de referência)
    problem = MDVRP_Problem()
    problem.carregar_cordeau(DATASET_FILE)

    # Carregar soluções de referência
    referencia = load_reference_solutions(problem)
    best_ref = min(referencia, key=lambda x: x["cost"]) if referencia else None

    if best_ref:
        print(f"[INFO] Melhor solução de referência: {best_ref['file']} → {best_ref['cost']:.2f}")

    # Executar GA paralelo
    results = run_parallel_experiments()

    # Salvar CSV
    save_csv(results)

    # Scatter comparação
    plot_scatter([r["cost"] for r in results], best_ref["cost"] if best_ref else None)

    # Sumário detalhado
    save_summary(results, best_ref)


if __name__ == "__main__":
    main()
