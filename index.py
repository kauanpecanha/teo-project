import math
import random
import matplotlib.pyplot as plt
import copy
import os
import datetime
import statistics

# --- 1. Modelagem MDVRP (Multi-Depot) ---

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
        self.max_veiculos_por_deposito = 0
        self.dist_matrix = {}

    def carregar_cordeau(self, caminho_arquivo):
        with open(caminho_arquivo, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # Parsing Header
        header = list(map(int, lines[0].split()))
        self.max_veiculos_por_deposito = header[1]
        num_clientes = header[2]
        num_depositos = header[3]

        # Parsing Specs
        specs_line_index = 1
        specs = list(map(int, lines[specs_line_index].split()))
        self.capacidade_veiculo = specs[1]

        start_customers = 1 + num_depositos

        # Parsing Clientes
        self.clientes = []
        for i in range(num_clientes):
            line_idx = start_customers + i
            dados = list(map(int, lines[line_idx].split()))
            c = Node(id=dados[0], x=dados[1], y=dados[2], demanda=dados[4], is_depot=False)
            self.clientes.append(c)

        # Parsing Depositos
        start_depots = start_customers + num_clientes
        self.depositos = []
        for i in range(num_depositos):
            line_idx = start_depots + i
            if line_idx >= len(lines): break
            dados = list(map(int, lines[line_idx].split()))
            d = Node(id=dados[0], x=dados[1], y=dados[2], demanda=0, is_depot=True)
            self.depositos.append(d)

        # Pre-calc Distances
        todos_nos = self.clientes + self.depositos
        self.dist_matrix = {}
        for n1 in todos_nos:
            for n2 in todos_nos:
                d = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
                self.dist_matrix[(n1.id, n2.id)] = d

    def get_dist(self, id1, id2):
        return self.dist_matrix.get((id1, id2), float('inf'))

# --- 2. Split Algorithm ---

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
                cliente_atual = self.p.clientes[self.get_client_idx(giant_tour[j-1])]
                
                if carga + cliente_atual.demanda > self.p.capacidade_veiculo:
                    break
                carga += cliente_atual.demanda
                
                if j > i:
                    cliente_anterior = self.p.clientes[self.get_client_idx(giant_tour[j-2])]
                    custo_interno += self.p.get_dist(cliente_anterior.id, cliente_atual.id)

                primeiro_cliente = self.p.clientes[self.get_client_idx(giant_tour[i-1])]
                ultimo_cliente = cliente_atual
                
                melhor_custo_rota = float('inf')
                melhor_deposito_idx = -1

                for idx_dep, deposito in enumerate(self.p.depositos):
                    d_entrada = self.p.get_dist(deposito.id, primeiro_cliente.id)
                    d_saida = self.p.get_dist(ultimo_cliente.id, deposito.id)
                    custo_total = d_entrada + custo_interno + d_saida
                    if custo_total < melhor_custo_rota:
                        melhor_custo_rota = custo_total
                        melhor_deposito_idx = idx_dep

                if V[i-1] + melhor_custo_rota < V[j]:
                    V[j] = V[i-1] + melhor_custo_rota
                    P[j] = i - 1
                    D[j] = melhor_deposito_idx
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

# --- 3. HGS ---

class HGS_MDVRP:
    def __init__(self, problema, pop_size=50, geracoes=100):
        self.p = problema
        self.pop_size = pop_size
        self.geracoes = geracoes
        self.decoder = SplitDecoderMDVRP(problema)
        self.populacao = []

    def inicializar(self):
        ids_base = [c.id for c in self.p.clientes]
        for _ in range(self.pop_size):
            random.shuffle(ids_base)
            fit = self.avaliar(ids_base)
            self.populacao.append((list(ids_base), fit))

    def avaliar(self, cromossomo):
        _, custo = self.decoder.split(cromossomo)
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
        # Reduced iterations for speed in repeated runs
        for _ in range(30): 
            i, j = random.sample(range(len(tour)), 2)
            neighbor = list(best_tour)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            fit = self.avaliar(neighbor)
            if fit < best_fit:
                best_fit = fit
                best_tour = neighbor
        return best_tour, best_fit

    def run(self, run_id):
        self.inicializar()
        self.populacao.sort(key=lambda x: x[1])
        best_sol = self.populacao[0]
        
        for g in range(self.geracoes):
            new_pop = self.populacao[:int(self.pop_size*0.2)]
            while len(new_pop) < self.pop_size:
                parent1 = min(random.sample(self.populacao, 5), key=lambda x: x[1])[0]
                parent2 = min(random.sample(self.populacao, 5), key=lambda x: x[1])[0]
                child_seq = self.crossover_ox(parent1, parent2)
                if random.random() < 0.6: 
                    child_seq, fit = self.local_search_swap(child_seq)
                else:
                    fit = self.avaliar(child_seq)
                new_pop.append((child_seq, fit))
            self.populacao = sorted(new_pop, key=lambda x: x[1])
            if self.populacao[0][1] < best_sol[1]:
                best_sol = self.populacao[0]
        
        return best_sol

# --- 4. Saving & Plotting ---

def save_plot(problem, rotas, custo_total, folder, filename="best_run.png"):
    plt.figure(figsize=(12, 10))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    
    # Clientes
    for c in problem.clientes:
        plt.scatter(c.x, c.y, c='grey', s=20, alpha=0.3)

    # Rotas
    for i, rota in enumerate(rotas):
        dep = rota['deposito']
        # Cor baseada no deposito
        idx_dep = -1
        for k, d in enumerate(problem.depositos):
            if d.id == dep.id:
                idx_dep = k
                break
        cor_rota = colors[idx_dep % len(colors)]
        
        path_x = [dep.x] + [c.x for c in rota['clientes']] + [dep.x]
        path_y = [dep.y] + [c.y for c in rota['clientes']] + [dep.y]
        plt.plot(path_x, path_y, c=cor_rota, alpha=0.8, linewidth=1.5)

    # Depositos
    for i, d in enumerate(problem.depositos):
        c = colors[i % len(colors)]
        plt.scatter(d.x, d.y, c=c, marker='s', s=150, edgecolors='black', zorder=10, label=f"Dep {d.id}")

    plt.title(f"Best Solution MDVRP - Cost: {custo_total:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(folder, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Graph saved to: {save_path}")

def run_experiment():
    DATASET_FILE = "./dataset/p01.mdvrp"
    NUM_RUNS = 30         
    POP_SIZE = 50
    GENERATIONS = 100     
    
    # Create Output Folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"results_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"--- Starting Experiment: {NUM_RUNS} cycles ---")
    print(f"Results will be saved in: {output_folder}/")
    
    problem = MDVRP_Problem()
    try:
        problem.carregar_cordeau(DATASET_FILE)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    results_log = []
    best_global_sol = None
    best_global_fit = float('inf')

    # --- STATISTICAL LOOP ---
    for i in range(1, NUM_RUNS + 1):
        ga = HGS_MDVRP(problem, pop_size=POP_SIZE, geracoes=GENERATIONS)
        
        # Run GA
        (sol_seq, sol_fit) = ga.run(run_id=i)
        
        # Log
        print(f"Run {i}/{NUM_RUNS} | Best Cost: {sol_fit:.2f}")
        results_log.append(sol_fit)
        
        # Save Global Best
        if sol_fit < best_global_fit:
            best_global_fit = sol_fit
            best_global_sol = sol_seq

    # --- STATISTICS ---
    avg_cost = statistics.mean(results_log)
    min_cost = min(results_log)
    max_cost = max(results_log)
    std_dev = statistics.stdev(results_log) if len(results_log) > 1 else 0.0

    summary = (
        f"--- EXPERIMENT SUMMARY ---\n"
        f"Dataset: {DATASET_FILE}\n"
        f"Runs: {NUM_RUNS}\n"
        f"Generations per run: {GENERATIONS}\n"
        f"--------------------------\n"
        f"MIN Cost (Best Global): {min_cost:.2f}\n"
        f"MAX Cost (Worst Run):   {max_cost:.2f}\n"
        f"AVG Cost:               {avg_cost:.2f}\n"
        f"STD DEV:                {std_dev:.2f}\n"
        f"--------------------------\n"
        f"All Runs Data: {results_log}\n"
    )

    print("\n" + summary)

    # --- SAVING RESULTS ---
    # 1. Save Text Summary
    with open(os.path.join(output_folder, "summary.txt"), "w") as f:
        f.write(summary)
        
    # 2. Decode and Save Plot of Best Solution
    decoder = SplitDecoderMDVRP(problem)
    rotas_finais, _ = decoder.split(best_global_sol)
    
    save_plot(problem, rotas_finais, best_global_fit, output_folder)
    
    # 3. Save Route Details
    with open(os.path.join(output_folder, "best_routes_details.txt"), "w") as f:
        f.write(f"BEST SOLUTION DETAILS (Cost: {best_global_fit:.2f})\n\n")
        for idx, r in enumerate(rotas_finais):
            f.write(f"Route {idx+1} (Depot {r['deposito'].id}): {[c.id for c in r['clientes']]}\n")
            f.write(f"   -> Route Cost impact: {r['custo']:.2f}\n")

if __name__ == "__main__":
    run_experiment()