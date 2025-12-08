import math
import random

# CLASSES DO MODELO
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