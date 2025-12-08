import glob
import os
import re

def sort_key(filename):
    # Ordena corretamente p01, p02... p10...
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

dataset_path = "./dataset/*.mdvrp"
files = sorted(glob.glob(dataset_path), key=sort_key)

for f_path in files:
    filename = os.path.basename(f_path)
    instancia = os.path.splitext(filename)[0]

    try:
        with open(f_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            header = list(map(int, lines[0].split()))
            num_veiculos = header[1]
            num_clientes = header[2]
            num_depositos = header[3]

            print(f"{instancia.upper()} & {num_depositos} & {num_clientes} & {num_veiculos}")
            
    except Exception as e:
        print(f"{instancia} & Erro & - & - % Erro ao ler arquivo: {e}")
