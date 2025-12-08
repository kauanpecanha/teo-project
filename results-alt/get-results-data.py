import pandas as pd
import glob
import os
import re

def sort_key(filename):
    # Ordena corretamente as instâncias de dados
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

# Procura os CSVs na pasta
result_files = sorted(glob.glob("resultados-p*.csv"), key=sort_key)
ref_dir = "."  # Diretório onde estão os arquivos .res

for f_res in result_files:
    # Extrai nome da instância do nome do arquivo
    match = re.search(r'(p\d+)', f_res)
    if not match: continue
    instancia = match.group(1)
    
    # Busca arquivo de solução
    f_ref = os.path.join(ref_dir, f"{instancia}.res")
    
    bks = None
    if os.path.exists(f_ref):
        try:
            with open(f_ref, 'r') as f:
                val = f.readline().strip()
                if val:
                    bks = float(val)
        except:
            bks = None
    
    # Ler resultados do GA
    try:
        df = pd.read_csv(f_res)
        
        if 'custo' in df.columns:
            costs = df['custo']
        else:
            costs = df.iloc[:, 1]
            
        best_ga = costs.min()
        avg_ga = costs.mean()
        std_ga = costs.std()
        
        if bks and bks > 0:
            gap = ((best_ga - bks) / bks) * 100
            gap_str = f"{gap:.2f}"
        else:
            gap_str = "-"
            if bks is None: bks = 0
            
        print(f"{instancia.upper()} & {bks:.2f} & {best_ga:.2f} & {avg_ga:.2f} & {gap_str}\\% & {std_ga:.2f} ")
        
    except Exception as e:
        print(f"{instancia.upper()} & - & - & - & - & - % Erro: {e}")
