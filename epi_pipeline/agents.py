import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

STATE_SUSCEPTIBLE = 0
STATE_EXPOSED = 1
STATE_INFECTIOUS = 2
STATE_RECOVERED = 3

@dataclass
class Variant:
    name: str
    p_epitope: float
    VE: float
    R0_eff: float
    beta: float

@dataclass
class Agent:
    state: int = STATE_SUSCEPTIBLE
    vaccinated: bool = False
    region: int = 0  # rótulo da região/cluster espacial
    variant: Optional[str] = None  # nome da variante se infetado
    days_in_state: int = 0

def run_abm_simulation(
    variants: Dict[str, 'Variant'], # Usa 'Variant' ou o tipo da tua classe
    days: int = 160,
    N: int = 5000,
    vaccine_coverage: float = 0.4,
    initial_infected: Optional[Dict[str, int]] = None,
    contacts_per_day: float = 10.0,
    latent_period: float = 2.0,       # Mudei para float para permitir médias não inteiras
    infectious_period: float = 5.0,   # Mudei para float
    distancing_factor: float = 1.0,
    extra_vaccination: float = 0.0,
    n_regions: int = 3,
    intra_region_prob: float = 0.8,
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Simula um modelo ABM S–E–I–R com transições estocásticas (Markovianas).
    Isto gera curvas mais suaves e realistas do que períodos fixos.
    """
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Configuração de vacinação
    base_vac = float(np.clip(vaccine_coverage, 0.0, 1.0))
    extra_vac = float(np.clip(extra_vaccination, 0.0, 1.0))
    total_vac = float(np.clip(base_vac + extra_vac, 0.0, 1.0))

    # Inicialização padrão se vazio
    if initial_infected is None:
        first_var = next(iter(variants.keys()))
        initial_infected = {first_var: 10}

    # 1. Criar População
    agents = []
    if n_regions <= 0: n_regions = 1
    
    for _ in range(N):
        vaccinated = random.random() < total_vac
        region = random.randrange(n_regions)
        # Cria agente suscetível
        agents.append(Agent(state=STATE_SUSCEPTIBLE, vaccinated=vaccinated, region=region))

    # Cache de índices por região para performance
    region_indices = {r: [] for r in range(n_regions)}
    for idx, ag in enumerate(agents):
        region_indices[ag.region].append(idx)

    # 2. Semear Infeções Iniciais
    all_indices = list(range(N))
    random.shuffle(all_indices)
    idx_cursor = 0
    
    for var_name, n_inf in initial_infected.items():
        count = 0
        while count < n_inf and idx_cursor < N:
            idx = all_indices[idx_cursor]
            idx_cursor += 1
            
            # Só infetar se não estiver vacinado (ou infetar vacinados? 
            # Normalmente força-se a infeção inicial independentemente da vacina para garantir o arranque)
            agents[idx].state = STATE_EXPOSED # Começam em Exposed para dar tempo de arranque
            agents[idx].variant = var_name
            agents[idx].days_in_state = 0
            count += 1

    # Probabilidades de transição diária (1 / média de dias)
    # Evitar divisão por zero usando max(0.1, valor)
    prob_become_infectious = 1.0 / max(0.1, float(latent_period))
    prob_recover = 1.0 / max(0.1, float(infectious_period))

    records = []

    # --- LOOP TEMPORAL ---
    for day in range(days):
        
        # A. Contagens para Estatística
        s = e = i = r = 0
        variant_I_counts = {name: 0 for name in variants.keys()}
        region_I_counts = {reg: 0 for reg in range(n_regions)}
        
        # Lista de índices dos infeciosos para otimizar o loop de contágio
        infectious_indices = []

        for idx, ag in enumerate(agents):
            if ag.state == STATE_SUSCEPTIBLE:
                s += 1
            elif ag.state == STATE_EXPOSED:
                e += 1
            elif ag.state == STATE_INFECTIOUS:
                i += 1
                infectious_indices.append(idx) # Guardar índice para usar no contágio
                if ag.variant in variant_I_counts:
                    variant_I_counts[ag.variant] += 1
                region_I_counts[ag.region] += 1
            elif ag.state == STATE_RECOVERED:
                r += 1

        # Registo dos dados
        rec = {"day": day, "S": s, "E": e, "I": i, "R": r, "N": N}
        for name, val in variant_I_counts.items():
            rec[f"I_{name}"] = val
        for reg, val in region_I_counts.items():
            rec[f"I_region_{reg}"] = val
        records.append(rec)

        # Critério de paragem (se a epidemia acabou)
        if i == 0 and e == 0 and day > 0:
            break

        # B. Dinâmica de Contágio (Loop apenas sobre os infeciosos)
        # Isto é mais eficiente do que iterar todos os agentes à procura de infeciosos
        to_infect = [] 
        
        effective_contacts = contacts_per_day * distancing_factor
        
        for idx in infectious_indices:
            ag = agents[idx]
            var = variants[ag.variant]
            
            # Número de contactos deste agente hoje (Poisson adiciona variabilidade realista)
            n_contacts = np.random.poisson(effective_contacts)
            
            if n_contacts == 0: continue

            for _ in range(n_contacts):
                # Escolha do alvo (Intra vs Extra região)
                if random.random() < intra_region_prob and region_indices[ag.region]:
                    target_idx = random.choice(region_indices[ag.region])
                else:
                    target_idx = random.randrange(N)
                
                target = agents[target_idx]
                
                if target.state == STATE_SUSCEPTIBLE:
                    # Cálculo da probabilidade de infeção
                    p_inf = var.beta
                    if target.vaccinated:
                        p_inf *= (1.0 - var.VE)
                    
                    if p_inf > 0 and random.random() < p_inf:
                        # Guardamos o índice e a variante para aplicar depois
                        to_infect.append((target_idx, ag.variant))

        # Aplicar novas infeções (fora do loop para evitar conflitos)
        for t_idx, v_name in to_infect:
            if agents[t_idx].state == STATE_SUSCEPTIBLE: # Verificação dupla necessária
                agents[t_idx].state = STATE_EXPOSED
                agents[t_idx].variant = v_name
                agents[t_idx].days_in_state = 0

        # C. Atualização de Estados (Transição Estocástica)
        for ag in agents:
            ag.days_in_state += 1
            
            if ag.state == STATE_EXPOSED:
                # Sorteio para ver se passa a Infecioso hoje
                if random.random() < prob_become_infectious:
                    ag.state = STATE_INFECTIOUS
                    ag.days_in_state = 0
            
            elif ag.state == STATE_INFECTIOUS:
                # Sorteio para ver se Recupera hoje
                if random.random() < prob_recover:
                    ag.state = STATE_RECOVERED
                    ag.days_in_state = 0

    return pd.DataFrame(records)