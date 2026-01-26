import numpy as np

def pepitope_to_VE(p_epitope: float) -> float:
    """Calcula eficácia vacinal teórica a partir de pEpitope.
    Fórmula baseada na literatura de Deem: VE ≈ -2.47 * p + 0.47
    Cortamos o valor para o intervalo [0, 1]."""
    ve = -2.47 * p_epitope + 0.47
    return float(np.clip(ve, 0.0, 1.0))

def pepitope_to_R0(p_epitope: float,
                     base_R0: float = 1.5,
                     vaccine_coverage: float = 0.4) -> float:
    """Mapeia pEpitope para um R0 efetivo simples.
    - base_R0: R0 de referência para uma variante bem coberta pela vacina.
    - vaccine_coverage: fração vacinada na população.

    Ideia: variantes com pEpitope alto têm VE baixa, logo a proteção da vacina
    cai e o R0 efetivo aumenta em relação ao cenário de boa proteção."""
    ve = pepitope_to_VE(p_epitope)
    # Fração efetivamente protegida: vacinados × VE
    protected = vaccine_coverage * ve
    susceptible_fraction = 1.0 - protected
    # R0 efetivo proporcional à fração suscetível
    return base_R0 * susceptible_fraction

def R0_to_beta(R0: float,
                infectious_period: float = 5.0,
                contacts_per_day: float = 10.0) -> float:
    """Converte R0 aproximado em probabilidade de transmissão por contacto (β).
    R0 ≈ beta * contacts_per_day * infectious_period
    → beta ≈ R0 / (contacts_per_day * infectious_period)"""
    return R0 / (contacts_per_day * infectious_period)

def pepitope_to_beta(p_epitope: float,
                     base_R0: float = 1.5,
                     vaccine_coverage: float = 0.4,
                     infectious_period: float = 5.0,
                     contacts_per_day: float = 10.0) -> float:
    R0 = pepitope_to_R0(p_epitope, base_R0=base_R0, vaccine_coverage=vaccine_coverage)
    return R0_to_beta(R0, infectious_period=infectious_period, contacts_per_day=contacts_per_day)