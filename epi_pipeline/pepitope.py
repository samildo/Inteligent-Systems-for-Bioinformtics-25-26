import numpy as np

def pepitope_to_VE(p_epitope: float) -> float:
    """
    Calculates theoretical Vaccine Efficacy (VE) based on the pEpitope antigenic distance metric.
    
    Uses the linear relationship proposed by Deem (VE ≈ -2.47 * p + 0.47), clipped to [0, 1].
    
    Args:
        p_epitope (float): The dominant epitope distance (0 to 1).
        
    Returns:
        float: Estimated vaccine efficacy (0.0 to 1.0).
    """
    ve = -2.47 * p_epitope + 0.47
    return float(np.clip(ve, 0.0, 1.0))

def pepitope_to_R0(p_epitope: float,
                     base_R0: float = 1.5,
                     vaccine_coverage: float = 0.4) -> float:
    """
    Estimates the effective reproductive number (Reff) by accounting for vaccine escape.
    
    Higher pEpitope values reduce VE, increasing the effective susceptible population.
    
    Args:
        p_epitope (float): Antigenic distance.
        base_R0 (float): Baseline R0 for a fully susceptible population (or perfect match).
        vaccine_coverage (float): Fraction of the population vaccinated.
        
    Returns:
        float: The effective reproductive number.
    """
    ve = pepitope_to_VE(p_epitope)
    # Fração efetivamente protegida: vacinados × VE
    protected = vaccine_coverage * ve
    susceptible_fraction = 1.0 - protected
    # R0 efetivo proporcional à fração suscetível
    return base_R0 * susceptible_fraction

def R0_to_beta(R0: float,
                infectious_period: float = 5.0,
                contacts_per_day: float = 10.0) -> float:
    """
    Derives transmission probability per contact (beta) from R0.
    
    Based on the SIR relation: R0 = beta * contacts * duration.
    
    Args:
        R0 (float): The reproductive number.
        infectious_period (float): Average duration of infectiousness in days.
        contacts_per_day (float): Average daily contacts per agent.
        
    Returns:
        float: Transmission probability (beta).
    """
    return R0 / (contacts_per_day * infectious_period)

def pepitope_to_beta(p_epitope: float,
                     base_R0: float = 1.5,
                     vaccine_coverage: float = 0.4,
                     infectious_period: float = 5.0,
                     contacts_per_day: float = 10.0) -> float:
    """
    Directly maps antigenic distance (pEpitope) to the transmission parameter (beta).
    
    A convenience wrapper combining VE estimation, Reff calculation, and beta derivation.
    
    Returns:
        float: The calculated transmission probability beta.
    """
    R0 = pepitope_to_R0(p_epitope, base_R0=base_R0, vaccine_coverage=vaccine_coverage)
    return R0_to_beta(R0, infectious_period=infectious_period, contacts_per_day=contacts_per_day)