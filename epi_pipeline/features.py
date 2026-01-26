import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def hamming_distance(s1, s2):
    # Ignora gaps (-) no cálculo da distância para não penalizar seqs parciais nas pontas
    return sum(1 for a, b in zip(s1, s2) if a != b and a != "-" and b != "-")

def get_glycosylation_sites(sequence):
    # Regex para o motivo NXT/S onde X != P
    # Usamos lookahead (?=...) para encontrar motivos sobrepostos se existirem
    sequence = str(sequence)
    sites = set()
    for m in re.finditer(r'(?=N[^P][ST])', sequence):
        sites.add(m.start() + 1) # Posição biológica (base 1)
    return sites

def compare_glycosylation(sample_seq, vaccine_seq):
    sample_sites = get_glycosylation_sites(sample_seq)
    vaccine_sites = get_glycosylation_sites(vaccine_seq)
    
    common = sample_sites.intersection(vaccine_sites)
    loss = vaccine_sites - sample_sites
    gain = sample_sites - vaccine_sites
    
    return len(common), len(loss), len(gain)


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def get_physicochemical_props(sequence):
    # Remove gaps para análise correta
    sequence = str(sequence)
    
    clean_seq = "".join(
        aa for aa in sequence
        if aa in VALID_AA
    )

    # proteção extra
    if len(clean_seq) == 0:
        return (None, None, None, None, None)

    analysis = ProteinAnalysis(clean_seq)
    
    return analysis.isoelectric_point(),analysis.molecular_weight(),analysis.aromaticity(), analysis.instability_index(), analysis.gravy()