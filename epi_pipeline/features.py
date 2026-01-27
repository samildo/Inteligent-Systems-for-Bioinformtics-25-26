import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def hamming_distance(s1, s2):
    """
    Computes the Hamming distance between two sequences, ignoring gap characters.
    
    Args:
        s1 (str): The first sequence.
        s2 (str): The second sequence.
        
    Returns:
        int: The count of mismatching positions, excluding gaps ('-').
    """
    return sum(1 for a, b in zip(s1, s2) if a != b and a != "-" and b != "-")

def get_glycosylation_sites(sequence):
    """
    Identifies N-linked glycosylation sites (N-X-S/T motif, X!=P).
    
    Uses regex lookahead to detect overlapping motifs.
    
    Args:
        sequence (str): The protein sequence.
        
    Returns:
        set: A set of 1-based indices representing the start of each site.
    """
    sequence = str(sequence)
    sites = set()
    for m in re.finditer(r'(?=N[^P][ST])', sequence):
        sites.add(m.start() + 1) # Posição biológica (base 1)
    return sites

def compare_glycosylation(sample_seq, vaccine_seq):
    """
    Quantifies differences in glycosylation profiles between a sample and a reference.
    
    Args:
        sample_seq (str): The query sequence (e.g., viral isolate).
        vaccine_seq (str): The reference sequence (e.g., vaccine strain).
        
    Returns:
        tuple: Counts of (common sites, lost sites, gained sites).
    """
    sample_sites = get_glycosylation_sites(sample_seq)
    vaccine_sites = get_glycosylation_sites(vaccine_seq)
    
    common = sample_sites.intersection(vaccine_sites)
    loss = vaccine_sites - sample_sites
    gain = sample_sites - vaccine_sites
    
    return len(common), len(loss), len(gain)


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def get_physicochemical_props(sequence):
    """
    Calculates physicochemical properties using Biopython's ProteinAnalysis.
    
    Sanitizes the input by removing gaps and invalid amino acids before calculation.
    
    Args:
        sequence (str): The protein sequence.
        
    Returns:
        tuple: (Isoelectric Point, Molecular Weight, Aromaticity, Instability Index, GRAVY)
               Returns a tuple of None if the sequence is empty after cleaning.
    """
    sequence = str(sequence)
    
    clean_seq = "".join(
        aa for aa in sequence
        if aa in VALID_AA
    )

    if len(clean_seq) == 0:
        return (None, None, None, None, None)

    analysis = ProteinAnalysis(clean_seq)
    
    return analysis.isoelectric_point(),analysis.molecular_weight(),analysis.aromaticity(), analysis.instability_index(), analysis.gravy()