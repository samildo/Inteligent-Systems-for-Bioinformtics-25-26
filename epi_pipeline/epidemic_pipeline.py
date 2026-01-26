import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from agents import run_abm_simulation, Variant
from pepitope import pepitope_to_beta,pepitope_to_R0,pepitope_to_VE
from features import hamming_distance, compare_glycosylation, get_physicochemical_props


class EpidemicPipeline:
    def __init__(self):
        """Carrega modelos e referências na inicialização."""
        print("A carregar recursos do sistema...")
        self.model = joblib.load('../epi_pipeline/modelo/rf_epitope_model.joblib')
        self.feature_names = joblib.load('../epi_pipeline/modelo/model_features.joblib')
        self.refs = joblib.load('../epi_pipeline/modelo/reference_sequences.joblib')
        
        self.vacinas = self.refs['dict_vacinas']
        self.consensus = self.refs['consensus_seq']
        self.latest_vac_key = self.refs['latest_vaccine']
        print("Sistema pronto.")

    def extract_features(self, sequence: str) -> pd.DataFrame:
        """
        Transforma uma string de DNA/Proteína crua num DataFrame 
        com as colunas exatas que o Random Forest espera.
        """
        seq_str = str(sequence)
        
        # 1. Calcular Distâncias Genéticas (Hamming)
        # O modelo aprendeu distâncias a vacinas específicas, temos de replicar.
        features = {
            'dist_HK2014': hamming_distance(seq_str, self.vacinas['HK2014']),
            'dist_KAN2017': hamming_distance(seq_str, self.vacinas['KAN2017']),
            'dist_TAS2020': hamming_distance(seq_str, self.vacinas['TAS2020']),
            'dist_DAR2021': hamming_distance(seq_str, self.vacinas['DAR2021']),
            'dist_MAS2022': hamming_distance(seq_str, self.vacinas['MAS2022']),
            'dist_consensus': hamming_distance(seq_str, self.consensus)
        }
        
        # 2. Glicosilação
        # Para novas estirpes, comparamos com a vacina MAIS RECENTE disponível (ex: MAS2022)
        # para determinar ganho/perda de proteção atual.
        vac_seq_ref = self.vacinas[self.latest_vac_key]
        common, loss, gain = compare_glycosylation(seq_str, vac_seq_ref)
        features['glyco_common'] = common
        features['glyco_loss'] = loss
        features['glyco_gain'] = gain
        
        # 3. Propriedades Físico-Químicas
        iso, weight, aroma, inst, gravy = get_physicochemical_props(seq_str)
        features['isoeletric_point'] = iso
        features['molecular_weight'] = weight
        features['aromaticity'] = aroma
        features['instability_index'] = inst
        features['gravy'] = gravy
        
        # Criar DataFrame e garantir ordem das colunas
        df = pd.DataFrame([features])
        
        # Preenchimento de segurança para garantir ordem exata
        df = df[self.feature_names]
        
        return df

    def predict_and_simulate(self, 
                             sequence: str, 
                             # -- Parâmetros Gerais --
                             sim_days: int = 160, 
                             sim_N: int = 5000, 
                             sim_vacc_coverage: float = 0.4,
                             sim_initial_cases: int = 50,
                             # -- Parâmetros Avançados da Simulação (Novos) --
                             contacts_per_day: float = 10.0,
                             latent_period: float = 2.0,
                             infectious_period: float = 5.0,
                             distancing_factor: float = 1.0,
                             extra_vaccination: float = 0.0,
                             n_regions: int = 3,
                             intra_region_prob: float = 0.8,
                             random_seed: int = 42):
        """
        Pipeline completo com controlo total sobre os parâmetros do ABM.
        """
        # --- PASSO A: ML PREDICTION ---
        features_df = self.extract_features(sequence)
        p_epitope_pred = self.model.predict(features_df)[0]
        
        print(f"\n--- Resultados da Análise Molecular ---")
        print(f"pEpitope Previsto: {p_epitope_pred:.5f}")
        
        if p_epitope_pred > 0.19:
            print("ALERTA: Possível Falha Vacinal Completa (p > 0.19)")
        elif p_epitope_pred > 0.10:
            print("AVISO: Deriva Antigénica Moderada.")
        else:
            print("INFO: Proteção Vacinal Provável.")

        # --- PASSO B: CÁLCULO DE PARÂMETROS EPIDEMIOLÓGICOS ---
        ve_calc = pepitope_to_VE(p_epitope_pred)
        r0_calc = pepitope_to_R0(p_epitope_pred, vaccine_coverage=sim_vacc_coverage)
        
        # IMPORTANTE: Atualizar o cálculo do Beta com os novos parâmetros de contacto/período
        # Se mudares os contactos ou o período infeccioso, o beta tem de ajustar para manter o R0 coerente.
        beta_calc = pepitope_to_beta(
            p_epitope_pred, 
            vaccine_coverage=sim_vacc_coverage,
            contacts_per_day=contacts_per_day,   # <-- Agora usa o argumento da função
            infectious_period=infectious_period  # <-- Agora usa o argumento da função
        )
        
        print(f"Eficácia Vacinal (VE) Estimada: {ve_calc:.2%}")
        print(f"R0 Efetivo Estimado: {r0_calc:.2f}")

        # --- PASSO C: SIMULAÇÃO ABM ---
        predicted_variant = Variant(
            name="New_Strain_X",
            p_epitope=p_epitope_pred,
            VE=ve_calc,
            R0_eff=r0_calc,
            beta=beta_calc
        )
        
        variants_dict = {"New_Strain_X": predicted_variant}
        
        print(f"\n--- A Correr Simulação ({sim_days} dias, N={sim_N}) ---")
        
        # Chamada da função original com TODOS os parâmetros expostos
        sim_results = run_abm_simulation(
            variants=variants_dict,
            days=sim_days,
            N=sim_N,
            vaccine_coverage=sim_vacc_coverage,
            initial_infected={"New_Strain_X": sim_initial_cases},
            # Parâmetros adicionais mapeados:
            contacts_per_day=contacts_per_day,
            latent_period=latent_period,
            infectious_period=infectious_period,
            distancing_factor=distancing_factor,
            extra_vaccination=extra_vaccination,
            n_regions=n_regions,
            intra_region_prob=intra_region_prob,
            random_seed=random_seed
        )
        
        return sim_results, p_epitope_pred

    def plot_results(self, sim_df, title_suffix=""):
        plt.figure(figsize=(10, 5))
        plt.plot(sim_df["day"], sim_df["I"], label="Infetados (I)", color='crimson', linewidth=2)
        plt.title(f"Previsão de Dinâmica Epidémica {title_suffix}")
        plt.xlabel("Dias")
        plt.ylabel("População")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ==========================================
# EXEMPLO DE UTILIZAÇÃO
# ==========================================

if __name__ == "__main__":
    
    # 1. Instanciar Pipeline
    pipeline = EpidemicPipeline()
    
    # 2. Inserir uma sequência (ex: copiar uma string FASTA de uma variante recente)
    # Exemplo: Uma sequência hipotética (apenas um pedaço para teste)
    nova_sequencia_viral = "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTLVKTITDDQIEVTNATELVQSSSTGKICNNPHRILDGIDCTLIDALLGDPHCDVFQNE..." 
    # NOTA: Para funcionar bem, usa uma sequência completa de HA (~566 aminoácidos)
    
    # Se quiseres testar com uma sequência real do teu dataset original:
    #nova_sequencia_viral = str(dataset_ml.iloc[-1]['seq_original_se_tiveres']) 

    # 3. Executar
    df_resultado, p_val = pipeline.predict_and_simulate(
        sequence=nova_sequencia_viral,
        sim_days=160,
        sim_N=10000,
        sim_vacc_coverage=0.45, # Podes editar a cobertura vacinal aqui
        sim_initial_cases=100
    )
    
    # 4. Visualizar
    pipeline.plot_results(df_resultado, title_suffix=f"| pEpitope: {p_val:.3f}")