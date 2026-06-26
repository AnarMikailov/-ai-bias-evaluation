"""
COMPAS Recidivism Dataset - CSV Generatoru

Melumat menbeleri:
  - ProPublica (2016). "Machine Bias." https://github.com/propublica/compas-analysis
  - Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016).
    Machine Bias: There's software used across the country to predict future criminals.
    And it's biased against blacks. ProPublica.

Dataset haqqinda:
  ABŞ-da Broward County məhkəmələrinin istifadə etdiyi COMPAS sistemi
  məhkumların 2 il ərzində yenidən cinayət edib-etməyəcəyini proqnozlaşdırır.
  ProPublica araşdırması göstərdi ki, sistem Qaradərili şəxsləri Ağdərilərə
  nisbətən 2 dəfə çox "yüksək risk" kimi etiketləyir.

Cixis fayllar:
  data/compas_cinayat_qerezli.csv   - Original COMPAS (irqi qerez movcuddur)
  data/compas_cinayat_qerezsiz.csv  - Stratified undersampling ile balanslasmis versiya
"""

import os
import numpy as np
import pandas as pd
import aif360

try:
    pd.options.future.infer_string = False
except AttributeError:
    pass

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_compas_raw():
    raw_path = os.path.join(
        os.path.dirname(aif360.__file__),
        'data', 'raw', 'compas', 'compas-scores-two-years.csv'
    )
    df = pd.read_csv(raw_path)

    # AIF360 ile eyni filterleme
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']
    df = df[df['race'].isin(['African-American', 'Caucasian'])]

    cols = [
        'age', 'c_charge_degree', 'race', 'sex',
        'priors_count', 'days_b_screening_arrest',
        'decile_score', 'score_text', 'two_year_recid'
    ]
    df = df[cols].copy()

    # Insan oxuyan adlar
    df = df.rename(columns={
        'c_charge_degree':        'ittiham_derece',
        'priors_count':           'onceki_cinayetler',
        'days_b_screening_arrest':'saxlanma_gun_ferqi',
        'decile_score':           'risk_bal',
        'score_text':             'risk_qiymeti',
        'two_year_recid':         'yeniden_cinayat'
    })
    df = df.dropna().reset_index(drop=True)
    return df


def print_stats(label, df):
    aa   = df[df['race'] == 'African-American']['yeniden_cinayat'].mean()
    cauc = df[df['race'] == 'Caucasian']['yeniden_cinayat'].mean()
    di   = aa / cauc if cauc > 0 else float('inf')
    print(f"\n{label}")
    print(f"  Umumi satr        : {len(df):,}")
    print(f"  African-American yeniden cinayat nisbeti : {aa:.1%}")
    print(f"  Caucasian        yeniden cinayat nisbeti : {cauc:.1%}")
    print(f"  Disparate Impact (AA/Cauc)               : {di:.3f}")


def create_unbiased(df):
    """
    Stratified undersampling:
    AA qrupunun yeniden_cinayat nisbetin Caucasian nisbetin endirir.
    Belelikle her iki irq ucun eyni nisbat tamin edilir (demographic parity).
    """
    cauc_rate = df[df['race'] == 'Caucasian']['yeniden_cinayat'].mean()

    aa_pos  = df[(df['race'] == 'African-American') & (df['yeniden_cinayat'] == 1)]
    aa_neg  = df[(df['race'] == 'African-American') & (df['yeniden_cinayat'] == 0)]
    cauc    = df[df['race'] == 'Caucasian']

    # AA pozitiv satirleri azalt ki nisbeti Caucasian ile eyni olsun
    target_aa_pos = int(cauc_rate / (1 - cauc_rate) * len(aa_neg))
    target_aa_pos = min(target_aa_pos, len(aa_pos))

    aa_pos_sampled = aa_pos.sample(n=target_aa_pos, random_state=42)

    balanced = pd.concat([aa_pos_sampled, aa_neg, cauc])
    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == '__main__':
    print("COMPAS dataseti yuklenilir...")
    df = load_compas_raw()

    print_stats("QEREZLI DATA (ProPublica COMPAS original):", df)
    biased_path = os.path.join(OUTPUT_DIR, 'compas_cinayat_qerezli.csv')
    df.to_csv(biased_path, index=False)
    print(f"  => Saxlanildi: {biased_path}")

    df_balanced = create_unbiased(df)
    print_stats("QEREZSIZ DATA (balanslasmis):", df_balanced)
    unbiased_path = os.path.join(OUTPUT_DIR, 'compas_cinayat_qerezsiz.csv')
    df_balanced.to_csv(unbiased_path, index=False)
    print(f"  => Saxlanildi: {unbiased_path}")

    print("\nHer iki fayl data/ qovluqunda hazirdir.")
    print("\nSitat ucun:")
    print("  Angwin, J. et al. (2016). Machine Bias. ProPublica.")
    print("  https://github.com/propublica/compas-analysis")
