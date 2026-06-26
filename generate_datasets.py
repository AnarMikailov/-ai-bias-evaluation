"""
Bu skript iki CSV fayl yaradir:
  qerezli_data.csv  - Original Adult Income dataseti (UCI ML Repository, 1994)
  qerezsiz_data.csv - Eyni datanin stratified undersampling ile balanslasmis versiyasi

Akademik menbə: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
Ucin University of California, Irvine, School of Information and Computer Sciences.
"""

import os
import numpy as np
import pandas as pd
import aif360

try:
    pd.options.future.infer_string = False
except AttributeError:
    pass

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_raw():
    base = os.path.join(os.path.dirname(aif360.__file__), 'data', 'raw', 'adult')
    train = pd.read_csv(
        os.path.join(base, 'adult.data'),
        names=COLUMNS, sep=r',\s*', engine='python', na_values=['?']
    )
    test = pd.read_csv(
        os.path.join(base, 'adult.test'),
        names=COLUMNS, sep=r',\s*', engine='python', na_values=['?'], skiprows=1
    )
    df = pd.concat([train, test], ignore_index=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df['income'] = df['income'].str.rstrip('.')
    df = df.drop(columns=['fnlwgt', 'education-num'])
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def create_unbiased(df):
    """
    Stratified undersampling: kisi >50K qrupunu azaldaraq
    her iki cinsin gelir nisbetini bərabərlesdirir (demographic parity).
    """
    female_pos_rate = (df[df['sex'] == 'Female']['income'] == '>50K').mean()

    male_pos  = df[(df['sex'] == 'Male')   & (df['income'] == '>50K')]
    male_neg  = df[(df['sex'] == 'Male')   & (df['income'] == '<=50K')]
    female_pos = df[(df['sex'] == 'Female') & (df['income'] == '>50K')]
    female_neg = df[(df['sex'] == 'Female') & (df['income'] == '<=50K')]

    # Kisi >50K-ni azalt ki nisbeti qadin nisbetine esit olsun
    target = int(female_pos_rate / (1 - female_pos_rate) * len(male_neg))
    target = min(target, len(male_pos))

    male_pos_sampled = male_pos.sample(n=target, random_state=42)

    balanced = pd.concat([male_pos_sampled, male_neg, female_pos, female_neg])
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced


def print_stats(label, df):
    male_rate   = (df[df['sex'] == 'Male']['income']   == '>50K').mean()
    female_rate = (df[df['sex'] == 'Female']['income'] == '>50K').mean()
    print(f"\n{label}")
    print(f"  Umumi satr sayi : {len(df):,}")
    print(f"  Kisi  >50K nisbeti : {male_rate:.1%}")
    print(f"  Qadin >50K nisbeti : {female_rate:.1%}")
    print(f"  Disparate Impact (taxmini) : {female_rate/male_rate:.3f}")


if __name__ == '__main__':
    print("Data yuklenilir...")
    df = load_raw()

    print_stats("QEREZLI DATA (original):", df)
    biased_path = os.path.join(OUTPUT_DIR, 'qerezli_data.csv')
    df.to_csv(biased_path, index=False)
    print(f"  => Saxlanildi: {biased_path}")

    df_balanced = create_unbiased(df)
    print_stats("QEREZSIZ DATA (balanslasmis):", df_balanced)
    unbiased_path = os.path.join(OUTPUT_DIR, 'qerezsiz_data.csv')
    df_balanced.to_csv(unbiased_path, index=False)
    print(f"  => Saxlanildi: {unbiased_path}")

    print("\nHər iki fayl hazirdir!")
