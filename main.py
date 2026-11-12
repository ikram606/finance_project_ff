import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) Charger les données
# =========================
df = pd.read_csv(r"C:\Users\pc\Downloads\tech_stocks_with_fundamentals_clean.csv")

print("========== APERCU DU DATASET ==========")
print(df.head())
print("\nColonnes du dataset :")
print(df.columns)

# =========================
# 2) Préparation
# =========================
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Ticker', 'Date'])
df = df.dropna()

# =========================
# 3) Calcul des rendements
# =========================
df['return'] = df.groupby('Ticker')['Close'].pct_change()
df = df.dropna()

print("\n========== PREMIERS RENDEMENTS ==========")
print(df[['Ticker', 'Date', 'return']].head())

# =========================
# 4) Taux sans risque
# =========================
rf_annual = 0.03
rf_daily = rf_annual / 252

print("\n========== TAUX SANS RISQUE ==========")
print(f"Rf annuel     = {rf_annual:.6f}")
print(f"Rf journalier = {rf_daily:.8f}")

# =========================
# 5) Sélection de AAPL
# =========================
aapl = df[df['Ticker'] == 'AAPL'].copy()

# =========================
# 6) Rendement du marché
# =========================
market = df.groupby('Date')['return'].mean().reset_index()
market.rename(columns={'return': 'market_return'}, inplace=True)

# Fusion AAPL + marché
aapl = aapl.merge(market, on='Date')

# Rendements excédentaires
aapl['excess_return'] = aapl['return'] - rf_daily
aapl['market_excess_return'] = aapl['market_return'] - rf_daily

print("\n========== AAPL + MARCHE ==========")
print(aapl[['Date', 'Ticker', 'return', 'market_return', 'excess_return', 'market_excess_return']].head())

# =========================
# 7) CAPM manuel avec Rf
# =========================
Ri_excess = aapl['excess_return'].values
Rm_excess = aapl['market_excess_return'].values

Ri_excess_mean = np.mean(Ri_excess)
Rm_excess_mean = np.mean(Rm_excess)

cov_im = np.mean((Ri_excess - Ri_excess_mean) * (Rm_excess - Rm_excess_mean))
var_m = np.mean((Rm_excess - Rm_excess_mean) ** 2)

beta_capm = cov_im / var_m
alpha_capm = Ri_excess_mean - beta_capm * Rm_excess_mean

Ri_excess_hat = alpha_capm + beta_capm * Rm_excess

residus_capm = Ri_excess - Ri_excess_hat

SSE_capm = np.sum((Ri_excess - Ri_excess_hat) ** 2)
SST_capm = np.sum((Ri_excess - Ri_excess_mean) ** 2)
R2_capm = 1 - (SSE_capm / SST_capm)

print("\n========== RESULTATS CAPM ==========")
print(f"Rendement excédentaire moyen AAPL    = {Ri_excess_mean:.6f}")
print(f"Rendement excédentaire moyen marché  = {Rm_excess_mean:.6f}")
print(f"Cov(Ri-Rf, Rm-Rf)                    = {cov_im:.6f}")
print(f"Var(Rm-Rf)                           = {var_m:.6f}")
print(f"Beta_CAPM                            = {beta_capm:.6f}")
print(f"Alpha_CAPM                           = {alpha_capm:.6f}")
print(f"R²_CAPM                              = {R2_capm:.6f}")

# =========================
# 8) Construction du ratio Book-to-Market
# =========================
# Approximation :
# BM_ratio = Book_Value / Close
df['BM_ratio'] = df['Book_Value'] / df['Close']

print("\n========== EXEMPLE BM RATIO ==========")
print(df[['Ticker', 'Date', 'Book_Value', 'Close', 'BM_ratio']].head())

# =========================
# 9) Construction des facteurs SMB et HML
# =========================
factor_rows = []

for date, group in df.groupby('Date'):
    group = group.dropna(subset=['Market_Cap', 'BM_ratio', 'return']).copy()

    if len(group) < 4:
        continue

    size_median = group['Market_Cap'].median()
    bm_median = group['BM_ratio'].median()

    # Groupes taille
    small_group = group[group['Market_Cap'] <= size_median]
    big_group = group[group['Market_Cap'] > size_median]

    # Groupes valeur
    high_group = group[group['BM_ratio'] >= bm_median]
    low_group = group[group['BM_ratio'] < bm_median]

    if len(small_group) == 0 or len(big_group) == 0 or len(high_group) == 0 or len(low_group) == 0:
        continue

    smb = small_group['return'].mean() - big_group['return'].mean()
    hml = high_group['return'].mean() - low_group['return'].mean()
    market_return = group['return'].mean()

    factor_rows.append([date, market_return, smb, hml])

factors = pd.DataFrame(factor_rows, columns=['Date', 'market_return', 'SMB', 'HML'])

print("\n========== FACTEURS FAMA-FRENCH ==========")
print(factors.head())

# =========================
# 10) Fusion AAPL avec SMB et HML
# =========================
# On garde market_excess_return déjà présent dans aapl
ff = aapl.merge(
    factors[['Date', 'SMB', 'HML']],
    on='Date',
    how='inner'
)

print("\n========== DATASET FINAL FAMA-FRENCH ==========")
print(ff[['Date', 'Ticker', 'excess_return', 'market_excess_return', 'SMB', 'HML']].head())

# =========================
# 11) Régression matricielle manuelle Fama-French
# =========================
# Y = Ri - Rf
Y = ff['excess_return'].values.reshape(-1, 1)

# X = [1, (Rm-Rf), SMB, HML]
X = np.column_stack((
    np.ones(len(ff)),
    ff['market_excess_return'].values,
    ff['SMB'].values,
    ff['HML'].values
))

print("\nDimensions de X :", X.shape)
print("Dimensions de Y :", Y.shape)

# Formule :
# beta_hat = (X^T X)^(-1) X^T Y
Xt = X.T
XtX = Xt @ X
XtX_inv = np.linalg.inv(XtX)
XtY = Xt @ Y
beta_hat = XtX_inv @ XtY

alpha_ff = beta_hat[0, 0]
beta_m_ff = beta_hat[1, 0]
s_i = beta_hat[2, 0]
h_i = beta_hat[3, 0]

# =========================
# 12) Valeurs estimées + ajustement
# =========================
Y_hat = X @ beta_hat
residus_ff = Y - Y_hat

SSE_ff = np.sum((Y - Y_hat) ** 2)
SST_ff = np.sum((Y - np.mean(Y)) ** 2)
R2_ff = 1 - (SSE_ff / SST_ff)

print("\n========== RESULTATS FAMA-FRENCH ==========")
print(f"Alpha_FF                              = {alpha_ff:.6f}")
print(f"Beta_m_FF                             = {beta_m_ff:.6f}")
print(f"s_i (sensibilité à SMB)               = {s_i:.6f}")
print(f"h_i (sensibilité à HML)               = {h_i:.6f}")
print(f"R²_FamaFrench                         = {R2_ff:.6f}")

# =========================
# 13) Comparaison finale
# =========================
print("\n========== COMPARAISON CAPM vs FAMA-FRENCH ==========")
print(f"R²_CAPM                               = {R2_capm:.6f}")
print(f"R²_FamaFrench                         = {R2_ff:.6f}")

if R2_ff > R2_capm:
    print("Conclusion : Fama-French ajuste mieux les rendements excédentaires de AAPL que le CAPM.")
elif R2_ff < R2_capm:
    print("Conclusion : le CAPM ajuste mieux les rendements excédentaires de AAPL que Fama-French.")
else:
    print("Conclusion : les deux modèles ont le même ajustement.")
# =========================
# 14) GRAPHIQUES
# =========================

# ----- Graphique 1 : Evolution du prix de clôture de AAPL -----
plt.figure(figsize=(10, 5))
plt.plot(aapl['Date'], aapl['Close'])
plt.title("Evolution du prix de clôture de AAPL")
plt.xlabel("Date")
plt.ylabel("Prix de clôture")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Graphique 2 : Rendement excédentaire AAPL vs Marché -----
plt.figure(figsize=(10, 5))
plt.plot(aapl['Date'], aapl['excess_return'], label="AAPL excess return")
plt.plot(aapl['Date'], aapl['market_excess_return'], label="Market excess return")
plt.title("Rendement excédentaire de AAPL vs Marché")
plt.xlabel("Date")
plt.ylabel("Rendement excédentaire")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Graphique 3 : Facteurs SMB et HML -----
plt.figure(figsize=(10, 5))
plt.plot(factors['Date'], factors['SMB'], label="SMB")
plt.plot(factors['Date'], factors['HML'], label="HML")
plt.title("Evolution des facteurs SMB et HML")
plt.xlabel("Date")
plt.ylabel("Valeur du facteur")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Graphique 4 : Rendements réels vs estimés par CAPM -----
capm_df = pd.DataFrame({
    'Date': aapl['Date'],
    'Real_Excess_Return': Ri_excess,
    'CAPM_Estimated_Return': Ri_excess_hat
})

plt.figure(figsize=(10, 5))
plt.plot(capm_df['Date'], capm_df['Real_Excess_Return'], label="Rendement réel")
plt.plot(capm_df['Date'], capm_df['CAPM_Estimated_Return'], label="Rendement estimé CAPM")
plt.title("CAPM : Rendements réels vs estimés")
plt.xlabel("Date")
plt.ylabel("Rendement excédentaire")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Graphique 5 : Rendements réels vs estimés par Fama-French -----
ff_plot = pd.DataFrame({
    'Date': ff['Date'],
    'Real_Excess_Return': Y.flatten(),
    'FF_Estimated_Return': Y_hat.flatten()
})

plt.figure(figsize=(10, 5))
plt.plot(ff_plot['Date'], ff_plot['Real_Excess_Return'], label="Rendement réel")
plt.plot(ff_plot['Date'], ff_plot['FF_Estimated_Return'], label="Rendement estimé Fama-French")
plt.title("Fama-French : Rendements réels vs estimés")
plt.xlabel("Date")
plt.ylabel("Rendement excédentaire")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(aapl['Date'], aapl['Close'])
plt.title("Evolution du prix de clôture de AAPL")
plt.xlabel("Date")
plt.ylabel("Prix de clôture")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/aapl_close.png")
plt.show()
import matplotlib.pyplot as plt

# =========================
# Graphique 1 : Prix AAPL
# =========================
plt.figure(figsize=(10, 5))

plt.plot(aapl['Date'], aapl['Close'])

plt.title("Evolution du prix de clôture de AAPL")
plt.xlabel("Date")
plt.ylabel("Prix de clôture")

plt.grid(True)

plt.tight_layout()

# Sauvegarde dans le dossier plots
plt.savefig("plots/aapl_close.png")

plt.show()