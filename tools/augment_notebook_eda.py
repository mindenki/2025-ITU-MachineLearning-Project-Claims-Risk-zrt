import json
import uuid
from pathlib import Path


def make_id():
    return str(uuid.uuid4())[:8]


def md_cell(text):
    return {
        "cell_type": "markdown",
        "id": make_id(),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "id": make_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_cells():
    cells = []

    # Overview
    cells.append(md_cell(
        "## EDA Enhancements\n\n"
        "This section adds exposure-aware exploratory analysis for claims frequency,\n"
        "data quality checks, target-aware plots, correlation diagnostics, and a quick baseline.\n"
    ))

    # Imports and config
    cells.append(code_cell(
        """
        # Utilities for EDA additions
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from math import sqrt
        sns.set_context('talk'); sns.set_style('whitegrid')
        pd.set_option('display.max_columns', 100)

        # Expect a DataFrame named df with columns: ClaimNb, Exposure, Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region
        assert 'df' in globals(), 'Please ensure your dataset is loaded into a DataFrame named `df` before running these cells.'
        """
    ))

    # Target definition and basic transforms
    cells.append(md_cell("### Target and Transforms"))
    cells.append(code_cell(
        """
        # Filter invalid exposure and create target/feature transforms
        df = df.copy()
        df = df[df['Exposure'] > 0].reset_index(drop=True)
        df['claim_rate'] = df['ClaimNb'] / df['Exposure']
        df['log_density'] = np.log1p(df['Density']) if 'Density' in df.columns else np.nan
        print(df[['ClaimNb','Exposure']].describe())
        print('Claim rate mean:', df['claim_rate'].mean())
        """
    ))

    # Data quality checks
    cells.append(md_cell("### Data Quality Checks"))
    cells.append(code_cell(
        """
        issues = {}
        # Duplicates by policy ID if available
        if 'IDpol' in df.columns:
            dup = df.duplicated(subset=['IDpol']).sum()
            issues['duplicate_IDpol'] = int(dup)
        # Impossible values
        issues['DrivAge_lt_18'] = int((df['DrivAge'] < 18).sum()) if 'DrivAge' in df.columns else None
        issues['VehAge_lt_0'] = int((df['VehAge'] < 0).sum()) if 'VehAge' in df.columns else None
        issues['Exposure_le_0'] = int((df['Exposure'] <= 0).sum())
        # Near-zero variance (categoricals/numerics)
        nzv = {}
        for c in df.columns:
            vc = df[c].value_counts(dropna=False)
            if len(vc) > 0 and (vc.iloc[0] / max(len(df), 1) > 0.98):
                nzv[c] = float(vc.iloc[0] / len(df))
        print('Issues summary:', issues)
        print('Near-zero variance features (top freq share):', nzv)
        """
    ))

    # Missingness analysis
    cells.append(md_cell("### Missingness Overview"))
    cells.append(code_cell(
        """
        missing_pct = df.isna().mean().sort_values(ascending=False)
        display(missing_pct.to_frame('missing_pct').style.format({'missing_pct': '{:.1%}'}))
        # Correlation of missingness with claim_rate
        corrs = {}
        for c in df.columns:
            if df[c].isna().any():
                m = df[c].isna().astype(float)
                cor = np.corrcoef(m, df['claim_rate'])[0,1] if df['claim_rate'].std() > 0 else np.nan
                corrs[c] = cor
        print('Missingness correlation with claim_rate (abs sorted):')
        for k,v in sorted(corrs.items(), key=lambda kv: (0 if kv[1] is None else -abs(kv[1] or 0))):
            print(f"{k}: {v:.3f}")
        """
    ))

    # Univariate distributions
    cells.append(md_cell("### Univariate Distributions"))
    cells.append(code_cell(
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Numeric histograms
        for c in num_cols:
            if c in ['ClaimNb', 'Exposure', 'claim_rate']:
                pass
            plt.figure(figsize=(6,4))
            sns.histplot(data=df, x=c, kde=False)
            plt.title(f'Distribution of {c}')
            plt.tight_layout(); plt.show()

        # Categorical counts (top-k)
        for c in cat_cols:
            plt.figure(figsize=(7,4))
            vc = df[c].value_counts().head(15)
            sns.barplot(x=vc.values, y=vc.index, orient='h')
            plt.title(f'Counts of {c} (top 15)')
            plt.tight_layout(); plt.show()
        """
    ))

    # Categorical cardinality
    cells.append(md_cell("### Categorical Cardinality and Rare Levels"))
    cells.append(code_cell(
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for c in cat_cols:
            counts = df[c].value_counts(dropna=False)
            rare = counts[counts < max(50, 0.01 * len(df))]
            print(f"{c}: {len(counts)} levels; rare (<1% or <50): {len(rare)}")
        """
    ))

    # Exposure-weighted rates by category
    cells.append(md_cell("### Exposure-Weighted Claim Rates by Category"))
    cells.append(code_cell(
        """
        cat_to_plot = [c for c in ['Area','VehGas','VehBrand','Region'] if c in df.columns]
        for c in cat_to_plot:
            grp = df.groupby(c).agg(Claims=('ClaimNb','sum'), Exp=('Exposure','sum'), n=('ClaimNb','size')).reset_index()
            grp['rate'] = grp['Claims'] / grp['Exp']
            grp = grp.sort_values('rate', ascending=False)
            plt.figure(figsize=(8, max(3, 0.3*len(grp))))
            sns.barplot(data=grp, x='rate', y=c)
            plt.title(f'Exposure-weighted claim rate by {c}')
            plt.tight_layout(); plt.show()
        """
    ))

    # Numeric relationships via binning
    cells.append(md_cell("### Numeric Relationships vs claim_rate (Exposure-Weighted Binning)"))
    cells.append(code_cell(
        """
        def binned_rate(x, y_claims, y_exp, bins=12):
            q = np.linspace(0, 1, bins+1)
            edges = np.unique(np.quantile(x, q))
            idx = np.clip(np.searchsorted(edges, x, side='right')-1, 0, len(edges)-2)
            dfb = pd.DataFrame({'bin': idx, 'x': x, 'claims': y_claims, 'exp': y_exp})
            g = dfb.groupby('bin').agg(xm=('x','mean'), claims=('claims','sum'), exp=('exp','sum')).reset_index(drop=True)
            g['rate'] = g['claims']/g['exp']
            return g

        num_for_rel = [c for c in ['DrivAge','VehAge','VehPower','Density'] if c in df.columns]
        for c in num_for_rel:
            g = binned_rate(df[c].astype(float), df['ClaimNb'], df['Exposure'])
            plt.figure(figsize=(6,4))
            sns.lineplot(data=g, x='xm', y='rate', marker='o')
            plt.title(f'Claim rate vs {c} (binned)')
            plt.xlabel(c)
            plt.tight_layout(); plt.show()
        """
    ))

    # Interactions heatmaps
    cells.append(md_cell("### Interactions"))
    cells.append(code_cell(
        """
        def interaction_heatmap(a, b):
            grp = df.groupby([a,b]).agg(Claims=('ClaimNb','sum'), Exp=('Exposure','sum')).reset_index()
            grp['rate'] = grp['Claims']/grp['Exp']
            pivot = grp.pivot(index=a, columns=b, values='rate')
            plt.figure(figsize=(8,6))
            sns.heatmap(pivot, cmap='viridis')
            plt.title(f'Claim rate heatmap: {a} x {b}')
            plt.tight_layout(); plt.show()

        pairs = []
        if 'Area' in df.columns and 'VehPower' in df.columns: pairs.append(('Area','VehPower'))
        if 'Area' in df.columns and 'Density' in df.columns: pairs.append(('Area','Density'))
        if 'BonusMalus' in df.columns and 'DrivAge' in df.columns: pairs.append(('BonusMalus','DrivAge'))
        for a,b in pairs:
            # Bin numeric second variable if needed
            if str(df[b].dtype) != 'object' and df[b].nunique() > 15:
                df['_tmp_bin'] = pd.qcut(df[b], q=min(10, df[b].nunique()), duplicates='drop')
                interaction_heatmap(a, '_tmp_bin')
                df.drop(columns=['_tmp_bin'], inplace=True)
            else:
                interaction_heatmap(a, b)
        """
    ))

    # Correlations and Cramer's V
    cells.append(md_cell("### Correlation and Association"))
    cells.append(code_cell(
        """
        # Spearman correlations among numeric columns
        num_cols2 = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['ClaimNb','Exposure']]
        if len(num_cols2) > 1:
            corr = df[num_cols2].corr(method='spearman')
            display(corr)
            plt.figure(figsize=(6,5))
            sns.heatmap(corr, vmin=-1, vmax=1, cmap='coolwarm', annot=False)
            plt.title('Spearman Correlation (numeric)')
            plt.tight_layout(); plt.show()

        # Cramer's V for selected categoricals
        def cramers_v(x, y):
            tab = pd.crosstab(x, y)
            n = tab.values.sum()
            row_sums = tab.sum(axis=1).values.reshape(-1,1)
            col_sums = tab.sum(axis=0).values.reshape(1,-1)
            expected = row_sums @ col_sums / n
            chi2 = ((tab.values - expected)**2 / np.where(expected==0, 1, expected)).sum()
            phi2 = chi2 / n
            r, k = tab.shape
            return sqrt(phi2 / max(1, min(r-1, k-1)))

        cat_for_assoc = [c for c in ['Area','VehBrand','VehGas','Region'] if c in df.columns]
        for i in range(len(cat_for_assoc)):
            for j in range(i+1, len(cat_for_assoc)):
                a,b = cat_for_assoc[i], cat_for_assoc[j]
                try:
                    v = cramers_v(df[a], df[b])
                    print(f"Cramers V {a}-{b}: {v:.3f}")
                except Exception as e:
                    print(f'CramersV failed for {a},{b}:', e)
        """
    ))

    # Insurance-specific checks
    cells.append(md_cell("### Insurance-Specific Checks"))
    cells.append(code_cell(
        """
        zero_share = (df['ClaimNb'] == 0).mean()
        pos = df.loc[df['ClaimNb']>0, 'ClaimNb']
        print(f'Zero-claim share: {zero_share:.2%}; positive ClaimNb mean {pos.mean():.3f}, var {pos.var():.3f}')
        mu = df['ClaimNb'].mean(); var = df['ClaimNb'].var()
        print(f'Overall ClaimNb mean {mu:.4f}, var {var:.4f}, var/mean {var/mu if mu>0 else np.nan:.2f}')
        # Monotonicity checks via Spearman between binned feature and rate
        for c in [k for k in ['BonusMalus','DrivAge'] if k in df.columns]:
            g = df.groupby(pd.qcut(df[c], q=min(10, df[c].nunique()), duplicates='drop')) \
                .agg(rate=('claim_rate','mean')).reset_index()
            g['x'] = range(len(g))
            rho = g[['x','rate']].corr(method='spearman').iloc[0,1]
            print(f'Monotonicity (Spearman) of {c} vs rate (by bins): {rho:.3f}')
        """
    ))

    # Spatial overview by Region
    cells.append(md_cell("### Spatial Overview (Region)"))
    cells.append(code_cell(
        """
        if 'Region' in df.columns:
            grp = df.groupby('Region').agg(Claims=('ClaimNb','sum'), Exp=('Exposure','sum')).reset_index()
            grp['rate'] = grp['Claims']/grp['Exp']
            grp = grp.sort_values('rate', ascending=False)
            plt.figure(figsize=(8, max(3, 0.25*len(grp))))
            sns.barplot(data=grp, x='rate', y='Region')
            plt.title('Claim rate by Region')
            plt.tight_layout(); plt.show()
        else:
            print('Region column not found; skipping spatial overview.')
        """
    ))

    # Leakage & splits (guidance and helper split)
    cells.append(md_cell("### Leakage & Train/Test Splits"))
    cells.append(code_cell(
        """
        # Train/Test split by IDpol if available to avoid leakage across policies
        from sklearn.model_selection import train_test_split
        if 'IDpol' in df.columns:
            uniq = df['IDpol'].dropna().unique()
            train_ids, test_ids = train_test_split(uniq, test_size=0.2, random_state=42)
            trn = df[df['IDpol'].isin(train_ids)].copy()
            tst = df[df['IDpol'].isin(test_ids)].copy()
            print('Split sizes (rows):', len(trn), len(tst))
        else:
            trn, tst = train_test_split(df, test_size=0.2, random_state=42)
            print('Row-wise split used; consider grouping by policy if possible.')
        """
    ))

    # Outliers and winsorization guidance
    cells.append(md_cell("### Outliers & Winsorization Candidates"))
    cells.append(code_cell(
        """
        def pct_cap(s, lo=0.01, hi=0.99):
            return s.quantile(lo), s.quantile(hi)
        for c in [x for x in ['BonusMalus','Density','VehPower','VehAge'] if x in df.columns]:
            lo, hi = pct_cap(df[c].astype(float))
            print(f'{c}: suggested caps [{lo:.2f}, {hi:.2f}]')
        """
    ))

    # Baseline Poisson and NB (if available)
    cells.append(md_cell("### Baseline Frequency Model (Poisson GLM with offset)"))
    cells.append(code_cell(
        """
        features = []
        features += [c for c in ['DrivAge','VehAge','VehPower','log_density'] if c in df.columns]
        features += [c for c in ['Area','VehGas','Region'] if c in df.columns]
        print('Features considered:', features)

        # One-hot encode categoricals minimally
        X = pd.get_dummies(df[features], drop_first=True) if features else pd.DataFrame(index=df.index)
        y = df['ClaimNb'].values
        offset = np.log(df['Exposure'].values)

        try:
            import statsmodels.api as sm
            model = sm.GLM(y, sm.add_constant(X, has_constant='add'), family=sm.families.Poisson(), offset=offset)
            res = model.fit()
            print(res.summary())
            mu_hat = res.predict(sm.add_constant(X, has_constant='add'), offset=offset)
            dev = res.deviance; null_dev = res.null_deviance
            print(f'Deviance: {dev:.2f}, Null deviance: {null_dev:.2f}')
        except Exception as e:
            print('statsmodels not available or failed:', e)
            try:
                from sklearn.linear_model import PoissonRegressor
                m = PoissonRegressor(alpha=1e-6, fit_intercept=True, max_iter=1000)
                m.fit(X, y, sample_weight=None)
                mu_hat = np.exp(m.intercept_ + X.values @ m.coef_) * 1.0
                print('Sklearn PoissonRegressor fitted. Mean pred:', mu_hat.mean())
            except Exception as e2:
                print('Could not fit baseline Poisson model:', e2)
        """
    ))

    # Documentation notes
    cells.append(md_cell(
        "### Notes and Next Steps\n\n"
        "- Review rare level grouping and decide thresholds.\n\n"
        "- Consider capping/winsorization for skewed numeric features.\n\n"
        "- Validate monotonic trends for tariff variables (e.g., BonusMalus).\n\n"
        "- Use `log(Exposure)` as offset in all frequency models.\n"
    ))

    return cells


def append_cells_to_notebook(nb_path: Path):
    data = json.loads(nb_path.read_text(encoding='utf-8'))
    if 'cells' not in data or not isinstance(data['cells'], list):
        raise RuntimeError('Notebook structure unexpected: missing cells list')
    new_cells = build_cells()
    data['cells'].extend(new_cells)
    # Ensure nbformat keys exist
    data.setdefault('nbformat', 4)
    data.setdefault('nbformat_minor', 5)
    data.setdefault('metadata', data.get('metadata', {}))
    nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    nb_file = Path('notebooks/01_Preprocessing_EDA.ipynb')
    if not nb_file.exists():
        raise SystemExit(f'Notebook not found: {nb_file}')
    append_cells_to_notebook(nb_file)
    print('EDA enhancement cells appended to', nb_file)

