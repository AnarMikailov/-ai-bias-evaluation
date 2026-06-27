import gc
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from bias_analysis import analyze_from_csvs

MAX_ROWS = 15_000

st.set_page_config(
    page_title="Alqoritmik Qərəzliyin Qiymətləndirilməsi",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .red-header   { color: #c62828; font-size: 1.4rem; font-weight: 700; }
    .green-header { color: #2e7d32; font-size: 1.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Başlıq ──────────────────────────────────────────────────────────────────
st.title("⚖️ Süni İntelektin Etik Məsələləri")
st.subheader("Alqoritmik Qərəzliyin Qiymətləndirilməsi")
st.markdown("---")

def _dataset_info(df, filename=""):
    """Yuklenen CSV-e gore dataset melumatlarini qaytarir."""
    if df is None:
        return None
    is_compas = 'yeniden_cinayat' in df.columns
    rows = len(df)
    cols = len(df.columns)

    if is_compas:
        protected    = 'race (Irq)'
        label        = 'yeniden_cinayat (Yenidən cinayət etmə)'
        priv_grp     = 'Caucasian'
        unpriv_grp   = 'African-American'
        task         = 'Məhkumun 2 il ərzində yenidən cinayət edib-etməyəcəyini proqnozlaşdır'
        source       = 'ProPublica (2016) — "Machine Bias" araşdırması'
        bias_type    = 'İrqi qərəz — AA məhkumlar Caucasian məhkumlara nisbət 2x yüksək risk etiketlənir'
        priv_rate    = f"{(df[df['race']=='Caucasian']['yeniden_cinayat'].mean()*100):.1f}%"
        unpriv_rate  = f"{(df[df['race']=='African-American']['yeniden_cinayat'].mean()*100):.1f}%"
        rate_label   = 'Yeniden cinayat nisbeti'
    else:
        protected    = 'sex (Cinsiyyət)'
        label        = 'income (Gəlir > 50K$)'
        priv_grp     = 'Kişi (Male)'
        unpriv_grp   = 'Qadın (Female)'
        task         = 'Şəxsin illik gəlirinin 50.000$-dan çox olub-olmadığını proqnozlaşdır'
        source       = 'UCI ML Repository — Adult Census Income (1994 ABŞ sayıyaalma məlumatı)'
        bias_type    = 'Gender qərəzi — qadın işçilər eyni işə görə kişi işçilərdən az qazanır'
        priv_rate    = f"{(df[df['sex']=='Male']['income'].str.rstrip('.').eq('>50K').mean()*100):.1f}%"
        unpriv_rate  = f"{(df[df['sex']=='Female']['income'].str.rstrip('.').eq('>50K').mean()*100):.1f}%"
        rate_label   = '>50K$ qazananlar'

    return {
        'rows': rows, 'cols': cols,
        'protected': protected, 'label': label,
        'priv_grp': priv_grp, 'unpriv_grp': unpriv_grp,
        'task': task, 'source': source, 'bias_type': bias_type,
        'priv_rate': priv_rate, 'unpriv_rate': unpriv_rate,
        'rate_label': rate_label, 'is_compas': is_compas
    }

# "Layihe haqqinda" — yuklenmis dataya gore dinamik
_uploaded_b = st.session_state.get('preview_b')
info = _dataset_info(_uploaded_b)

with st.expander("ℹ️ Layihə haqqında", expanded=(info is None)):
    if info is None:
        st.info("CSV faylları yüklədikdən sonra bu bölmə avtomatik yenilənəcək.")
        st.markdown("""
        | | |
        |---|---|
        | **Model** | Logistic Regression |
        | **Ölçü aləti** | IBM AI Fairness 360 |
        | **Dəstəklənən datasetlər** | Adult Income (UCI) · COMPAS Recidivism (ProPublica) |
        """)
    else:
        st.markdown(f"""
        | | |
        |---|---|
        | **Mənbə** | {info['source']} |
        | **Tapşırıq** | {info['task']} |
        | **Sətir sayı** | {info['rows']:,} sətir · {info['cols']} sütun |
        | **Qorunan atribut** | {info['protected']} |
        | **Hədəf dəyişən** | {info['label']} |
        | **İmtiyazlı qrup** | {info['priv_grp']} — {info['rate_label']}: **{info['priv_rate']}** |
        | **Qorunan qrup** | {info['unpriv_grp']} — {info['rate_label']}: **{info['unpriv_rate']}** |
        | **Qərəz növü** | {info['bias_type']} |
        | **Model** | Logistic Regression |
        | **Ölçü aləti** | IBM AI Fairness 360 |
        """)

# ── Fayl yükləmə ────────────────────────────────────────────────────────────
st.markdown("## 📂 Data Yüklə")

col_up1, col_up2 = st.columns(2, gap="large")

def _preview_stats(df):
    """Dataset tipine gore qrup statistikasini qaytarir."""
    if 'yeniden_cinayat' in df.columns:
        r1 = (df[df['race'] == 'Caucasian']['yeniden_cinayat']).mean()
        r2 = (df[df['race'] == 'African-American']['yeniden_cinayat']).mean()
        return f"Caucasian cinayət: **{r1:.1%}** | African-American cinayət: **{r2:.1%}**"
    else:
        r1 = (df[df['sex'] == 'Male']['income']   == '>50K').mean()
        r2 = (df[df['sex'] == 'Female']['income'] == '>50K').mean()
        return f"Kişi >50K: **{r1:.1%}** | Qadın >50K: **{r2:.1%}**"

with col_up1:
    st.markdown("### 🔴 Qərəzli Data")
    biased_file = st.file_uploader(
        "CSV faylını seç",
        type=['csv'],
        key='biased'
    )
    if biased_file:
        biased_df = pd.read_csv(biased_file)
        st.session_state['preview_b'] = biased_df
        st.success(f"✅ {len(biased_df):,} sətir yükləndi")
        st.caption(_preview_stats(biased_df))
        with st.expander("İlk 5 sətirə bax"):
            st.dataframe(biased_df.head(), use_container_width=True)

with col_up2:
    st.markdown("### 🟢 Qərəzsiz Data")
    unbiased_file = st.file_uploader(
        "CSV faylını seç",
        type=['csv'],
        key='unbiased'
    )
    if unbiased_file:
        unbiased_df = pd.read_csv(unbiased_file)
        st.session_state['preview_u'] = unbiased_df
        st.success(f"✅ {len(unbiased_df):,} sətir yükləndi")
        st.caption(_preview_stats(unbiased_df))
        with st.expander("İlk 5 sətirə bax"):
            st.dataframe(unbiased_df.head(), use_container_width=True)

# ── Data Vizualizasiyası (upload sonrası, analiz əvvəli) ───────────────────
_prev_b = st.session_state.get('preview_b')
_prev_u = st.session_state.get('preview_u')

if _prev_b is not None and _prev_u is not None:
    st.markdown("---")
    st.markdown("## 📊 Data Vizualizasiyası")
    st.caption("Modeli işlətmədən əvvəl — xam datanın qrup bölgüsü")

    _is_compas = 'yeniden_cinayat' in _prev_b.columns

    if _is_compas:
        _grp_col = 'race'
        _out_col = 'yeniden_cinayat'
        _g1, _g2 = 'Caucasian', 'African-American'
        _rate_label = 'Yeniden cinayat (%)'
        _chart_title = 'İrq üzrə Cinayətkarlıq Nisbəti'
    else:
        _grp_col = 'sex'
        _out_col = 'income'
        _g1, _g2 = 'Male', 'Female'
        _rate_label = '>50K$ qazananlar (%)'
        _chart_title = 'Cinsiyyət üzrə Gəlir Nisbəti'

    def _get_pos_rate(df, grp, is_compas):
        sub = df[df[_grp_col] == grp]
        if is_compas:
            return sub[_out_col].mean() * 100
        return (sub[_out_col].str.rstrip('.') == '>50K').mean() * 100

    _br1 = _get_pos_rate(_prev_b, _g1, _is_compas)
    _br2 = _get_pos_rate(_prev_b, _g2, _is_compas)
    _ur1 = _get_pos_rate(_prev_u, _g1, _is_compas)
    _ur2 = _get_pos_rate(_prev_u, _g2, _is_compas)

    dv1, dv2 = st.columns(2, gap="large")

    with dv1:
        st.markdown(f"### {_chart_title}")
        _rate_df = pd.DataFrame({
            'Qrup': [_g1, _g2, _g1, _g2],
            'Dataset': ['🔴 Qərəzli', '🔴 Qərəzli', '🟢 Qərəzsiz', '🟢 Qərəzsiz'],
            _rate_label: [_br1, _br2, _ur1, _ur2]
        })
        _fig_r = px.bar(
            _rate_df, x='Qrup', y=_rate_label, color='Dataset', barmode='group',
            color_discrete_map={'🔴 Qərəzli': '#ef5350', '🟢 Qərəzsiz': '#66bb6a'},
            text_auto='.1f'
        )
        _fig_r.update_traces(textposition='outside')
        _fig_r.update_layout(height=380, legend_title='')
        st.plotly_chart(_fig_r, use_container_width=True)

    with dv2:
        st.markdown("### Qrup Həcmləri")
        _bn1 = int((_prev_b[_grp_col] == _g1).sum())
        _bn2 = int((_prev_b[_grp_col] == _g2).sum())
        _un1 = int((_prev_u[_grp_col] == _g1).sum())
        _un2 = int((_prev_u[_grp_col] == _g2).sum())
        _cnt_df = pd.DataFrame({
            'Qrup': [_g1, _g2, _g1, _g2],
            'Dataset': ['🔴 Qərəzli', '🔴 Qərəzli', '🟢 Qərəzsiz', '🟢 Qərəzsiz'],
            'Sətir sayı': [_bn1, _bn2, _un1, _un2]
        })
        _fig_c = px.bar(
            _cnt_df, x='Qrup', y='Sətir sayı', color='Dataset', barmode='group',
            color_discrete_map={'🔴 Qərəzli': '#ef5350', '🟢 Qərəzsiz': '#66bb6a'},
            text_auto=',.0f'
        )
        _fig_c.update_traces(textposition='outside')
        _fig_c.update_layout(height=380, legend_title='')
        st.plotly_chart(_fig_c, use_container_width=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric(f"{_g1} (Qərəzli)", f"{_br1:.1f}%")
    mc2.metric(f"{_g2} (Qərəzli)", f"{_br2:.1f}%", delta=f"{_br2-_br1:+.1f} pp")
    mc3.metric(f"{_g1} (Qərəzsiz)", f"{_ur1:.1f}%")
    mc4.metric(f"{_g2} (Qərəzsiz)", f"{_ur2:.1f}%", delta=f"{_ur2-_ur1:+.1f} pp")

st.markdown("---")

# ── Analiz düyməsi ──────────────────────────────────────────────────────────
if biased_file and unbiased_file:
    if st.button("▶  Analizi Başlat", type="primary", use_container_width=True):
        with st.spinner("Modellər train edilir..."):
            biased_file.seek(0)
            unbiased_file.seek(0)
            biased_df   = pd.read_csv(biased_file)
            unbiased_df = pd.read_csv(unbiased_file)
            if len(biased_df) > MAX_ROWS:
                biased_df = biased_df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
            if len(unbiased_df) > MAX_ROWS:
                unbiased_df = unbiased_df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
            b, u = analyze_from_csvs(biased_df, unbiased_df)
            del biased_df, unbiased_df
            gc.collect()
            st.session_state['b'] = b
            st.session_state['u'] = u
        st.success("✅ Analiz tamamlandı!")
else:
    st.info("Hər iki CSV faylını yükləyin, sonra analiz başlayacaq.")

# ── Nəticələr ───────────────────────────────────────────────────────────────
if 'b' in st.session_state:
    b = st.session_state['b']
    u = st.session_state['u']

    st.markdown("## Nəticələrin Müqayisəsi")

    def di_label(v):
        return "⚠️ Qərəzli (< 0.8)" if v < 0.8 else "✅ Ədalətli (≥ 0.8)"

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="red-header">🔴 Qərəzli Model</p>', unsafe_allow_html=True)
        st.caption("Orijinal, emal edilməmiş data ilə train edilib")
        st.metric("Dəqiqlik", f"{b['accuracy']:.1%}")
        st.metric("Disparate Impact", f"{b['disparate_impact']:.3f}",
                  delta=di_label(b['disparate_impact']), delta_color="off")
        st.metric("Statistical Parity Diff.", f"{b['statistical_parity']:.3f}")
        st.metric("Equal Opportunity Diff.", f"{b['equal_opportunity']:.3f}")
        st.metric("Avg. Odds Diff.", f"{b['avg_odds_diff']:.3f}")
        pp_b = b.get('predictive_parity', float('nan'))
        if not (pp_b != pp_b):  # nan check
            st.metric("Predictive Parity Diff.", f"{pp_b:.3f}")

    with col2:
        st.markdown('<p class="green-header">🟢 Qərəzsiz Model</p>', unsafe_allow_html=True)
        st.caption("Balanslaşdırılmış data ilə train edilib")
        st.metric("Dəqiqlik", f"{u['accuracy']:.1%}")
        st.metric("Disparate Impact", f"{u['disparate_impact']:.3f}",
                  delta=di_label(u['disparate_impact']), delta_color="off")
        st.metric("Statistical Parity Diff.", f"{u['statistical_parity']:.3f}")
        st.metric("Equal Opportunity Diff.", f"{u['equal_opportunity']:.3f}")
        st.metric("Avg. Odds Diff.", f"{u['avg_odds_diff']:.3f}")
        pp_u = u.get('predictive_parity', float('nan'))
        if not (pp_u != pp_u):
            st.metric("Predictive Parity Diff.", f"{pp_u:.3f}")

    st.markdown("---")

    # ── Bar chart ──────────────────────────────────────────────────────────
    cfg = b.get('config', {})
    if cfg.get('protected') == 'race':
        _bar_title = "Caucasian vs African-American — Müsbət Proqnoz Nisbəti (%)"
        _bar_cap   = "Modelin hər iki irq qrupu üçün 'yenidən cinayət edəcək' deməsi ehtimalı"
    else:
        _bar_title = "Kişi vs Qadın — Müsbət Proqnoz Nisbəti (%)"
        _bar_cap   = "Modelin hər iki qrup üçün '50K$-dan çox qazanır' deməsi ehtimalı"
    st.markdown(f"## {_bar_title}")
    st.caption(_bar_cap)
    if cfg.get('protected') == 'race':
        priv_label   = 'Caucasian'
        unpriv_label = 'African-American'
    else:
        priv_label   = 'Kişi'
        unpriv_label = 'Qadın'

    chart_df = pd.DataFrame({
        'Qrup':  [priv_label, unpriv_label, priv_label, unpriv_label],
        'Model': ['Qərəzli', 'Qərəzli', 'Qərəzsiz', 'Qərəzsiz'],
        'Nisbət (%)': [
            b['privileged_rate'] * 100, b['unprivileged_rate'] * 100,
            u['privileged_rate'] * 100, u['unprivileged_rate'] * 100,
        ]
    })

    fig = px.bar(
        chart_df, x='Qrup', y='Nisbət (%)', color='Model', barmode='group',
        color_discrete_map={'Qərəzli': '#ef5350', 'Qərəzsiz': '#66bb6a'},
        text_auto='.1f'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Disparate Impact chart ─────────────────────────────────────────────
    st.markdown("## Disparate Impact Müqayisəsi")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=['Qərəzli Model', 'Qərəzsiz Model'],
        y=[b['disparate_impact'], u['disparate_impact']],
        marker_color=['#ef5350', '#66bb6a'],
        text=[f"{b['disparate_impact']:.3f}", f"{u['disparate_impact']:.3f}"],
        textposition='outside'
    ))
    fig2.add_hline(y=0.8, line_dash='dash', line_color='orange',
                   annotation_text="0.8 — Qəbul edilən hüdud")
    fig2.add_hline(y=1.0, line_dash='dot', line_color='green',
                   annotation_text="1.0 — İdeal ədalət")
    fig2.update_layout(height=380, yaxis_title="Disparate Impact", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── İzahat ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Metriklər Nə Deməkdir?")

    if cfg.get('protected') == 'race':
        _priv_lbl   = 'Caucasian'
        _unpriv_lbl = 'African-American'
        _di_desc    = 'African-American məhkumlar Caucasian məhkumlara nisbətən ədalətsiz mənfi proqnoz alır.'
    else:
        _priv_lbl   = 'Kişi'
        _unpriv_lbl = 'Qadın'
        _di_desc    = 'Qadınlar kişilərə nisbətən ədalətsiz mənfi proqnoz alır.'

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.error("**DI < 0.8** → Qərəzli")
        st.write(_di_desc)
    with mc2:
        st.warning("**DI 0.8–1.0** → Qəbul edilən hüdud")
        st.write("ABŞ Equal Employment qanununa görə minimum tələb.")
    with mc3:
        st.success("**DI ≈ 1.0** → Ədalətli")
        st.write("Hər iki qrup eyni ehtimalla müsbət proqnoz alır.")

    st.info(
        f"**Disparate Impact:** DI = P(müsbət | {_unpriv_lbl}) ÷ P(müsbət | {_priv_lbl})\n\n"
        f"**Statistical Parity:** P(müsbət | {_unpriv_lbl}) − P(müsbət | {_priv_lbl}) → 0-a yaxın olmalıdır\n\n"
        f"**Equal Opportunity:** TPR({_unpriv_lbl}) − TPR({_priv_lbl}) → 0-a yaxın olmalıdır"
    )

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔍 SHAP — Model Qərarlarının İzahı (XAI)")
    st.caption(
        "SHAP (SHapley Additive exPlanations) hər xüsusiyyətin modelin qərarına "
        "nə qədər təsir etdiyini göstərir. Uzun sütun = güclü təsir."
    )

    b_shap = b.get('shap', {})
    u_shap = u.get('shap', {})

    if b_shap.get('ok') and u_shap.get('ok'):
        method = b_shap.get('method', '')
        if method:
            st.caption(f"Metod: **{method}**")
        sc1, sc2 = st.columns(2, gap="large")

        with sc1:
            st.markdown('<p class="red-header">🔴 Qərəzli Model</p>', unsafe_allow_html=True)
            fig_b = go.Figure(go.Bar(
                x=b_shap['importance'][::-1],
                y=b_shap['features'][::-1],
                orientation='h',
                marker_color='#ef5350',
                text=[f"{v:.4f}" for v in b_shap['importance'][::-1]],
                textposition='outside'
            ))
            fig_b.update_layout(
                height=420, margin=dict(l=10, r=60, t=10, b=10),
                xaxis_title="Orta |SHAP| dəyəri",
                yaxis=dict(tickfont=dict(size=11))
            )
            st.plotly_chart(fig_b, use_container_width=True)

        with sc2:
            st.markdown('<p class="green-header">🟢 Qərəzsiz Model</p>', unsafe_allow_html=True)
            fig_u = go.Figure(go.Bar(
                x=u_shap['importance'][::-1],
                y=u_shap['features'][::-1],
                orientation='h',
                marker_color='#66bb6a',
                text=[f"{v:.4f}" for v in u_shap['importance'][::-1]],
                textposition='outside'
            ))
            fig_u.update_layout(
                height=420, margin=dict(l=10, r=60, t=10, b=10),
                xaxis_title="Orta |SHAP| dəyəri",
                yaxis=dict(tickfont=dict(size=11))
            )
            st.plotly_chart(fig_u, use_container_width=True)

        st.info(
            "**Necə oxumaq olar:** Hər sütun bir xüsusiyyətin modelin qərarına "
            "orta təsirini göstərir. İki modelin ən vacib xüsusiyyətlərini müqayisə "
            "etmək — qərəzin haradan gəldiyini anlamağa kömək edir."
        )
    else:
        err = b_shap.get('error', 'Naməlum xəta')
        st.warning(f"SHAP hesablanmadı: {err}")

    # ── Avtomatik nəticə xülasəsi ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📝 Nəticə Xülasəsi")

    _di_b = b['disparate_impact']
    _di_u = u['disparate_impact']
    _di_improvement = abs(_di_u - _di_b)
    _sp_b = abs(b['statistical_parity'])
    _sp_u = abs(u['statistical_parity'])

    _di_verdict_b = "qərəzli (0.8 həddindən aşağı)" if _di_b < 0.8 else "ədalətli"
    _di_verdict_u = "ədalətli (1.0-a yaxın)" if 0.8 <= _di_u <= 1.2 else "hələ qərəzli"

    if cfg.get('protected') == 'race':
        _grp_priv   = "Caucasian"
        _grp_unpriv = "African-American"
        _context    = "cinayətkarlıq proqnozu"
    else:
        _grp_priv   = "kişi"
        _grp_unpriv = "qadın"
        _context    = "gəlir proqnozu"

    _acc_diff = (u['accuracy'] - b['accuracy']) * 100

    _summary = f"""
**Qərəzli model** orijinal data üzərində train edilib və {_context} tapşırığında \
Disparate Impact **{_di_b:.3f}** göstərib — bu {_di_verdict_b} nəticədir. \
Başqa sözlə, model {_grp_unpriv} qrupuna {_grp_priv} qrupuna nisbətən əhəmiyyətli dərəcədə \
az müsbət proqnoz verib (Statistical Parity fərqi: **{b['statistical_parity']:.3f}**).

**Qərəzsiz model** stratified undersampling ilə balanslaşdırılmış data üzərində train edilib. \
Disparate Impact **{_di_u:.3f}**-ə çatıb — {_di_verdict_u}. \
Statistical Parity fərqi **{u['statistical_parity']:.3f}**-ə enib, \
yəni hər iki qrup demək olar ki, eyni nisbətdə müsbət proqnoz alıb.

**Nəticə:** Qərəzlilik azaldılması metodu Disparate Impact-i **{_di_improvement:.3f}** vahid yaxşılaşdırıb. \
Model dəqiqliyi isə {"artmış" if _acc_diff >= 0 else "cüzi azalmış"} \
(**{_acc_diff:+.1f}%**{"" if abs(_acc_diff) > 0.5 else " — praktiki olaraq dəyişməyib"}), \
bu da göstərir ki, ədalətliliyi artırmaq dəqiqlikdən əhəmiyyətli güzəşt tələb etmir.
"""
    st.success(_summary)
