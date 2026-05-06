"""
电商 / 社媒舆情分析工具（Streamlit）
板块一：单文本情感极性分析
板块二：显式 vs 隐式情感表达对比
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import streamlit as st
import plotly.graph_objects as go
from huggingface_hub import close_session
from transformers import pipeline

MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

LABEL_ZH = {
    "positive": "积极 Positive",
    "neutral": "中性 Neutral",
    "negative": "消极 Negative",
}

GAUGE_COLORS = {
    "positive": {"bar": "#2e7d32", "steps": ["#e8f5e9", "#c8e6c9", "#a5d6a7"]},
    "neutral": {"bar": "#f9a825", "steps": ["#fff8e1", "#ffecb3", "#ffe082"]},
    "negative": {"bar": "#c62828", "steps": ["#ffebee", "#ffcdd2", "#ef9a9a"]},
}

PIE_COLORS = {
    "positive": "#6dbe9a",
    "neutral": "#9eb0e8",
    "negative": "#e5989b",
}

# 整体偏轻松：暖白渐变背景、略圆角与柔和阴影（与 Streamlit 默认组件协调）
LIGHT_APP_CSS = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #fffdf9 0%, #f6f3ee 55%, #f3f1ec 100%);
    }
    [data-testid="stHeader"] { background: rgba(255, 253, 249, 0.85); }
    .block-container { padding-top: 1.25rem; }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """
    Streamlit 会多次重载脚本；huggingface_hub 的全局 httpx 客户端可能在未清空全局引用的情况下
    已被关闭，从而触发 “Cannot send a request, as the client has been closed”。
    在加载 pipeline 前 reset 会话，并在该错误出现时重试。
    """
    last_err: BaseException | None = None
    for _ in range(3):
        close_session()
        try:
            return pipeline(
                "sentiment-analysis",
                model=MODEL_ID,
                top_k=3,
                truncation=True,
                max_length=512,
            )
        except RuntimeError as exc:
            last_err = exc
            if "client has been closed" in str(exc).lower():
                continue
            raise
    assert last_err is not None
    raise last_err


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def render_hf_hub_troubleshoot_expander() -> None:
    with st.expander("首次运行无法从 Hugging Face 下载模型？", expanded=False):
        st.markdown(
            """
            首次推理需从 `huggingface.co` 拉取权重。若出现 **连接超时**（Windows 上常见为 `WinError 10060`）或 **无法访问 Hub**，
            请先检查本机网络、防火墙或代理；也可在启动 Streamlit **之前** 使用镜像端点（示例，按需选用）：

            ```powershell
            $env:HF_ENDPOINT = "https://hf-mirror.com"
            streamlit run week10.py
            ```

            网络恢复后，若仍报错，可在 Streamlit 右上角菜单 **「Clear cache」** 后重试。
            """
        )


def load_pipeline_or_show_error() -> Any | None:
    try:
        return load_sentiment_pipeline()
    except OSError as err:
        st.error(
            "无法从 Hugging Face 下载或加载模型。常见原因：对 `huggingface.co` 不可达、连接超时或需代理。\n\n"
            "请展开上方 **「首次运行无法从 Hugging Face 下载模型？」** 查看镜像与环境变量说明；"
            "修复网络后使用菜单 **Clear cache** 再试。\n\n"
            f"`{type(err).__name__}`: {err}"
        )
        return None
    except RuntimeError as err:
        if "client has been closed" in str(err).lower():
            st.error(
                "Hugging Face Hub 的 HTTP 客户端状态异常（常见于页面快速重载或网络中断后）。\n\n"
                "请在右上角 **「Clear cache」** 清除缓存后重试；若仍失败，请完全停止并重新启动 Streamlit。"
                f"\n\n`RuntimeError`: {err}"
            )
        else:
            raise
        return None


def parse_pipeline_batch(batch: Any) -> list[dict[str, Any]]:
    if isinstance(batch, list) and batch and isinstance(batch[0], dict):
        scores_raw = batch
    elif isinstance(batch, list) and batch and isinstance(batch[0], list):
        scores_raw = batch[0]
    else:
        scores_raw = [batch] if isinstance(batch, dict) else []
    return sorted(scores_raw, key=lambda x: float(x["score"]), reverse=True)


def generate_mock_corpus() -> list[str]:
    """生成 10–15 条模拟电商评价，覆盖好评 / 中评 / 差评。"""
    n = random.randint(10, 15)
    pos_pool = [
        "回购第三次了，真心推荐，性价比拉满！",
        "开箱惊艳，做工细节比预期好太多。",
        "客服响应快，物流隔天到，体验满分。",
        "音质清澈，佩戴舒适，通勤必备。",
        "和详情页描述一致，甚至略超预期。",
        "活动价入手太香了，已经安利给同事。",
        "包装严实，赠品实用，会再光顾。",
    ]
    neu_pool = [
        "收到货品，外观正常，先用一段时间再评价。",
        "功能该有的都有，属于中规中矩的一款。",
        "价格还行，没有特别惊喜也没有明显槽点。",
        "说明书有点简略，自己摸索了一会儿才上手。",
        "颜色比图片略深一点，还能接受。",
        "快递比预计晚了一天，东西本身没问题。",
    ]
    neg_pool = [
        "用了一周外壳就开裂，质量堪忧。",
        "续航虚标严重，玩游戏两小时就没电了。",
        "客服推诿不解决，售后体验很差。",
        "异味很大，放了一周还是刺鼻。",
        "经常断连，稳定性完全不行。",
        "与宣传不符，感觉被套路了。",
        "退货流程繁琐，浪费时间。",
    ]
    if n < 3:
        n = 3
    n_pos = random.randint(1, n - 2)
    n_neu = random.randint(1, n - n_pos - 1)
    n_neg = n - n_pos - n_neu
    texts: list[str] = []
    texts.extend(random.choices(pos_pool, k=n_pos))
    texts.extend(random.choices(neu_pool, k=n_neu))
    texts.extend(random.choices(neg_pool, k=n_neg))
    random.shuffle(texts)
    return texts


def batch_top_predictions(pipe: Any, texts: list[str]) -> list[tuple[str, str, float]]:
    """对多条文本批量推理，返回每条 (原始 label, 规范化 label, 置信度)。"""
    if not texts:
        return []
    raw = pipe(texts)
    if not isinstance(raw, list):
        raw = [raw]
    out: list[tuple[str, str, float]] = []
    for item in raw:
        if isinstance(item, dict):
            scores_sorted = parse_pipeline_batch(item)
        elif isinstance(item, list) and item and isinstance(item[0], dict):
            scores_sorted = sorted(item, key=lambda x: float(x["score"]), reverse=True)
        else:
            continue
        if not scores_sorted:
            continue
        top = scores_sorted[0]
        lab = str(top["label"])
        key = normalize_label(lab)
        out.append((lab, key, float(top["score"])))
    return out


def build_sentiment_pie_figure(counts: dict[str, int], total: int) -> go.Figure:
    order = ("positive", "neutral", "negative")
    labels = [LABEL_ZH[k] for k in order if counts.get(k, 0) > 0]
    values = [counts[k] for k in order if counts.get(k, 0) > 0]
    colors = [PIE_COLORS[k] for k in order if counts.get(k, 0) > 0]
    if not values:
        labels = ["无数据"]
        values = [1]
        colors = ["#c4bfb6"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                sort=False,
                direction="clockwise",
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                texttemplate="<b>%{label}</b><br>%{percent}",
                textposition="outside",
                textfont=dict(size=13, color="#3d3a36"),
                insidetextfont=dict(color="#2c2825", size=14),
                hovertemplate="%{label}<br>条数: %{value}<br>占比: %{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text=f"口碑结构 · N = {total}",
            font=dict(size=18, color="#4a453f", family="Microsoft YaHei, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            x=0.5,
            xanchor="center",
            font=dict(color="#5c5850", size=12),
        ),
        margin=dict(t=56, b=80, l=24, r=24),
        annotations=[
            dict(
                text="口碑<br>占比",
                x=0.5,
                y=0.5,
                font=dict(size=14, color="#8a847a"),
                showarrow=False,
            )
        ],
    )
    return fig


def render_tab_batch_dashboard() -> None:
    st.subheader("批量情感分析看板")

    with st.container(border=True):
        st.caption("阅读说明")
        st.markdown(
            """
            将 **单条评论的情感分类** 扩展到 **一批评论（小型语料库）**，即可进入**意见挖掘（Opinion Mining）**的雏形：
            不再只看一句话的正负，而是看**整体分布**——好评是否占主导、负面是否形成「长尾」、中性是否暗示「无感/观望」。
            下方按钮会随机生成 **10–15 条** 含好、中、差评的模拟数据，并一次性送入同一模型批量推理。
            """
        )
        with st.expander("从「单句分析」到「大规模语料」：如何支撑商业决策？", expanded=False):
            st.markdown(
                """
                1. **产品改进（VoC）**：批量统计后，可结合**主题模型/关键词**（本演示以情感聚合为主）定位「负面是否集中在质量、物流、售后」等维度，为迭代排期提供依据。
                2. **危机预警**：负面占比或**高置信消极**条数在短期内陡升，可作为舆情告警信号，触发公关或客服预案；需配合时间序列与渠道分层（本模块为静态快照演示）。
                3. **营销与定位**：积极占比高但中性偏高，可能意味着「认可但不兴奋」——适合挖掘差异化卖点；反之消极抬头需先修复体验再投放。
                4. **方法局限**：模拟数据量小、模型为轻量级多语言蒸馏，**宏观比例仅供参考**；真实业务需更大样本、抽样偏差控制、以及隐式讽刺等难例的人工审计。

                **观察建议**：多次点击生成，看随机批次下三类计数与饼图如何变化——体会 **「分布」比「单点预测」** 更接近管理层报表语言。
                """
            )
        render_hf_hub_troubleshoot_expander()

    if st.button("生成测试舆情数据", type="primary", use_container_width=True, key="btn_gen_batch"):
        st.session_state["m3_reviews"] = generate_mock_corpus()
        st.session_state["m3_analyzed"] = False

    reviews = st.session_state.get("m3_reviews")
    if not reviews:
        st.info("点击 **「生成测试舆情数据」** 以生成模拟评论并运行批量情感分析。")
        return

    with st.expander("当前批次原始评论", expanded=False):
        for i, line in enumerate(reviews, start=1):
            st.markdown(f"{i}. {line}")

    if not st.session_state.get("m3_analyzed", False):
        with st.spinner("正在批量推理（首次会加载模型）…"):
            pipe = load_pipeline_or_show_error()
            if pipe is None:
                return
            preds = batch_top_predictions(pipe, reviews)
        if len(preds) != len(reviews):
            st.error("批量推理结果条数与输入不一致，请重试或 Clear cache。")
            return
        st.session_state["m3_preds"] = preds
        st.session_state["m3_analyzed"] = True

    preds = st.session_state.get("m3_preds") or []
    keys = [p[1] for p in preds]
    ctr = Counter(keys)
    known = {"positive", "neutral", "negative"}
    stray = sum(c for lab, c in ctr.items() if lab not in known)
    counts = {
        "positive": int(ctr.get("positive", 0)),
        "neutral": int(ctr.get("neutral", 0)) + stray,
        "negative": int(ctr.get("negative", 0)),
    }
    if stray:
        st.caption(f"注：有 {stray} 条标签落在预期三分类之外，已并入「中性」计数以便展示。")
    total = len(reviews)

    with st.container(border=True):
        st.markdown("##### 分析结果")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Positive · 积极", counts["positive"])
        mc2.metric("Neutral · 中性", counts["neutral"])
        mc3.metric("Negative · 消极", counts["negative"])
        pie = build_sentiment_pie_figure(counts, total)
        st.plotly_chart(pie, use_container_width=True, key="tab3_pie")
        st.markdown("**逐条预测明细**（原文 · 预测标签 · 模型置信度）")
        rows = []
        for text, (_raw, k, conf) in zip(reviews, preds):
            rows.append(
                {
                    "评论原文": text,
                    "预测": LABEL_ZH.get(k, k),
                    "置信度": f"{conf:.2%}",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


def build_gauge_figure(confidence_pct: float, pred_key: str) -> go.Figure:
    palette = GAUGE_COLORS.get(pred_key, GAUGE_COLORS["neutral"])
    steps = [
        {"range": [0, 33.33], "color": palette["steps"][0]},
        {"range": [33.33, 66.66], "color": palette["steps"][1]},
        {"range": [66.66, 100], "color": palette["steps"][2]},
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence_pct, 2),
            number={"suffix": " %", "font": {"size": 36}},
            title={"text": "预测类别置信度<br><sub>Confidence（对当前预测标签的概率）</sub>"},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#444"},
                "bar": {"color": palette["bar"]},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#ccc",
                "steps": steps,
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.8,
                    "value": confidence_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=24, r=24, t=48, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Microsoft YaHei, sans-serif"},
    )
    return fig


def render_sentiment_results(scores_sorted: list[dict[str, Any]], *, key_prefix: str) -> None:
    top = scores_sorted[0]
    pred_key = normalize_label(top["label"])
    conf = float(top["score"])
    conf_pct = conf * 100.0
    label_display = LABEL_ZH.get(pred_key, top["label"])

    m1, m2 = st.columns(2)
    with m1:
        st.metric("预测情感标签", label_display)
    with m2:
        st.metric("该标签置信度（概率）", f"{conf:.2%}")

    st.plotly_chart(
        build_gauge_figure(conf_pct, pred_key),
        use_container_width=True,
        key=f"{key_prefix}_gauge",
    )

    st.markdown("**各类别概率一览**")
    names = [LABEL_ZH.get(normalize_label(s["label"]), s["label"]) for s in scores_sorted]
    vals = [round(float(s["score"]) * 100, 2) for s in scores_sorted]
    dist_fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=["#1976d2" if i == 0 else "#90caf9" for i in range(len(names))],
            text=[f"{v}%" for v in vals],
            textposition="auto",
        )
    )
    dist_fig.update_layout(
        xaxis_title="概率（%）",
        yaxis_title="",
        height=220,
        margin=dict(l=8, r=8, t=32, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(dist_fig, use_container_width=True, key=f"{key_prefix}_bar")


def render_tab_single_text() -> None:
    st.subheader("单文本情感极性分析")
    with st.container(border=True):
        st.caption("阅读说明")
        st.markdown(
            "使用轻量多语言模型 `lxyuan/distilbert-base-multilingual-cased-sentiments-student`，"
            "输出三分类：积极 / 中性 / 消极。"
        )
        with st.expander("为什么在工程里要看「置信度 / 概率」，而不只看分类结果？", expanded=False):
            st.markdown(
                """
                1. **决策与阈值**：业务上常把「积极但仅 52%」与「积极且 98%」区别对待；没有概率就无法做分级策略（如仅对高置信消极进线客服）。
                2. **校准与可解释**：Softmax 输出反映模型在当前标签空间下的相对把握；低置信往往对应**措辞模糊、反讽、混合情感**或**分布外文本**，需要人工复核而非自动执行强动作。
                3. **错误分析**：若模型常给出错误标签却伴随**异常高的置信度**，说明可能存在过拟合、数据偏差或对抗样本，需要重点排查。
                4. **监控与漂移**：线上可统计置信度分布；若整体置信度持续下降或某类文本置信异常，可能提示数据分布变化或模型退化。

                **小实验建议**：对比「五星好评式短句」与「整体满意但夹带一句抱怨」的长评——后者常出现**类别不变但置信度下降**或**中性概率上升**，这正是概率信息的价值。
                """
            )
        render_hf_hub_troubleshoot_expander()

    default_hint = (
        "在此输入一段中文商品评论。\n"
        "可刻意输入「非常明显的好评」或「非常明显的差评」，观察仪表盘上置信度的变化。"
    )
    text = st.text_area("评论文本", height=160, placeholder=default_hint)

    col_run, _ = st.columns([1, 3])
    with col_run:
        analyze = st.button("分析情感", type="primary", use_container_width=True)

    if not analyze:
        return

    content = (text or "").strip()
    if not content:
        st.warning("请先输入一段评论文本。")
        return

    with st.spinner("正在加载模型（首次运行会下载权重）并推理…"):
        pipe = load_pipeline_or_show_error()
        if pipe is None:
            return
        batch = pipe(content)

    scores_sorted = parse_pipeline_batch(batch)
    if not scores_sorted:
        st.error("模型未返回有效结果，请稍后重试。")
        return

    with st.container(border=True):
        st.markdown("##### 分析结果")
        render_sentiment_results(scores_sorted, key_prefix="tab1")


def render_tab_explicit_implicit() -> None:
    st.subheader("显式 vs 隐式情感表达（对比实验）")
    with st.container(border=True):
        st.caption("阅读说明")
        st.markdown(
            "同一模型、两次推理：观察「褒贬词明显」与「客观陈述但含态度」时，标签与置信度如何变化。"
        )
        st.markdown(
            """
            **什么是「显式情感」？**  
            文本里带有**明显的褒贬或情绪词**，情感倾向直接写在字面上。例如：「太棒了」「非常满意」「烂透了」「坑爹」。模型往往更容易从词汇表面模式判断极性。

            **什么是「隐式情感」？**  
            表面上是**事实、现象或客观描述**，不一定出现「好/坏」字眼，但结合常识能推断说话人的态度。例如：「手机玩游戏半小时就没电了」——未说「差」，却常表达续航不满；又如「包装皱了，里面还好」可能混合中性事实与轻微负面。  
            这类句子考验模型是否具备**常识推理、领域知识**以及是否见过足够多样的标注样本。
            """
        )
        with st.expander("小型深度学习模型，能「听懂」隐式负面吗？", expanded=False):
            st.markdown(
                """
                1. **训练目标**：多数情感分类器学习的是「字面线索 ↔ 标签」的统计关联；**显式情感词**是最强信号，隐式态度往往更依赖上下文与常识，小模型容量有限时容易「抓表面」。
                2. **中性陷阱**：客观陈述句在语法上像「说明文」，模型若缺乏深层语义，可能给出 **Neutral 偏高**——并非「读不懂汉字」，而是**未把事实与负面后果绑定**。
                3. **置信度的信号作用**：隐式句若被判为负面但**概率仅略高于中性**，通常说明模型「不太确定」；若判错却**置信度很高**，则提示数据偏差或过拟合某些表面模式（值得记录为 bad case）。
                4. **改进方向（工程视角）**：领域数据微调、Aspect-Based 情感、引入更大模型或知识增强、以及对低置信/隐式类文本做**人机协同审核**。

                **建议对比**：左框写带强烈褒义词的短评，右框写你举的「半小时没电」类客观描述——看 **预测标签是否一致**、**消极/中性概率差**、以及 **置信度是否明显下降**。
                """
            )
        render_hf_hub_troubleshoot_expander()

    c1, c2 = st.columns(2)
    with c1:
        explicit = st.text_area(
            "显式情感评价",
            height=140,
            placeholder="例：太棒了，音质和做工都超出预期！",
            key="explicit_input",
        )
    with c2:
        implicit = st.text_area(
            "隐式客观描述",
            height=140,
            placeholder="例：手机玩游戏半小时就没电了。",
            key="implicit_input",
        )

    if st.button("对比分析两条文本", type="primary", use_container_width=True):
        ex = (explicit or "").strip()
        im = (implicit or "").strip()
        if not ex or not im:
            st.warning("请同时在两个输入框中填写内容，再进行对比。")
            return

        with st.spinner("正在推理（首次运行会加载模型）…"):
            pipe = load_pipeline_or_show_error()
            if pipe is None:
                return
            batch_ex = pipe(ex)
            batch_im = pipe(im)

        scores_ex = parse_pipeline_batch(batch_ex)
        scores_im = parse_pipeline_batch(batch_im)
        if not scores_ex or not scores_im:
            st.error("模型未返回有效结果，请稍后重试。")
            return

        st.markdown("---")
        with st.container(border=True):
            st.markdown("##### 分析结果")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("###### 显式情感评价 · 模型输出")
                render_sentiment_results(scores_ex, key_prefix="tab2_explicit")
            with rc2:
                st.markdown("###### 隐式客观描述 · 模型输出")
                render_sentiment_results(scores_im, key_prefix="tab2_implicit")


def main() -> None:
    st.set_page_config(
        page_title="舆情分析工具",
        page_icon="📊",
        layout="wide",
    )
    st.markdown(LIGHT_APP_CSS, unsafe_allow_html=True)
    st.title("电商 / 社交媒体 · 舆情分析工具")

    tab1, tab2, tab3 = st.tabs(
        [
            "① 单文本情感分析",
            "② 显式 / 隐式对比",
            "③ 批量舆情看板",
        ]
    )

    with tab1:
        render_tab_single_text()

    with tab2:
        render_tab_explicit_implicit()

    with tab3:
        render_tab_batch_dashboard()


if __name__ == "__main__":
    main()
