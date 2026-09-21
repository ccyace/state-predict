#!/usr/bin/env python3
"""Export denoise-formula-aware channel scaling derivation to Word (.docx)."""

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm
from docx.oxml.ns import qn


def set_doc_style(doc: Document):
    section = doc.sections[0]
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(16)
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")


def add_para(doc, text, bold=False, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    run = p.add_run(text)
    run.bold = bold
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    return p


def add_eq(doc, text, number=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Cambria Math"
    run.font.size = Pt(11)
    if number is not None:
        run2 = p.add_run(f"    ({number})")
        run2.font.name = "Times New Roman"
        run2.font.size = Pt(11)


def add_bullet(doc, text):
    p = doc.add_paragraph(text, style="List Bullet")
    for run in p.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def build_document(output_path: str):
    doc = Document()
    set_doc_style(doc)

    add_title(doc, "基于 DDIM 去噪公式的通道缩放因子解析推导")
    add_para(doc, "（Denoise-Formula-Aware Channel Scaling for Diffusion UNet Quantization）", indent=False)
    doc.add_paragraph()

    add_para(
        doc,
        "摘要：本文从 DDIM 确定性去噪更新公式出发，而非仅对激活幅值作跨时间步能量对齐，"
        "推导扩散 UNet 激活量化中逐时间步最优通道缩放因子。核心思想是将通道缩放置于去噪误差传播链中，"
        "最小化量化扰动对下一步潜变量 x_{t−1} 的影响；在线性化假设下给出闭式解，"
        "并说明如何聚合为静态部署因子及吸收到下一层权重。最后与基于激活能量 A^DDIM 的启发式方法作对比。",
        indent=True,
    )

    # ===== 1 =====
    add_heading(doc, "1  问题设定与符号", level=1)

    add_para(doc, "1.1  DDIM 确定性去噪更新", bold=True)
    add_para(
        doc,
        "设扩散过程共有 T 个离散时间步。DDIM（η=0）确定性采样的一步更新为：",
        indent=True,
    )
    add_eq(doc, "x_{t−1} = ā_{t−1} · x̂_{0,t} + b_{t−1} · ε̂_t", 1)
    add_para(doc, "其中 DDIM 系数定义为：", indent=True)
    add_eq(doc, "a_t = √ᾱ_t ,    b_t = √(1 − ᾱ_t)", 2)
    add_para(doc, "网络预测分量为：", indent=True)
    add_eq(
        doc,
        "ε̂_t = ε_θ(x_t, t) ,    "
        "x̂_{0,t} = (x_t − b_t · ε̂_t) / a_t",
        3,
    )
    add_para(
        doc,
        "式 (1) 表明：每一步去噪输出 x_{t−1} 是「预测干净样本 x̂_{0,t}」与「噪声方向 ε̂_t」"
        "的显式线性组合，系数 a_{t−1}, b_{t−1} 由噪声调度确定。",
        indent=True,
    )

    add_para(doc, "1.2  等价 PF-ODE 视角", bold=True)
    add_para(
        doc,
        "将 DDIM 更新视为概率流 ODE（PF-ODE）的一步 Euler 离散化：",
        indent=True,
    )
    add_eq(doc, "(x_{t−1} − x_t) / Δt ≈ F_θ(x_t, t) ,    Δt = (t−1) − t", 4)
    add_para(
        doc,
        "第 ℓ 层激活 h_{ℓ,t} = H_ℓ(x_t, t) 为 UNet 从输入到第 ℓ 层的前缀映射。"
        "沿 PF-ODE 轨迹，激活动力学满足全导数链式法则（见附录 A）。",
        indent=True,
    )

    add_para(doc, "1.3  激活量化与通道缩放", bold=True)
    add_para(
        doc,
        "对第 ℓ 层输出激活施加逐通道缩放（部署时为静态 s_{ℓ,c}）：",
        indent=True,
    )
    add_eq(doc, "h′_{ℓ,t,c,u} = h_{ℓ,t,c,u} / s_{ℓ,c}", 5)
    add_para(
        doc,
        "随后对 h′ 做均匀仿射量化 Q(·)，并将缩放吸收到下一层权重 W_{ℓ+1}（见第 6 节），"
        "以保证浮点前向等价。本文目标：从式 (1) 出发，求使去噪误差最小的 s_{ℓ,c}。",
        indent=True,
    )

    # ===== 2 =====
    add_heading(doc, "2  从去噪公式出发的误差传播", level=1)

    add_para(doc, "2.1  量化误差如何进入 DDIM 更新", bold=True)
    add_para(
        doc,
        "设第 ℓ 层经缩放与量化后，实际送入后续网络的激活为：",
        indent=True,
    )
    add_eq(doc, "h̃_{ℓ,t} = s_{ℓ,t} · Q(h_{ℓ,t} / s_{ℓ,t})", 6)
    add_para(doc, "定义量化扰动：", indent=True)
    add_eq(doc, "δh_{ℓ,t} = h̃_{ℓ,t} − h_{ℓ,t}", 7)
    add_para(
        doc,
        "扰动 δh_{ℓ,t} 经 UNet 后缀传播，改变网络预测：",
        indent=True,
    )
    add_eq(doc, "δε̂_t ≈ G^{(ε)}_{ℓ,t} · δh_{ℓ,t} ,    δx̂_{0,t} ≈ G^{(0)}_{ℓ,t} · δh_{ℓ,t}", 8)
    add_para(
        doc,
        "其中 G^{(ε)}_{ℓ,t} = ∂ε_θ/∂h_{ℓ,t}|_{x_t,t}，G^{(0)}_{ℓ,t} = ∂x̂_{0,t}/∂h_{ℓ,t}|_{x_t,t} "
        "为后缀 Jacobian（对通道 c 可写为向量/矩阵块）。",
        indent=True,
    )

    add_para(doc, "2.2  去噪输出的一阶扰动（核心公式）", bold=True)
    add_para(
        doc,
        "将式 (8) 代入 DDIM 更新式 (1)，由一阶 Taylor 展开得：",
        indent=True,
    )
    add_eq(
        doc,
        "δx_{t−1} ≈ a_{t−1} · δx̂_{0,t} + b_{t−1} · δε̂_t",
        9,
    )
    add_eq(
        doc,
        "δx_{t−1} ≈ (a_{t−1} · G^{(0)}_{ℓ,t} + b_{t−1} · G^{(ε)}_{ℓ,t}) · δh_{ℓ,t}",
        10,
    )
    add_para(
        doc,
        "式 (10) 是从去噪公式本身直接导出的误差传播式：第 ℓ 层的量化误差 δh "
        "经 DDIM 系数加权的后缀敏感度矩阵，映射到下一步潜变量扰动 δx_{t−1}。"
        "这与仅对齐激活能量 E[h²] 的方法有本质区别——优化对象是去噪轨迹上的状态误差。",
        indent=True,
    )

    add_para(doc, "2.3  量化误差模型", bold=True)
    add_para(
        doc,
        "对通道 c，设均匀量化步长 Δ_{ℓ,t,c} 与缩放 s_{ℓ,t,c} 相关。"
        "在常见假设下，δh_{ℓ,t,c,u} 的方差近似正比于：",
        indent=True,
    )
    add_eq(doc, "Var(δh_{ℓ,t,c,u}) ∝ Δ_{ℓ,t,c}² ∝ (h_{ℓ,t,c,u} / s_{ℓ,t,c})² · 2^{−2n}", 11)
    add_para(
        doc,
        "其中 n 为激活量化位宽。因此增大 s_{ℓ,t,c} 可减小量化步长（相对 h 的粒度更细），"
        "但会通过权重吸收改变下一层权重量化尺度（见第 6 节）。",
        indent=True,
    )

    # ===== 3 =====
    add_heading(doc, "3  去噪感知最优缩放：目标函数", level=1)

    add_para(doc, "3.1  单步去噪 MSE 目标", bold=True)
    add_para(
        doc,
        "对固定的层 ℓ、时间步 t，定义去噪感知目标：",
        indent=True,
    )
    add_eq(
        doc,
        "J(s_{ℓ,t,c}) = E_{x,u} [ ‖(δx_{t−1})_{·,c,u}‖² ]",
        12,
    )
    add_para(
        doc,
        "在通道解耦近似下（忽略通道间 Jacobian 耦合），将式 (10) 对通道 c 写为：",
        indent=True,
    )
    add_eq(
        doc,
        "J(s_{ℓ,t,c}) ≈ D_{ℓ,t,c} / s_{ℓ,t,c}² + 正则项",
        13,
    )
    add_para(doc, "其中去噪敏感度能量定义为：", indent=True)
    add_eq(
        doc,
        "D_{ℓ,t,c} = E_{x,u} [ ‖ a_{t−1} · g^{(0)}_{ℓ,t,c,u} + b_{t−1} · g^{(ε)}_{ℓ,t,c,u} ‖² ]",
        14,
    )
    add_para(
        doc,
        "g^{(0)}_{ℓ,t,c,u}、g^{(ε)}_{ℓ,t,c,u} 分别为 G^{(0)}、G^{(ε)} 在第 c 通道、空间位置 u 的作用分量；"
        "期望在校准样本与空间位置上取平均。",
        indent=True,
    )

    add_para(doc, "3.2  线性化下的 D_{ℓ,t,c} 与 DDIM 两分支结构", bold=True)
    add_para(
        doc,
        "对前缀映射 H_ℓ 在当前轨迹点 (x_t, t) 作一阶线性化：",
        indent=True,
    )
    add_eq(doc, "h_{ℓ,t} ≈ J_{ℓ,t} · x_t + r_{ℓ,t}", 15)
    add_para(doc, "代入 x_t = a_t · x̂_{0,t} + b_t · ε̂_t（由式 (2)(3) 可得），有：", indent=True)
    add_eq(
        doc,
        "h_{ℓ,t,c,u} ≈ [J_{ℓ,t} · a_t · x̂_{0,t}]_{c,u} + [J_{ℓ,t} · b_t · ε̂_t]_{c,u} + r_{ℓ,t,c,u}",
        16,
    )
    add_para(
        doc,
        "注意：式 (16) 描述的是激活如何由 DDIM 两分量构成；"
        "而式 (14) 的 D_{ℓ,t,c} 描述的是激活扰动如何经 DDIM 更新影响 x_{t−1}。"
        "二者通过不同 Jacobian 联系：",
        indent=True,
    )
    add_bullet(doc, "式 (16)：前缀 J_{ℓ,t}，解释 h 的构成（输入侧）")
    add_bullet(doc, "式 (14)：后缀 G^{(0)}, G^{(ε)}，解释 δh 对 x_{t−1} 的影响（输出侧）")

    add_para(
        doc,
        "对后缀 Jacobian，由 x̂_{0,t} = (x_t − b_t ε̂_t)/a_t 及 ε̂_t = ε_θ(x_t,t) 可得：",
        indent=True,
    )
    add_eq(
        doc,
        "G^{(0)}_{ℓ,t} = (1/a_t) · ∂x_t/∂h_{ℓ,t} − (b_t/a_t) · G^{(ε)}_{ℓ,t}",
        17,
    )
    add_para(
        doc,
        "代入式 (14)，去噪敏感度可完全用 G^{(ε)} 与输入侧偏导表示，"
        "实际计算时可通过有限差分直接测量 δx_{t−1}，无需显式构造 Jacobian（见第 7 节）。",
        indent=True,
    )

    # ===== 4 =====
    add_heading(doc, "4  跨时间步参考与闭式最优解", level=1)

    add_para(doc, "4.1  跨时间参考能量", bold=True)
    add_para(
        doc,
        "设 T_cal 为校准用 DDIM 时间步集合。定义通道 c 的跨时间参考去噪敏感度：",
        indent=True,
    )
    add_eq(
        doc,
        "D^{ref}_{ℓ,c} = (1/|T_cal|) · Σ_{t∈T_cal} D_{ℓ,t,c}",
        18,
    )

    add_para(doc, "4.2  逐时间步最优缩放闭式解", bold=True)
    add_para(
        doc,
        "为使不同时间步下量化扰动对 x_{t−1} 的影响对齐到同一参考水平，构造最小二乘：",
        indent=True,
    )
    add_eq(
        doc,
        "min_{s_{ℓ,t,c}>0}  ( D_{ℓ,t,c} / s_{ℓ,t,c}² − D^{ref}_{ℓ,c} )²",
        19,
    )
    add_para(doc, "令 z_{ℓ,t,c} = 1/s_{ℓ,t,c}²，对 z 求导并令其为零，当 D_{ℓ,t,c} > 0 时：", indent=True)
    add_eq(doc, "z*_{ℓ,t,c} = D^{ref}_{ℓ,c} / D_{ℓ,t,c}", 20)
    add_eq(
        doc,
        "s*_{ℓ,t,c} = √( D_{ℓ,t,c} / D^{ref}_{ℓ,c} )",
        21,
    )
    add_para(
        doc,
        "物理解释：若第 ℓ 层通道 c 在时间步 t 对 x_{t−1} 的敏感度 D_{ℓ,t,c} 高于跨时间参考，"
        "则 s* > 1，缩放后降低该通道激活幅值、细化量化粒度，从而抑制 δx_{t−1}；反之 s* < 1。",
        indent=True,
    )

    add_para(doc, "4.3  与激活能量法的结构对比", bold=True)
    add_para(
        doc,
        "激活能量法（原 PDF 方案）使用：",
        indent=True,
    )
    add_eq(doc, "A^{DDIM}_{ℓ,t,c} ≈ E_{x,u}[ h²_{ℓ,t,c,u} ]", 22)
    add_eq(doc, "s*_{act,ℓ,t,c} = √( A^{DDIM}_{ℓ,t,c} / A^{ref}_{ℓ,c} )", 23)
    add_para(
        doc,
        "本文去噪感知法使用：",
        indent=True,
    )
    add_eq(doc, "D_{ℓ,t,c} = E_{x,u}[ ‖ a_{t−1} g^{(0)} + b_{t−1} g^{(ε)} ‖² ]", 24)
    add_eq(doc, "s*_{denoise,ℓ,t,c} = √( D_{ℓ,t,c} / D^{ref}_{ℓ,c} )", 25)
    add_para(
        doc,
        "两者闭式结构相同（s* = √(M/M_ref)），但统计量 M 不同："
        "A^{DDIM} 仅反映激活幅值；D_{ℓ,t,c} 反映激活扰动经 DDIM 公式传播到 x_{t−1} 的敏感度。"
        "当后缀 Jacobian 与激活幅值高度相关时两者近似；一般情形下不等价。",
        indent=True,
    )

    # ===== 5 =====
    add_heading(doc, "5  融合 ODE 轨迹项的扩展", level=1)

    add_para(
        doc,
        "若希望同时惩罚激活沿 PF-ODE 轨迹的快速变化（时间敏感性），可定义速度能量：",
        indent=True,
    )
    add_eq(
        doc,
        "V_{ℓ,t,c} = E_{x,u}[ ((h_{ℓ,t−1,c,u} − h_{ℓ,t,c,u}) / Δt)² ]",
        26,
    )
    add_para(doc, "扩展统计量：", indent=True)
    add_eq(doc, "B_{ℓ,t,c} = D_{ℓ,t,c} + λ · Δt² · V_{ℓ,t,c}", 27)
    add_para(
        doc,
        "其中 λ ≥ 0 控制对轨迹速度的权重。将式 (19) 中 D 替换为 B，得 ODE-去噪联合缩放：",
        indent=True,
    )
    add_eq(
        doc,
        "s*_{ℓ,t,c} = √( B_{ℓ,t,c} / B^{ref}_{ℓ,c} ) ,    "
        "B^{ref}_{ℓ,c} = (1/|T_cal|) Σ_{t} B_{ℓ,t,c}",
        28,
    )
    add_para(
        doc,
        "注意：V_{ℓ,t,c} 仍来自激活动力学式 (4)，而 D_{ℓ,t,c} 来自去噪公式 (1)。"
        "联合方案同时考虑「扰动对下一步去噪的影响」与「激活沿轨迹的变化率」。",
        indent=True,
    )

    # ===== 6 =====
    add_heading(doc, "6  静态部署因子与权重吸收", level=1)

    add_para(doc, "6.1  逐时间步因子聚合为静态因子", bold=True)
    add_para(
        doc,
        "部署时通常不为每个时间步维护独立缩放，需将 {s*_{ℓ,t,c}}_{t∈T_cal} 聚合为单个 s^{deploy}_{ℓ,c}。",
        indent=True,
    )
    add_para(doc, "算术平均：", indent=True)
    add_eq(doc, "s̄_{ℓ,c} = (1/|T_cal|) · Σ_{t∈T_cal} s*_{ℓ,t,c}", 29)
    add_para(doc, "推荐：对数域（几何）平均，最小化跨时间对数尺度偏差：", indent=True)
    add_eq(
        doc,
        "s̄_{ℓ,c} = exp( (1/|T_cal|) · Σ_{t∈T_cal} log(s*_{ℓ,t,c} + ε) )",
        30,
    )
    add_para(doc, "保守部署（指数 α 与裁剪）：", indent=True)
    add_eq(
        doc,
        "s^{deploy}_{ℓ,c} = clip( s̄_{ℓ,c}^α ,  s_min,  s_max ) ,    α ∈ [0, 1]",
        31,
    )
    add_para(
        doc,
        "α=0 表示不启用缩放；α=1 完全采用闭式解；0<α<1 为保守折中。"
        "建议初值：α=0.5，s_min=0.5，s_max=2.0。",
        indent=True,
    )

    add_para(doc, "6.2  权重吸收与浮点等价性", bold=True)
    add_para(
        doc,
        "对卷积层 W_{ℓ+1} ∈ R^{C_out×C_in×k_h×k_w}，缩放 S_ℓ = diag(s^{deploy}_{ℓ,1}, …, s^{deploy}_{ℓ,C}) 吸收为：",
        indent=True,
    )
    add_eq(doc, "W′_{ℓ+1,o,c,:,:} = s^{deploy}_{ℓ,c} · W_{ℓ+1,o,c,:,:}", 32)
    add_para(doc, "则对任意时间步 t：", indent=True)
    add_eq(
        doc,
        "W′_{ℓ+1} · (S_ℓ)^{−1} · h_{ℓ,t} = W_{ℓ+1} · h_{ℓ,t}",
        33,
    )
    add_para(
        doc,
        "浮点推理严格等价；变化的是激活量化器与下一层权重量化器所见的通道尺度分布。",
        indent=True,
    )

    # ===== 7 =====
    add_heading(doc, "7  工程实现：有限差分估计 D_{ℓ,t,c}", level=1)

    add_para(
        doc,
        "无需显式计算 G^{(0)}、G^{(ε)}。校准阶段沿 DDIM 轨迹执行：",
        indent=True,
    )
    add_bullet(doc, "Step 1：对校准样本运行完整 DDIM 前向，缓存各层 h_{ℓ,t} 及 x_{t−1}")
    add_bullet(doc, "Step 2：对第 ℓ 层通道 c 注入小扰动 ε_fd（如 ε_fd = η · std(h_{ℓ,t,c})，η≈10^{−3}）")
    add_bullet(doc, "Step 3：重新前向，测量 δx_{t−1}^{(fd)} = x_{t−1}^{pert} − x_{t−1}^{base}")
    add_bullet(doc, "Step 4：估计 D_{ℓ,t,c} ≈ ‖δx_{t−1}^{(fd)}‖² / ε_fd²（对样本与空间位置平均）")
    add_bullet(doc, "Step 5：按式 (18)(21)(30)(31) 计算 s^{deploy}_{ℓ,c}，按式 (32) 吸收权重")

    add_para(
        doc,
        "计算复杂度：每层每通道需额外一次（或两次，中心差分）前向传播。"
        "可只对 QuantModule 输出层做缩放，与 q-diffusion 中 BRECQ/LSQ 流程兼容："
        "先 ODE-去噪感知缩放 + 权重吸收，再进行权重量化与激活量化校准。",
        indent=True,
    )

    # ===== 8 =====
    add_heading(doc, "8  完整算法流程", level=1)

    add_para(doc, "算法 1：基于 DDIM 去噪公式的通道缩放校准", bold=True)
    steps = [
        "输入：浮点 UNet θ，校准集，DDIM 时间步集合 T_cal，超参 λ, α, s_min, s_max, ε",
        "沿 DDIM 轨迹收集各 QuantModule 输出激活 {h_{ℓ,t}}",
        "（可选）收集相邻步激活，计算 V_{ℓ,t,c}，构造 B_{ℓ,t,c} = D_{ℓ,t,c} + λΔt²V_{ℓ,t,c}",
        "有限差分或解析 Jacobian 计算 D_{ℓ,t,c}（或 B_{ℓ,t,c}）",
        "计算 M^{ref}_{ℓ,c} = mean_t M_{ℓ,t,c}，M ∈ {D, B}",
        "逐时间步闭式解：s*_{ℓ,t,c} = √( (M_{ℓ,t,c}+ε) / (M^{ref}_{ℓ,c}+ε) )",
        "几何平均 + 保守指数 + 裁剪得 s^{deploy}_{ℓ,c}",
        "将 s^{deploy} 写入模型：激活除以 s，下一层权重乘以 s",
        "输出：缩放后的浮点模型，供后续 PTQ（BRECQ/LSQ）使用",
    ]
    for i, s in enumerate(steps, 1):
        add_para(doc, f"{i}. {s}")

    # ===== 9 =====
    add_heading(doc, "9  结论", level=1)

    add_para(
        doc,
        "本文给出了从 DDIM 去噪公式 (1) 出发的通道缩放完整推导链："
        "量化扰动 δh → 预测扰动 δx̂_0, δε̂ → 去噪输出扰动 δx_{t−1}（式 10）→ "
        "最小化去噪 MSE → 闭式解 s* = √(D/D^{ref})（式 21）→ 静态聚合与权重吸收（式 30–33）。",
        indent=True,
    )
    add_para(
        doc,
        "与原激活能量方案 s* = √(A/A^{ref}) 相比，本文方法的优化目标与 DDIM 采样语义一致，"
        "理论上更直接地服务于「减小量化对去噪轨迹的破坏」。"
        "实践中可通过有限差分高效估计 D_{ℓ,t,c}，无需显式 Jacobian。",
        indent=True,
    )

    # ===== Appendix =====
    add_heading(doc, "附录 A  激活动力学式 (4) 的推导", level=1)
    add_para(doc, "由 h_{ℓ,t} = H_ℓ(x_t, t)，对时间求全导数：", indent=True)
    add_eq(doc, "dh_{ℓ,t}/dt = (∂H_ℓ/∂x)(dx_t/dt) + ∂H_ℓ/∂t", "A.1")
    add_para(doc, "PF-ODE 速度场 dx_t/dt = F_θ(x_t, t)，Euler 离散化：", indent=True)
    add_eq(
        doc,
        "(h_{ℓ,t−1} − h_{ℓ,t}) / Δt ≈ (∂H_ℓ/∂x) F_θ(x_t,t) + ∂H_ℓ/∂t",
        "A.2",
    )

    add_heading(doc, "附录 B  符号表", level=1)
    symbols = [
        ("x_t", "时间步 t 的潜变量"),
        ("x̂_{0,t}, ε̂_t", "网络预测的干净样本与噪声"),
        ("a_t, b_t", "DDIM 系数 √ᾱ_t, √(1−ᾱ_t)"),
        ("h_{ℓ,t}", "第 ℓ 层激活"),
        ("s_{ℓ,t,c}, s^{deploy}_{ℓ,c}", "逐时间步 / 静态部署缩放因子"),
        ("D_{ℓ,t,c}", "去噪敏感度能量（本文核心统计量）"),
        ("A^{DDIM}_{ℓ,t,c}", "激活能量 E[h²]（原 PDF 方案）"),
        ("G^{(0)}, G^{(ε)}", "后缀 Jacobian ∂x̂_0/∂h, ∂ε̂/∂h"),
        ("T_cal", "校准时间步集合"),
        ("α, s_min, s_max", "保守指数与裁剪超参"),
    ]
    table = doc.add_table(rows=1 + len(symbols), cols=2)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text = "符号"
    hdr[1].text = "含义"
    for i, (sym, meaning) in enumerate(symbols, 1):
        table.rows[i].cells[0].text = sym
        table.rows[i].cells[1].text = meaning

    doc.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "Denoise_Aware_Channel_Scaling_Derivation.docx")
    build_document(out)
