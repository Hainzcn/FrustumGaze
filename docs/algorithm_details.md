# Python服务端 核心算法详解

本段详细描述了 FrustumGaze 项目中用于计算用户头部、手部空间位置以及视线方向的核心算法。

## 1. 坐标系定义与空间变换

在构建三维交互系统前，首要任务是确立统一的空间参照系。本项目涉及多个坐标空间的变换与映射。

*   **图像坐标系 (Pixel Coordinates, $\mathcal{UV}$)**:
    *   定义：原点位于图像左上角，向右为 $u$ 轴正方向，向下为 $v$ 轴正方向。
    *   单位：像素 (pixel)。
    *   用途：直接对应摄像头采集的二维图像数据。

*   **归一化坐标系 (Normalized Coordinates, $\mathcal{NDC}$)**:
    *   定义：MediaPipe 输出的无量纲坐标，范围 $[0, 1]$。
    *   变换： $u_{pixel} = u_{norm} \times W_{img}, \quad v_{pixel} = v_{norm} \times H_{img}$ 。

*   **相机坐标系 (Camera Space, $\mathcal{C}$)**:
    *   定义：遵循右手坐标系定则，以摄像头光心 (Optical Center) 为原点。
        *   $X_c$：水平向右。
        *   $Y_c$：垂直向下。
        *   $Z_c$：沿光轴向外（深度方向）。
    *   单位：厘米 (cm)。
    *   用途：所有三维空间计算（头部、手部位置）的基准参考系。

---

## 2. 头部空间位姿解算 (Head Pose & Position Estimation)

头部追踪的核心挑战在于从单目二维图像中恢复三维空间信息，这是一个典型的**不适定问题**，需引入几何知识进行求解。

### 2.1 双通道头部深度估算 (Dual-Channel Head Depth Estimation)

为了提高单目深度估算的鲁棒性，特别是在头部发生旋转（Yaw/Pitch）时，我们采用了类似于手部追踪的双通道融合策略。

#### 2.1.1 核心思想
利用人脸结构的几何不变性，构建两个独立的深度估算通道，并根据头部姿态进行动态融合与校准。
*   **通道 A（宽度通道）**：基于双眼外眼角的水平间距。
*   **通道 B（长度通道）**：基于眉心（Glabella）到鼻尖（Nose Tip）的垂直距离。

#### 2.1.2 通道定义与特性
1.  **宽度通道 ($Z_{width}$)**
    *   **特征点**：左外眼角 (33) - 右外眼角 (263)。
    *   **物理基准**：$W_{ref}$ (初始约为 9cm，动态校准)。
    *   **特性**：对 Pitch 旋转不敏感，但易受 Yaw 旋转（透视缩短）影响。
    *   **计算**：
        $$Z_{width} = \frac{f \cdot W_{ref} \cdot |\cos(\text{Yaw})|}{d_{width\_pixel}}$$

2.  **长度通道 ($Z_{length}$)**
    *   **特征点**：眉心 (168) - 鼻尖 (1)。
    *   **物理基准**：$L_{ref} = 6.0 \text{cm}$（`FACE_REF_LENGTH_CM`，固定先验值）。
    *   **特性**：对 Yaw 旋转不敏感，但易受 Pitch 旋转影响。
    *   **计算**：
        $$Z_{length} = \frac{f \cdot L_{ref} \cdot |\cos(\text{Pitch})|}{d_{length\_pixel}}$$

#### 2.1.3 动态校准 (Dynamic Calibration)
由于个体脸型差异（眼距不同），我们以垂直方向的**长度通道**（假设眉心到鼻尖距离相对固定或作为尺度标准）为基准，动态校准**宽度通道**的物理宽度 $W_{ref}$。

*   **校准条件**：头部姿态端正（$|\text{Yaw}|$ < `min_valid_yaw` 且 $|\text{Pitch}|$ < `min_valid_pitch`，默认均为 15°）。
*   **异常拒绝**：当单帧估计宽度偏离当前校准值超过 `max_deviation_ratio`（默认 20%）时，跳过本帧更新，防止噪声突变污染校准值。
*   **更新逻辑**：使用 EMA 更新 $W_{ref}$（$\alpha$ = `width_correction_alpha`，默认 0.05）：
    $$W_{ref} \leftarrow (1 - \alpha) \cdot W_{ref} + \alpha \cdot \hat{W}$$
    其中 $\hat{W} = \frac{Z_{length} \cdot d_{width\_pixel}}{f \cdot |\cos(\text{Yaw})|}$ 是从长度通道反推的宽度估计。
*   **漂移约束**：校准值被钳位到参考值的 $\pm$`clamp_ratio`（默认 30%）范围内：
    $$W_{ref} \in [W_{init} \cdot (1 - 0.3),\; W_{init} \cdot (1 + 0.3)]$$

#### 2.1.4 深度融合 (Depth Fusion)
最终深度 $Z_{est}$ 是两通道的加权平均，权重取决于头部旋转角度（角度越小，投影越可靠）：
$$Z_{est} = \frac{w_{width} \cdot Z_{width} + w_{length} \cdot Z_{length}}{w_{width} + w_{length}}$$

权重使用余弦的 $n$ 次幂（`weight_power`，默认 $n=2$）：
$$w_{width} = |\cos(\text{Yaw})|^n, \quad w_{length} = |\cos(\text{Pitch})|^n$$

幂次 $n > 1$ 使权重在角度增大时衰减更快，从而更积极地抑制大角度通道的贡献。当面部正对镜头时 ($\text{Yaw} \approx 0, \text{Pitch} \approx 0$)，两通道权重近似相等。

#### 2.1.5 Z轴滤波 (Z-Filtering)
在将深度 $Z$ 用于反投影计算 $(X, Y)$ 坐标前，先对 $Z$ 进行 `OneEuroFilter` 滤波，防止深度的噪声导致平面坐标 $(X, Y)$ 的抖动。


### 2.2 头部姿态解算 (Head Pose Estimation)

头部姿态（Yaw, Pitch）通过**面部法向量法 (Face Normal Method)** 从关键点几何关系直接推导，不依赖 PnP 求解。

#### 2.2.1 算法路径

1.  **特征点选取**：选取 4 个具有良好几何分布的面部特征点：
    *   左外眼角 (33)、右外眼角 (263) — 定义水平方向
    *   下巴 (152) — 定义垂直方向
    *   眉心 (168) — 与下巴构成纵向轴

2.  **面部平面向量构建**：
    *   水平向量：$\vec{v}_{h} = P_{263} - P_{33}$（从左眼角指向右眼角）
    *   垂直向量：$\vec{v}_{v} = P_{152} - P_{168}$（从眉心指向下巴）

3.  **法向量计算**：
    通过叉乘得到面部平面的法向量 $\vec{n}$：
    $$\vec{n} = \vec{v}_{h} \times \vec{v}_{v}$$
    并对 $\vec{n}$ 进行归一化。法向量指向面部"正前方"。

4.  **欧拉角提取**：
    从归一化法向量 $\hat{n} = (n_x, n_y, n_z)$ 直接提取 Yaw 和 Pitch：
    $$\text{Yaw} = \arctan2(n_x, n_z)$$
    $$\text{Pitch} = -\arcsin(n_y) + \beta$$
    其中 $\beta \approx 30°$ 是经验性偏置修正，用于补偿人脸静息时 MediaPipe 特征点分布产生的固有法向量倾斜。

5.  **旋转矩阵构建**：
    从 Yaw/Pitch 欧拉角直接构造旋转矩阵 $\mathbf{R} = \mathbf{R}_y(\text{Yaw}) \cdot \mathbf{R}_x(\text{Pitch})$，而非从 PnP 获得。`rvec` 则通过 `cv2.Rodrigues(R)` 从该矩阵转换得到，仅用于可视化绘制。

6.  **滤波**：对 Yaw 和 Pitch 分别应用 `OneEuroFilter` 进行平滑处理。

### 2.3 平面位置反投影 (Back-Projection)

获得稳定的深度 $Z_{corrected}$ 后，利用针孔模型逆变换，将图像平面坐标 $(u, v)$ 反推回相机空间坐标 $(X, Y)$：

$$X = Z_{corrected} \times \frac{u - c_x}{f_x}, \quad Y = Z_{corrected} \times \frac{v - c_y}{f_y}$$

### 至此，我们完成了从 2D 图像到 3D 空间 $(X, Y, Z)$ 的完整重构。

---

## 3. 视线追踪算法 (Gaze Estimation)

本项目**基于纯 2D-3D 反投影的几何模型完成视线估计**，不依赖 PnP 的旋转/平移矩阵，而是通过相机内参将 2D 特征点反投影至 3D 空间，再利用射线-球面交点重建视线向量。

### 3.1 单眼视线计算流程

对每只眼睛独立执行以下步骤（`_compute_single_eye_gaze`）：

1.  **眼球中心 2D 定位**：
    取内外眼角 landmark 的中点作为眼球中心的 2D 投影，并对该中点应用 `OneEuroFilter` 平滑：
    $$P_{eye\_2d} = \frac{P_{inner} + P_{outer}}{2}$$

2.  **2D → 3D 反投影**：
    利用相机内参逆矩阵 $\mathbf{K}^{-1}$ 将 2D 齐次坐标转换为 3D 射线方向：
    $$\vec{d}_{eye} = \mathbf{K}^{-1} \cdot \begin{pmatrix} u_{eye} \\ v_{eye} \\ 1 \end{pmatrix}$$
    沿射线投影到头部估计深度 $Z_{head}$ 处，得到眼球表面点 $P_{surface}$：
    $$P_{surface} = \vec{d}_{eye} \cdot \frac{Z_{head}}{(\vec{d}_{eye})_z}$$

3.  **球心计算**：
    沿射线方向向内偏移一个眼球半径 $r$（`EYE_BALL_RADIUS_CM`，默认 1.2cm），得到球心 $C$：
    $$C = P_{surface} + r \cdot \hat{d}_{eye}$$

4.  **虹膜射线投射**：
    虹膜中心 2D 坐标（经 `OneEuroFilter` 平滑后）同样通过 $\mathbf{K}^{-1}$ 反投影为射线 $\vec{d}_{iris}$。

5.  **射线-球面求交**：
    解方程 $\| O + t \cdot \vec{d}_{iris} - C \|^2 = r^2$（其中 $O$ 为相机光心位置），取较近的交点 $P_{iris\_3d}$。若无交点，回退为射线上距球心最近点。

6.  **视线向量生成**：
    $$\vec{V}_{gaze} = \frac{P_{iris\_3d} - C}{\| P_{iris\_3d} - C \|}$$

### 3.2 双眼置信度融合

系统对双眼分别计算置信度（`_compute_eye_confidence`），并加权融合得到最终视线。

**置信度由两个因子相乘得到：**

1.  **Yaw 几何可见性权重**：当头部向一侧偏转时，该侧眼睛的虹膜投影被压缩、不再可靠：
    $$w_{yaw} = \text{clamp}\left(\frac{|\text{Yaw}_{max}| - |\text{Yaw}|}{|\text{Yaw}_{max}| - |\text{Yaw}_{min}|},\; 0,\; 1\right)$$
    左眼在 Yaw > 0（向右偏转）时权重下降，右眼反之。

2.  **眼睑开合度**：取上下眼睑 landmark 的垂直距离与眼睛水平宽度的比值；眼睛越闭合，置信度越低。

**最终融合：**
$$\vec{V}_{final} = \frac{w_L \cdot \vec{V}_L + w_R \cdot \vec{V}_R}{w_L + w_R}$$

若某只眼置信度为 0 则完全使用另一只；若两者均为 0 则回退为头部朝向。

### 3.3 屏幕平面交点

获得融合后的 3D 视线向量后，计算其与虚拟屏幕平面的交点（`calculate_screen_intersection`）：

*   以屏幕中心为原点、屏幕法线 $\hat{n}_{screen}$ 定义平面。
*   求射线 $(P_{eye}, \vec{V}_{gaze})$ 与该平面的交点参数 $t$：
    $$t = \frac{(\vec{P}_{screen} - \vec{P}_{eye}) \cdot \hat{n}_{screen}}{\vec{V}_{gaze} \cdot \hat{n}_{screen}}$$
*   交点坐标归一化到 $[0, 1]$ 范围映射为屏幕 UV。

### 3.4 备注

旋转矩阵 $\mathbf{R}$ 和旋转向量 $\vec{r}$（由 2.2 节面部法向量法构建）**不参与视线计算**，仅用于 OpenCV 可视化绘制面部坐标轴。

---

## 4. 手部追踪算法详解 (Hand Tracking Algorithm Details)

本段详细描述了 `FrustumGaze` 项目中手部追踪模块（`hand_tracker.py`）所使用的核心算法，包括几何特征提取、深度估算、动态噪声调整及状态融合策略。该算法旨在从单目摄像头输入中稳健地估计手部的 3D 空间位置，并重点解决了手部形变（如握拳）对深度估算的影响。

### 4.1 核心流程概述

手部追踪的主要流程如下：

1.  **MediaPipe Landmarks 检测**：获取手部 21 个关键点的归一化坐标。
2.  **关键点滤波**：对每个 landmark 的 $(x, y)$ 坐标应用 `OneEuroFilter`，减少抖动。
3.  **几何特征计算**：构建统一坐标系，计算手部的 Yaw/Pitch 旋转角及聚拢系数（Grip Factor）。
4.  **多通道深度估算**：
    *   **长度通道 (Up)**：基于手腕到中指根部的长度。
    *   **宽度通道 (Across)**：基于食指根部到小指根部的宽度（作为基准）。
5.  **深度融合**：根据手部姿态和聚拢程度，动态加权融合各通道深度。
6.  **Z 轴滤波**：融合后的深度先经 `OneEuroFilter` 平滑，再用于反投影 XY 坐标。
7.  **时序平滑与锚定**：
    *   **Motion Score**：基于深度变化率检测手部运动状态。
    *   **Depth Anchor**：在低置信度状态下（如握拳）引入历史高置信度值。
    *   **Kalman Filtering**：动态调整观测噪声协方差（R），实现自适应平滑。
8.  **捏合检测**：基于拇指与食指 3D 距离判定捏合状态，并进行去抖处理。

### 4.2 详细算法逻辑

#### 4.2.1 坐标系与几何特征

我们首先定义一个统一的坐标空间，并计算手部的法向量以推导旋转角。

*   **统一坐标**：将 MediaPipe 输出的 $(x, y, z)$ 转换为以图像宽度为基准的统一尺度，消除纵横比影响。
*   **关键向量**：
    *   $\vec{v}_{up} = P_{MiddleMCP} - P_{Wrist}$ （纵向向量）
    *   $\vec{v}_{across} = P_{PinkyMCP} - P_{IndexMCP}$ （横向向量）
*   **法向量与旋转角**：
    *   $\vec{n} = \text{normalize}(\vec{v}_{up} \times \vec{v}_{across})$
    *   **Yaw**：法向量在 X-Z 平面的投影角度。
    *   **Pitch**：法向量在 Y 轴的分量推导出的俯仰角。

#### 4.2.2 聚拢系数 (Grip Factor)

为了量化手部的握拳程度，我们引入聚拢系数 $G$：

$$
G = \text{clamp}\left( \frac{R_{open} - \text{ratio}}{R_{open} - R_{closed}}, 0, 1 \right)
$$

其中 $\text{ratio}$ 是指尖到手腕的平均距离与手掌参考长度的比值。
*   $G \approx 0$：手掌完全展开。
*   $G \approx 1$：手掌完全握拳。
*   该系数经过 EMA（指数移动平均）平滑，用于后续的权重调整。

#### 4.2.3 多通道深度估算

由于单目深度估算依赖于透视投影原理（$Z = \frac{f \cdot L_{real}}{L_{pixel}}$），手部形变会导致严重的估算误差。我们采用双通道策略：

##### A. 宽度通道 ($Z_{across}$) - **主基准**
*   **依据**：食指根部到小指根部的宽度。
*   **特点**：受握拳影响较小（骨骼结构相对固定），且在手掌侧转（Yaw）时仍能提供稳定参考。
*   **计算**：
    $$
    Z_{across} = \frac{f \cdot L_{ref\_width} \cdot |\cos(\text{Yaw})|}{L_{pixel\_across}}
    $$

##### B. 长度通道 ($Z_{up}$) - **被校准**
*   **依据**：手腕到中指根部的距离。
*   **特点**：手掌展开时准确，但在握拳时严重失真。
*   **动态长度校准 (Dynamic Length Correction)**：
    由于个体手掌比例差异，长度通道需要校准以匹配宽度通道。系统维护一个校准系数 $C_{length}$，在手掌**展开、静止且正对**摄像头时自动学习：
    $$
    C_{length} \leftarrow \text{EMA}\left( \frac{Z_{across}}{Z_{up\_raw}} \right)
    $$
    最终深度：$Z_{up} = Z_{up\_raw} \cdot C_{length}$

#### 4.2.4 深度融合策略 (Depth Fusion)

最终深度 $Z_{est}$ 是两个通道的加权平均：

$$
Z_{est} = \frac{w_{up} \cdot Z_{up} + w_{across} \cdot Z_{across}}{w_{up} + w_{across}}
$$

**权重计算**：
*   **几何权重**：基于投影角度的余弦值绝对值（角度越小，投影越准）。
    *   $w_{up\_base} = |\cos(\text{Pitch})|$
    *   $w_{across} = |\cos(\text{Yaw})|$
*   **握拳修正**：当手部握拳时，长度通道发生形变，不再可靠。
    *   $w_{up} = w_{up\_base} \cdot (1.0 - G)$
    *   当完全握拳 ($G=1$) 时，$w_{up}$ 降为 0，深度估算完全依赖宽度通道。

#### 4.2.5 时序平滑与锚定

##### A. 运动置信度 (Motion Score)
计算深度历史值的标准差 $\sigma$，归一化为运动分数 $M \in [0, 1]$。
*   $M \approx 0$：静止。
*   $M \approx 1$：运动。

##### B. 深度锚定 (Depth Anchor)
当系统处于高置信度状态（展开且正对）时，记录当前的深度值作为“锚点”。
当进入低置信度状态（握拳）时，将锚点值作为额外的观测源注入，权重取决于：
1.  **新鲜度**：随时间指数衰减。
2.  **运动状态**：手越静止，锚点权重越高。
3.  **握拳程度**：握拳越紧，越依赖锚点。

##### C. 自适应卡尔曼滤波 (Adaptive Kalman Filter)
我们使用卡尔曼滤波器对 3D 坐标进行平滑，关键在于**动态观测噪声协方差 ($R$)** 的调整：

$$
R_z = R_{base} + R_{grip\_max} \cdot G \cdot (1 - M)
$$

*   **展开或运动时**：$R_z$ 较小，滤波器信任当前观测值，响应快。
*   **握拳且静止时**：$R_z$ 增大，滤波器信任预测模型，抑制因握拳产生的抖动和漂移。

#### 4.2.6 捏合检测 (Pinch Detection)

通过拇指尖与食指尖的 3D 距离判定捏合手势（`_detect_pinch_raw`）。

1.  **距离计算**：
    将拇指尖 (4) 和食指尖 (8) 的归一化坐标反投影为相机空间 3D 坐标（利用当前帧深度 $Z$），并计算欧氏距离 $d_{pinch}$。

2.  **阈值判定**：
    $$\text{pinch\_raw} = \begin{cases} 1 & d_{pinch} < \text{PINCH\_THRESHOLD\_CM} \\ 0 & \text{otherwise} \end{cases}$$
    其中 `PINCH_THRESHOLD_CM` 默认为 4.0cm。

3.  **去抖 (Debounce)**：
    原始判定需连续维持 `PINCH_DEBOUNCE_FRAMES`（默认 2 帧）后才切换状态，避免临界距离附近的抖动。

#### 4.2.7 滤波策略补充

手部追踪中使用了多层滤波，执行顺序如下：

1.  **关键点 OneEuro**：对 21 个 landmark 的 $(x, y)$ 像素坐标滤波（配置：`FILTER_CONFIG['HAND']['KEYPOINT']`）。
2.  **像素距离 OneEuro**：对两通道的像素距离 $d_{pixel}$ 滤波（配置：`FILTER_CONFIG['HAND']['PIXEL_DIST']`），防止距离突变。
3.  **Z 轴 OneEuro**：对融合后的深度 $Z_{est}$ 滤波（配置：`FILTER_CONFIG['HAND']['DEPTH']`），**在反投影 XY 之前执行**，确保平面坐标稳定性。
4.  **3D 坐标 Kalman**：最终对 $(X, Y, Z)$ 三维坐标进行自适应卡尔曼滤波。

焦距回退逻辑：当相机内参不可用时，使用预计算的 $\tan(\text{fov}/2)$ 从图像宽度估算焦距：$f = \frac{W_{img}/2}{\tan(\text{fov}/2)}$，其中 `tan_half_fov` 在 `__init__` 中预计算。

### 4.3 参数配置

关键参数可在 `config/settings.py` 中调整：

| 参数名 | 说明 | 典型值 |
| :--- | :--- | :--- |
| `HAND_REF_WIDTH_CM` | 手掌横向参考宽度 (cm) | 6.0 |
| `HAND_REF_LENGTH_CM` | 手掌纵向参考长度 (cm) | 9.0 |
| `PINCH_THRESHOLD_CM` | 捏合判定距离阈值 (cm) | 4.0 |
| `PINCH_DEBOUNCE_FRAMES` | 捏合状态切换去抖帧数 | 2 |
| `FILTER_CONFIG['HAND']['KALMAN']['r_grip_max']` | 握拳时最大附加观测噪声 | 1.0 |
| `FILTER_CONFIG['HAND']['KALMAN']['depth_anchor_halflife']` | 锚定值半衰期 (帧) | 45 |
| `FILTER_CONFIG['HAND']['KALMAN']['grip_smoothing_alpha']` | 聚拢系数平滑因子 | 0.3 |
| `FILTER_CONFIG['HAND']['KEYPOINT']` | 关键点 OneEuro 滤波参数 | — |
| `FILTER_CONFIG['HAND']['PIXEL_DIST']` | 像素距离 OneEuro 滤波参数 | — |
| `FILTER_CONFIG['HAND']['DEPTH']` | 深度 Z 轴 OneEuro 滤波参数 | — |

---

# Unity 渲染端 核心算法详解

为了在普通 2D 屏幕上营造 3D 视觉错觉，必须打破标准透视投影的对称性限制，采用**离轴投影 (Off-Axis Projection)** 技术。

## 1. 广义透视投影原理

传统游戏摄像机使用对称视锥体 (Symmetric Frustum)，假设观察者始终位于屏幕中心法线上。而在本项目中，观察者（用户）的位置是实时变化的。

为了呈现正确的透视关系，我们需要构建一个**非对称视锥体 (Asymmetric Frustum)**，其顶点固定在观察者眼睛位置，底面固定在物理屏幕的四个角上。

## 2. 投影矩阵推导

脚本 `VirtualWindowController.cs` 实现了这一算法。核心在于动态计算视锥体的边界 $(l, r, b, t)$ 并引入**切变 (Shear)** 操作。

### 2.1 边界计算
设用户眼睛在屏幕空间的相对坐标为 $(x, y, z)$，近裁剪面距离为 $n$。根据相似三角形：

$$l = (-\frac{W}{2} - x) \cdot \frac{n}{z}, \quad r = (\frac{W}{2} - x) \cdot \frac{n}{z}$$

$$b = (-\frac{H}{2} - y) \cdot \frac{n}{z}, \quad t = (\frac{H}{2} - y) \cdot \frac{n}{z}$$

### 2.2 矩阵构建
标准的透视投影矩阵通常不仅包含缩放，还包含切变项 $P_{0,2}$ 和 $P_{1,2}$，用于实现视锥体的偏斜：

$$
\mathbf{P} = \begin{bmatrix}
\frac{2n}{r-l} & 0 & \frac{r+l}{r-l} & 0 \\
0 & \frac{2n}{t-b} & \frac{t+b}{t-b} & 0 \\
0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

通过实时更新此矩阵，无论用户从哪个角度观察屏幕，虚拟世界的透视线都能与现实世界的视线完美接合，从而产生屏幕后方存在真实空间的错觉。

---

## 3. 仿生眼动控制 (Biometric Eye Movement)

**骨骼初始偏移校准**：
    为了兼容任意模型的骨骼绑定方式（有些模型的眼球骨骼朝向可能不是标准的 Z 轴向前，这可能导致人物出现对眼、白眼等情况，或是因轴向模糊导致眼球贴图自旋等*诡异*情况），脚本在启动时计算当前旋转与标准“正视前方”旋转之间的差异矩阵 $Q_{offset}$：

$$Q_{offset} = Q_{ref}^{-1} \times Q_{current}$$

   以此对角色眼球进行轴向锁定，在后续更新时，总是先计算标准旋转，再叠加此偏移。这排除了镜像骨骼等问题，保证了人物眼球的正确转动。

为了赋予虚拟角色生命力，脚本 `Sight.cs` 模拟了人类眼球运动的微观特征。

*   **前庭动眼反射 (VOR) 逆运算**：
    计算眼球旋转 $Q_{look\_at}$ 以抵消头部运动，保持注视点锁定。

*   **微扫视 (Microsaccades)**：
    引入高频低幅的随机噪声 $Q_{jitter}$，模拟眼部肌肉的生理性震颤，避免死鱼眼现象。

*   **扫视 (Saccades)**：
    模拟注意力的无意识转移，随机地将视线快速移开再复位的瞥视行为，大大增加了人物的真实感。

最终眼球旋转四元数：

$$Q_{final} = Q_{look\_at} \cdot Q_{saccade} \cdot Q_{jitter} \cdot Q_{calibration\_offset}$$
