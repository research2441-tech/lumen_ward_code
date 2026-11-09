"""Manuscript content for Paper 2 (LUMEN-Ward). Result-dependent sections live in build_paper2.py."""

TITLE = ('LUMEN-Ward: Severity-Weighted Information Freshness for Multi-Patient Hybrid Optical–RF Hospital Wards '
         'with a Physics-Structured Generative Channel Twin and Certified Links')
AUTHORS = '[First Author], [Second Author], [Third Author], and [Corresponding Author]'
AFFIL = '[Department, Institution, City, Country]  |  Corresponding author: [email]'

INTRO = [
"Continuous wireless monitoring is moving from intensive care into general wards, where several patients share a room, nurses move between beds and the monitoring budget is set by clinical risk rather than by bed type [@leenen20, rcp17]. Each patient carries a small set of body-worn sensors, and every sensor competes for the same access infrastructure. Optical wireless access suits this setting well. Infrared emitters on the body and photodiodes in the ceiling luminaires avoid radio emission near medical equipment, every bed already has a luminaire above it, and adjacent luminaires can receive at the same time because light stays local [@kahn97, haas16, torkestani12]. Optical medical body sensor networks (MBSNs) have been demonstrated for ICU monitoring and wearable devices [@lebas18, zwaag21, guana24].",
"A ward is not a single link, however. Patients turn in bed and hide a chest patch against the mattress, staff stand between a bed and the ceiling during care, and privacy curtains are drawn around a bed for minutes at a time. Each of these events removes the line of sight of some node–receiver pairs and leaves the rest untouched. A short-range radio link keeps a node reachable when every optical path is shadowed, but the radio channel shared by a whole ward carries few packets per slot. Who transmits, through which receiver, and when the radio fallback is used therefore become ward-level scheduling decisions.",
"Those decisions should also reflect clinical urgency. A patient whose National Early Warning Score 2 (NEWS2) is seven or more needs continuous observation, while a patient scoring zero needs observations every twelve hours [@rcp17, smith13]. Treating every packet as equally urgent spends scarce capacity on stable patients. Weighting by severity alone creates a different hazard: a stable patient whose updates are deprioritised may raise a sudden alarm that then waits behind high-severity traffic. Bedside monitors already raise alarms at high rates [@drew14], so alarm delivery cannot be an afterthought.",
"The age of information (AoI) measures how fresh the latest delivered update is, and it is the natural objective for monitoring traffic [@kaul12, yates21]. Index policies minimise weighted AoI over unreliable channels [@kadota18, hsu20, tripathi19], and urgency-aware variants weight the age by a context-dependent factor [@zheng20]. Hybrid LiFi–WiFi studies balance load and handover between optical and radio access points [@wang15, wu19, wu20], and a recent graph-neural scheduler for hybrid radio–optical Internet-of-Things networks, referred to here as DGET, jointly maximises throughput and minimises delivery AoI [@hamrouni25].",
"These results cannot be transferred to a ward directly, for four reasons. First, existing hybrid optical–RF schedulers treat every packet as equally urgent, so the clinical meaning of the traffic never enters the decision. Second, the optical link state is taken either as a known quantity or as a point estimate, without any statement about how often a scheduled transmission will fail. Third, index policies assume one server and independent channels, whereas ward receivers interfere with each other and a single occluder shadows several links of one node at once. Fourth, a learned scheduler needs many hours of realistic channel experience, and site-specific ray tracing of a whole ward with moving staff, curtains and patients is expensive.",
"The gap, stated precisely, is the absence of a ward-scale scheduler that (i) allocates optical and radio resources according to patient severity, (ii) keeps alarm latency inside a chance constraint, (iii) admits only links whose reliability is certified online, and (iv) is trained on channel experience that preserves the occlusion dynamics of real wards without requiring ray tracing for every training episode.",
"This paper presents LUMEN-Ward, which closes that gap by combining four elements. Each node–receiver link carries an online calibrated certificate, a lower bound on its gain that is violated with a prescribed long-run frequency, following the calibration principle used for single-patient optical links in [@antaki25] and in online conformal control [@gibbs21]. Severity weights are derived from NEWS2 trajectories computed on the open MIMIC-IV and eICU-CRD demonstration databases [@johnson23, pollard18]. A physics-structured generative channel twin keeps the closed-form line-of-sight geometry and learns the parts that cannot be observed: the diffuse field and a coupled Markov occlusion process. Finally, a scorer shared by all receivers adds a learned residual to a severity-weighted Whittle index and is trained by Lagrangian evolution strategies under an alarm-latency chance constraint.",
]

OBJECTIVES = [
"**RO1.** Build a ward-scale optical channel model in which beds, staff, curtains and patient posture produce realistic occlusion dynamics, and determine whether a generative twin trained on part of it can replace ray tracing for scheduler training.",
"**RO2.** Formulate multi-patient hybrid optical–RF scheduling as minimisation of severity-weighted AoI under an alarm-latency chance constraint and per-link reliability certificates.",
"**RO3.** Quantify the freshness, alarm-latency, reliability and energy effects of severity weighting, certificates and learning against equal-urgency and throughput-oriented schedulers, across reporting delays, radio budgets, ward sizes and a second clinical database.",
]

CONTRIB = [
"**A ray-traced multi-patient ward corpus.** Sixty-four procedurally generated wards with six to ten beds, curtains, mobile staff and in-bed and ambulatory patients (CMU motion capture) yield 128 one-minute episodes of 10 ms uplink gains for every node–receiver pair, with subject-disjoint and layout-disjoint splits.",
"**A physics-structured generative channel twin.** The twin keeps the closed-form line of sight and learns a diffuse-field model and a state-dependent occlusion process whose links share a uniform variate per node. It reproduces the shadowing fraction, onset rate and duration of held-out wards, whereas an unconstrained autoregressive mixture-density twin diverges when run freely.",
"**Certificate-gated severity-weighted scheduling.** Receivers poll nodes through an interference-aware greedy matching that admits a link only when its certified signal-to-interference-plus-noise ratio (SINR) clears the decoding threshold. Priorities start from a severity-weighted Whittle index, and a residual scorer shared by all receivers is learned with a Lagrangian alarm-latency constraint.",
"**Evidence on held-out wards.** The evaluation covers six baselines and ablations, three seeds, paired Wilcoxon tests with Holm correction, sweeps of reporting delay, alarm rate, emitter power, radio budget and latency bound, larger unseen wards and severity mixes from a second database.",
]

RELATED = {
"A. Optical and Hybrid Optical–RF Access in Hospitals": [
"Indoor optical wireless channels are built from a Lambertian line of sight and diffuse reflections from walls and ceiling [@kahn97, barry93]. Recursive and integrating-sphere models of the diffuse field, together with reference channel models for VLC, made site-specific optical channel studies routine [@komine04, miramirkhani15]. Hospital deployments use infrared or visible light to keep radio emission away from medical equipment [@torkestani12, lebas18, zwaag21]. Surveys of VLC in e-healthcare and of body-centric propagation describe optical MBSNs as a candidate access layer for monitoring [@guana24, celik22, movassaghi14]. Single-patient studies have shown how device orientation and human blockers shape optical body links [@soltani19, pal24], and recent work optimised modulation and channel estimation for one optical MBSN at a time [@antaki25].",
"Hybrid access combines optical cells with a radio overlay. Analyses of hybrid RF/VLC systems showed that the radio tier absorbs users that the optical tier cannot serve [@basnayaka17]. Load balancing with handover between LiFi and WiFi was then extended to user mobility and light-path blockage [@wang15, wu19, wu20]. DCO-OFDM clipping and noise models fix the SINR that each optical cell can reach [@dimitrov12, haas16]. DGET applies graph neural networks with multi-task learning to dual-technology scheduling in radio–optical IoT networks. It jointly maximises throughput and minimises delivery AoI under energy and availability constraints, imitating an optimisation solver [@hamrouni25]. None of these works weights traffic by clinical urgency, bounds the failure probability of a scheduled link, or treats an alarm-latency requirement as a constraint.",
],
"B. Freshness-Aware and Urgency-Aware Scheduling": [
"The AoI concept and its queueing theory were established for status-update systems [@kaul12, sun17] and later surveyed in depth [@yates21]. For wireless broadcast networks with unreliable links, max-weight and Whittle-index policies come close to the optimal weighted AoI. Variants handle throughput constraints and random arrivals [@kadota18, kadota19, hsu20]. The Whittle approach extends to general functions of age [@tripathi19, whittle88], and tail-aware formulations control the probability that age exceeds a bound [@abdelaziz20]. Urgency of information weights the age by a context-dependent factor [@zheng20], which is the closest existing notion to severity weighting. These policies assume a single server or independent channels and treat link reliability as known. In a ward, receivers interfere with each other, occlusion is correlated across links, and the reliability of a link has to be estimated online.",
],
"C. Clinical Severity and Monitoring Traffic": [
"Track-and-trigger systems turn routine vital signs into a risk score. NEWS2 standardises the score and the clinical response across the National Health Service, and its predecessor discriminated patients at risk of cardiac arrest, unanticipated ICU admission and death within 24 h [@rcp17, smith13]. Continuous wearable monitoring on general wards is technically feasible, but the clinical evidence is still mostly from validation and feasibility studies [@leenen20]. Physiological monitors in intensive care already produce very large numbers of alarms, most of them non-actionable, so an alarm channel has to be reliable and timely without assuming that every alarm is rare [@drew14]. Open de-identified critical-care databases make severity trajectories reproducible [@johnson23, pollard18]. None of these studies links the risk score to the scheduling of the network that carries the monitoring data.",
],
"D. Learning-Based Multi-Agent Radio Resource Management": [
"Multi-agent deep reinforcement learning has been used for power control and for joint user scheduling and power allocation [@nasir19, naderializadeh21]. Proximal policy optimisation with shared parameters is a strong cooperative baseline [@yu22]. Graph neural networks provide permutation-equivariant schedulers that scale with network size [@shen21, eisen20]. Constrained Markov decision processes and reward-constrained policy optimisation handle long-run constraints through Lagrange multipliers [@altman99, tessler19], and drift-plus-penalty methods give the queueing counterpart [@neely10]. Evolution strategies estimate policy gradients from returns alone and parallelise well when the environment is not differentiable [@salimans17]. In medical wards, these learners have not been combined with clinical severity, and their training has relied on simplified channel models.",
"Calibration of wireless predictors has recently been formalised with conformal methods [@cohen23, angelopoulos23b]. Online variants keep long-run coverage under distribution shift by adjusting a quantile from observed errors [@gibbs21]. The certificates used here apply this mechanism per link to lower bounds on optical gain, so that a scheduler can admit a link only when its failure frequency is controlled.",
],
"E. Generative Channel Models and Digital Twins": [
"Generative adversarial and variational models learn stochastic channel distributions from data [@yang19], and conditional generative networks reproduce path-loss and multipath statistics for millimetre-wave links [@xia22]. Mixture-density networks provide explicit likelihoods for multimodal targets [@bishop94]. Differentiable ray tracers and real-time digital twins aim to mirror a deployment site closely enough to train and test network control [@hoydis23, alkhateeb23]. Vision-aided blockage prediction showed that occlusion is the dominant event at high frequencies [@charan21]. For body-area radio, measured models describe on-body and off-body shadowing [@smithd13]. Optical ward channels have no generative twin yet. The structure of the ward channel is unusual: the line-of-sight geometry is observable from node pose and luminaire positions, while the occluders that matter, staff and curtains, are not. The twin proposed here exploits this split.",
"Motion-capture data from the CMU database drive the ambulatory patients [@cmu], and paired Wilcoxon tests with Holm correction are used for statistical comparison [@wilcoxon45, holm79].",
],
}

RW_TABLE = [
["Work", "Setting", "Links", "Objective", "Severity / alarms", "Channel knowledge"],
["[@antaki25]", "One optical MBSN, hospital rooms", "VLC, ray-traced", "Adaptive modulation", "No / no", "LSTM path-loss estimate"],
["[@wu20]", "Hybrid LiFi–WiFi, mobile users", "Optical + RF", "Load balancing, handover", "No / no", "Blockage statistics"],
["[@hamrouni25]", "Hybrid radio–optical IoT", "OWC + RF", "Throughput and delivery AoI", "No / no", "Observed link state"],
["[@kadota18]", "Broadcast wireless network", "Single server", "Weighted AoI", "Static weights / no", "Known success probability"],
["[@zheng20]", "Remote control systems", "Shared channel", "Urgency of information", "Context weight / no", "Known"],
["LUMEN-Ward", "Multi-patient ward, staff, curtains", "IR uplink + RF", "Severity-weighted AoI", "NEWS2 / chance constraint", "Online certificates, generative twin"],
]

NOTATION = [
["Symbol", "Meaning"],
["$\\mathcal{N},\\ \\mathcal{L}$", "body nodes; ceiling infrared receivers"],
["$x_{n\\ell,t}$", "uplink gain of node $n$ at receiver $\\ell$ in slot $t$ (dB)"],
["$g^{0}_{n\\ell,t}$", "closed-form unobstructed line-of-sight gain (dB)"],
["$B_{n\\ell,t}$", "occlusion state (1 shadowed, 0 clear)"],
["$c_{n\\ell,t},\\ q_{n\\ell,t}$", "certified gain and certificate margin (dB)"],
["$\\Gamma,\\ \\Gamma_{\\mathrm{RF}}$", "optical and radio decoding thresholds"],
["$w_n$", "NEWS2 severity weight of the patient wearing node $n$"],
["$a_{n,t}$", "age of information of node $n$ (slots)"],
["$L_{\\max},\\ \\delta$", "alarm-latency bound and allowed violation probability"],
["$D$", "channel-reporting delay (slots)"],
["$\\theta,\\ \\lambda$", "shared scorer parameters; Lagrange multiplier"],
]

# ----------------------------------------------------------------------------------- methodology
METHOD = {
'A': ("System Model", [
('p', "A ward contains $P$ patients, each wearing three nodes: a chest ECG patch, a wrist pulse oximeter and an upper-arm cuff with a thermometer. The set of nodes is $\\mathcal{N}$ with $|\\mathcal{N}|=3P$. A subset $\\mathcal{L}$ of the ceiling luminaires carries infrared photodiodes, arranged as a checkerboard over the luminaire grid, and one ceiling radio access point serves the whole ward. Time is slotted with a slot of 10 ms. In each slot every receiver $\\ell\\in\\mathcal{L}$ polls at most one node, a node transmits through at most one receiver, and $K_{\\mathrm{RF}}$ radio slots are shared by all nodes that the optical tier does not serve."),
('p', "A node with position $\\mathbf{r}_n$ and unit normal $\\mathbf{n}_n$ emits with Lambertian order $m$. Receiver $\\ell$ at $\\mathbf{s}_{\\ell}$ has area $A$, field of view $\\Psi$ and concentrator gain $g_c$. The unobstructed line-of-sight gain is"),
('eq', "g^{0}_{n\\ell}=\\frac{(m+1)A\\,g_c}{2\\pi d_{n\\ell}^{2}}\\cos^{m}\\phi_{n\\ell}\\cos\\psi_{n\\ell}\\,\\mathbf{1}\\{\\psi_{n\\ell}\\le\\Psi\\}"),
('p', "where $d_{n\\ell}=\\|\\mathbf{s}_{\\ell}-\\mathbf{r}_n\\|$, $\\cos\\phi_{n\\ell}=\\mathbf{n}_n^{\\top}(\\mathbf{s}_{\\ell}-\\mathbf{r}_n)/d_{n\\ell}$ and $\\psi_{n\\ell}$ is the incidence angle at the downward-facing receiver. The received gain adds first-order diffuse reflections from discretised room surfaces and the integrating-sphere tail $\\eta$ [@barry93, komine04], and loses the line of sight when an occluder crosses it:"),
('eq', "G_{n\\ell,t}=\\bigl(1-B_{n\\ell,t}\\bigr)G^{0}_{n\\ell,t}+\\sum_{j}G^{(1)}_{n j\\ell,t}+\\eta"),
('p', "with $B_{n\\ell,t}=1$ when a body, a staff member or a curtain intersects the segment, and $x_{n\\ell,t}=10\\log_{10}G_{n\\ell,t}$. The uplink is DCO-OFDM with clipping factor $K$ and clipping-noise variance $\\sigma_c^2$ [@dimitrov12]. With emitter optical power $P_{\\mathrm{o}}$, responsivity $R$ and receiver noise $N_0$, the SINR of node $n$ at receiver $\\ell$ when the set $\\mathcal{S}_t$ transmits is"),
('eq', "\\gamma_{n\\ell,t}=\\frac{\\kappa_{s}G_{n\\ell,t}^{2}}{N_0+\\kappa_{c}G_{n\\ell,t}^{2}+\\sum_{j\\in\\mathcal{S}_t\\setminus\\{n\\}}\\kappa_{s}G_{j\\ell,t}^{2}}"),
('p', "where $\\kappa_s=(RKP_{\\mathrm{o}}/\\kappa)^2$, $\\kappa_c=(RP_{\\mathrm{o}}/\\kappa)^2\\sigma_c^2$ and $\\kappa$ is the clipping level in standard deviations. A packet is decoded when $\\gamma_{n\\ell,t}\\ge\\Gamma$, the 4-QAM threshold for a symbol error rate of $10^{-3}$. On the radio tier, log-distance path loss, own-body and staff shadowing and a slow log-normal term give the mean SNR $\\bar{\\rho}_{n,t}$ [@smithd13]; under Rayleigh fading the success probability of one radio slot is"),
('eq', "p^{\\mathrm{RF}}_{n,t}=\\exp\\!\\bigl(-\\Gamma_{\\mathrm{RF}}/\\bar{\\rho}_{n,t}\\bigr)"),
]),
'B': ("Clinical Severity, Freshness and Alarms", [
('p', "Each patient carries the NEWS2 score of a patient-hour drawn from the clinical databases [@rcp17]. NEWS2 maps the score to a clinical-response tier $k$ with a minimum observation interval $\\tau_k$ of 12 h (score 0), 5 h (scores 1–4, the midpoint of the 4–6 h guidance), 1 h (a single parameter scoring 3, or a total of 5–6) and 0.25 h for a total of 7 or more, where the guideline asks for continuous monitoring. The severity weight is the guideline observation rate"),
('eq', "w_n=1/\\tau_{k(n)}\\ \\mathrm{h}^{-1}"),
('p', "so the weights span a factor of 48. Each node generates a fresh sample whenever it is polled, and its age evolves as"),
('eq', "a_{n,t+1}=\\begin{cases}1,& \\text{if node } n \\text{ delivers in slot } t,\\\\ a_{n,t}+1,& \\text{otherwise.}\\end{cases}"),
('p', "The freshness objective is the severity-weighted average AoI"),
('eq', "\\bar{A}_{w}=\\lim_{T\\to\\infty}\\frac{1}{T}\\sum_{t=1}^{T}\\frac{\\sum_{n}w_n a_{n,t}}{\\sum_{n}w_n}"),
('p', "Alarms arrive at each patient as a Bernoulli process with rate $\\lambda_a$ per second and attach to one of the patient's nodes. An alarm raised in slot $t_0$ is delivered with the node's next successful packet, so its latency is $\\ell=t_1-t_0+1$, where $t_1$ is the first successful slot after $t_0$. The ward must satisfy the chance constraint"),
('eq', "\\Pr\\{\\ell>L_{\\max}\\}\\le\\delta"),
('p', "The scheduling problem is a constrained Markov decision process [@altman99]: find a policy $\\pi$ that maps the observed state to receiver assignments and radio slots so as to"),
('eq', "\\begin{aligned}\\min_{\\pi}\\ &\\bar{A}_{w}(\\pi)\\\\ \\text{s.t.}\\ &\\Pr_{\\pi}\\{\\ell>L_{\\max}\\}\\le\\delta,\\\\ &\\Pr\\{\\gamma_{n\\ell,t}<\\Gamma\\mid (n,\\ell)\\in\\mathcal{S}_t\\}\\lesssim\\alpha\\end{aligned}"),
('p', "The last condition limits the frequency of failed optical transmissions to roughly $\\alpha$, which saves emitter energy and keeps retransmissions from delaying alarms."),
]),
'C': ("Online Link Certificates", [
('p', "The controller observes link gains through beacons with a delay of $D$ slots. The forecast $\\hat{x}_{n\\ell,t}=x_{n\\ell,t-D}$ is combined with a per-link margin $q_{n\\ell,t}$ to form the certified gain $c_{n\\ell,t}=\\hat{x}_{n\\ell,t}-q_{n\\ell,t}$. The margin follows the online quantile update of adaptive conformal inference [@gibbs21]:"),
('eq', "q_{n\\ell,t+1}=\\left[q_{n\\ell,t}+\\eta\\bigl(\\mathbf{1}\\{x_{n\\ell,t}<c_{n\\ell,t}\\}-\\alpha\\bigr)\\right]_{0}^{q_{\\max}}"),
('p', "For any gain sequence, the frequency of $x<c$ on each link exceeds $\\alpha$ by at most $q_{\\max}/(\\eta T)$ (Proposition 1 in Section IV-D), the same type of guarantee used for single-node link control. A pair $(n,\\ell)$ is admissible when its certified SINR clears the threshold given the interference already committed at $\\ell$, and adding $n$ must not break any receiver already committed:"),
('eq', "\\begin{aligned}&\\frac{\\kappa_s C_{n\\ell}^{2}}{N_0+\\kappa_c C_{n\\ell}^{2}+I_{\\ell}}\\ge\\Gamma,\\\\ &\\kappa_s\\hat{G}_{n\\ell'}^{2}\\le \\frac{\\kappa_s C_{j\\ell'}^{2}}{\\Gamma}-N_0-\\kappa_c C_{j\\ell'}^{2}-I_{\\ell'}\\ \\ \\forall \\ell' \\text{ committed}\\end{aligned}"),
('p', "Here $C=10^{c/10}$, $\\hat{G}=10^{\\hat{x}/10}$, $j$ is the node already assigned to $\\ell'$ and $I_{\\ell}$ is the interference that the committed transmitters create at $\\ell$, estimated from $\\hat{x}$."),
]),
'D': ("Physics-Structured Generative Channel Twin", [
('p', "Ray tracing every training episode of a ward with moving staff and curtains is expensive, so the scheduler is trained on a twin. The twin keeps what the scheduler can observe and generates what it cannot. Node poses follow from the body-worn inertial sensors and the bed geometry, so $g^{0}$ in (1) is computed exactly. Staff positions, curtain states and other patients' bodies are not observed, so the twin generates the occlusion state $B$. The diffuse level depends on room reflectances that are rarely known, so it is learned as a distribution. The generated gain is"),
('eq', "\\tilde{x}_{n\\ell,t}=10\\log_{10}\\left(\\bigl(1-\\tilde{B}_{n\\ell,t}\\bigr)10^{g^{0}_{n\\ell,t}/10}+10^{\\tilde{d}_{n\\ell,t}/10}\\right)"),
('p', "The diffuse level is Gaussian with mean $\\mu_d$ and spread $\\sigma_d$ given by a multilayer perceptron (MLP) of geometry features $\\mathbf{f}^{d}$ (horizontal distance, incidence cosine, node height and normal, room size and node type), with a slow AR(1) fluctuation:"),
('eq', "\\tilde{d}_{n\\ell,t}=\\mu_d(\\mathbf{f}^{d}_{n\\ell,t})+\\sigma_d(\\mathbf{f}^{d}_{n\\ell,t})\\,\\epsilon_{n\\ell,t},\\quad \\epsilon_{t}=\\varrho\\,\\epsilon_{t-1}+\\sqrt{1-\\varrho^{2}}\\,\\xi_t"),
('p', "with $\\varrho=e^{-1/100}$ (a 1 s correlation time) and $\\xi_t\\sim\\mathcal{N}(0,1)$. The occlusion state of every visible link is a Markov process whose probability of being shadowed in the next slot is"),
('eq', "\\pi_{n\\ell,t}=\\sigma\\!\\left(h_{o}\\bigl(\\mathbf{f}^{o}_{n\\ell,t},\\ \\tilde{B}_{n\\ell,t-1},\\ \\log(1+r_{n\\ell,t-1}),\\ \\bar{B}^{-\\ell}_{n,t-1}\\bigr)\\right)"),
('p', "where $h_o$ is an MLP, $\\mathbf{f}^{o}$ contains the unobstructed gain, its change and the geometry, $r$ is the length of the current state run and $\\bar{B}^{-\\ell}_{n}$ is the shadowed fraction of the node's other visible links. A single uniform variate $u_{n,t}$ per node and slot drives all links of that node:"),
('eq', "\\tilde{B}_{n\\ell,t}=\\mathbf{1}\\{u_{n,t}<\\pi_{n\\ell,t}\\},\\qquad u_{n,t}\\sim\\mathcal{U}(0,1)"),
('p', "One occluder near a node therefore tends to shadow every receiver it lies in front of. The twin is fitted on teacher-forced samples from the training wards. The occlusion head minimises the binary cross-entropy against the ray-traced states, and the diffuse head minimises the Gaussian negative log-likelihood on slots in which the line of sight is absent or shadowed:"),
('eq', "\\begin{aligned}\\mathcal{J}_{\\mathrm{twin}}=&-\\sum\\left[B\\log\\pi+(1-B)\\log(1-\\pi)\\right]\\\\ &+\\sum\\left[\\frac{(x-\\mu_d)^{2}}{2\\sigma_d^{2}}+\\log\\sigma_d\\right]\\end{aligned}"),
('p', "Because every generated gain lies between the diffuse floor and the unobstructed value, the twin cannot drift, which unconstrained autoregressive generators do (Section VI-A)."),
]),
'E': ("Severity-Weighted Index and Learned Residual", [
('p', "For a link with success probability $p$, the Whittle index of weighted AoI is $W=\\tfrac{1}{2}w\\,p\\,a\\,(a+2/p-1)$ [@kadota18, tripathi19]. A certified optical link has $p\\approx1$, so its base score is $\\log w_n+\\log a_n+\\log(a_n+1)$. The radio tier uses the full index with $p=p^{\\mathrm{RF}}_{n,t}$. Every receiver applies the same scorer to its candidate nodes:"),
('eq', "\\begin{aligned}s_{n\\ell,t}=&\\log W_{n,t}+\\epsilon_m\\,m_{n\\ell,t}+f_{\\theta}(\\mathbf{z}_{n\\ell,t})\\\\ &+\\beta_{\\theta}\\,\\mathbf{1}\\{\\text{alarm}\\}\\left(1+\\omega_{n,t}/L_{\\max}\\right)\\end{aligned}"),
('p', "where $m_{n\\ell,t}$ is the certified SINR margin in dB, $\\epsilon_m=0.05$ breaks ties toward the strongest receiver, $\\omega_{n,t}$ is the waiting time of a pending alarm and $f_\\theta$ is a two-layer MLP. Its input $\\mathbf{z}$ holds the log weight, log age, alarm flag and waiting time, certified margin, the margin relative to the node's best receiver, the gain trend over 100 ms, the number of receivers that certify the node, the radio success probability, a flag for the receiver that served the node last and the node's index relative to the ward mean. A second MLP $f'_{\\theta}$ with node-level inputs adds a residual to the radio index. The output layers of $f_\\theta$ and $f'_\\theta$ start at zero, so training starts from the analytic index policy. Pairs are committed greedily in descending score subject to (11), and the $K_{\\mathrm{RF}}$ best remaining nodes with $p^{\\mathrm{RF}}>0.3$ receive the radio slots."),
('p', "Since $\\theta$ is shared, the scorer runs as identical agents on all receivers with a common commitment order, and it does not depend on the number of beds or receivers. Training minimises the Lagrangian"),
('eq', "\\mathcal{L}(\\theta,\\lambda)=\\mathbb{E}\\left[\\frac{\\bar{A}_{w}(\\theta)}{\\bar{A}_{w}(\\pi_0)}\\right]+\\lambda\\left(\\Pr_{\\theta}\\{\\ell>L_{\\max}\\}-\\delta\\right)"),
('p', "where $\\pi_0$ is the analytic index policy evaluated on the same windows, which normalises away the difficulty of each ward. The greedy matching is not differentiable, so the gradient is estimated by antithetic evolution strategies [@salimans17] with centred ranks $\\varphi(\\cdot)$ of the $M$ perturbed fitness values:"),
('eq', "\\hat{\\nabla}_{\\theta}\\mathcal{L}=\\frac{1}{M\\sigma}\\sum_{i=1}^{M}\\varphi\\bigl(\\mathcal{L}(\\theta+\\sigma\\boldsymbol{\\varepsilon}_i,\\lambda)\\bigr)\\boldsymbol{\\varepsilon}_i,\\quad \\boldsymbol{\\varepsilon}_{i+M/2}=-\\boldsymbol{\\varepsilon}_i"),
('p', "followed by an Adam step on $\\theta$ and projected dual ascent on the multiplier:"),
('eq', "\\lambda_{k+1}=\\left[\\lambda_k+\\eta_{\\lambda}\\bigl(\\widehat{\\Pr}_{\\theta_k}\\{\\ell>L_{\\max}\\}-\\delta\\bigr)\\right]_{+}"),
('p', "All candidates of one iteration share the same windows, severity draws and random numbers (common random numbers), which reduces the variance of the rank comparison. The parameters with the best validation score, $\\bar{A}_w$ ratio plus a penalty of ten times any excess violation, are retained."),
]),
'F': ("Properties of the Scheduler and the Twin", [
('p', "Three properties follow directly from the construction and make the guarantees of LUMEN-Ward explicit. The first concerns the certificates. Write $v_{n\\ell,t}=\\mathbf{1}\\{x_{n\\ell,t}<c_{n\\ell,t}\\}$ and sum the update (10) over $t=1,\\dots,T$. The margin telescopes; the clip at zero only adds margin, and the clip at $q_{\\max}$ does not bind as long as $q_{\\max}$ exceeds the dynamic range of the gain. Hence, for every gain sequence,"),
('eq', "\\frac{1}{T}\\sum_{t=1}^{T}v_{n\\ell,t}\\le\\alpha+\\frac{q_{\\max}-q_{n\\ell,1}}{\\eta\\,T}"),
('p', "**Proposition 1.** On every link, the long-run frequency with which the realised gain falls below its certificate exceeds $\\alpha$ by at most $O(1/T)$, without any assumption on the channel process, the reporting delay or the scheduling decisions. With $q_{\\max}=40$ dB and $\\eta=0.05$ the excess is below 0.23 percentage points after one hour of slots. The frequency can fall below $\\alpha$ when the channel is steadier than its reports suggest, which only makes the certificate conservative. The guarantee is marginal over time, and it covers the shadowing onsets that a delayed report cannot anticipate."),
('p', "The second property concerns the matching. A committed pair is never broken by a later commitment, because condition (11) checks the slack of every receiver already in use before a new transmitter is added. At the end of each slot every committed pair therefore satisfies"),
('eq', "\\frac{\\kappa_s C_{n\\ell}^{2}}{N_0+\\kappa_c C_{n\\ell}^{2}+\\sum_{j\\in\\mathcal{S}_t\\setminus\\{n\\}}\\kappa_s\\hat{G}_{j\\ell}^{2}}\\ge\\Gamma\\quad\\forall (n,\\ell)\\in\\mathcal{S}_t"),
('p', "**Proposition 2.** A committed optical transmission can fail only if its own gain falls below the certificate or the interference exceeds its reported value. By the union bound, its failure probability is at most $\\alpha$ plus the probability that some interferer's gain at $\\ell$ has risen since its report. The second term is small because interfering links are mostly diffuse or distant, and Section VI confirms that the failure rate stays below $\\alpha$ at every reporting delay tested (0.09% at $D=1$ and 0.67% at $D=20$, against 2.4% and 12.3% for uncertified MW-AoI)."),
('p', "The third property concerns the twin. Because $\\tilde{B}\\in\\{0,1\\}$, every generated gain is confined to"),
('eq', "\\tilde{d}_{n\\ell,t}\\le\\tilde{x}_{n\\ell,t}\\le10\\log_{10}\\left(10^{g^{0}_{n\\ell,t}/10}+10^{\\tilde{d}_{n\\ell,t}/10}\\right)"),
('p', "and the diffuse term is a stationary AR(1) process around a geometry-dependent mean, so free-running generation cannot drift. Finally, the complexity of one slot decision is dominated by at most $|\\mathcal{L}|$ rounds of the matching, each of which scans every node–receiver pair and checks the committed receivers:"),
('eq', "\\mathcal{C}_{\\mathrm{slot}}=O\\left(|\\mathcal{N}|\\,|\\mathcal{L}|^{2}+|\\mathcal{N}|\\,|\\mathcal{L}|\\,H\\,(F+1)\\right)"),
('p', "where $H=16$ is the hidden width of the scorer and $F=11$ its input dimension. For the largest wards ($|\\mathcal{N}|=30$, $|\\mathcal{L}|=9$) this amounts to a few tens of thousands of operations per slot, so a luminaire controller can run it easily."),
]),
}

ALGO1 = ("Algorithm 1: Twin episode generation", [
"Input: ward geometry, node poses $\\{\\mathbf{r}_{n,t},\\mathbf{n}_{n,t}\\}$, twin heads $\\mu_d,\\sigma_d,h_o$",
"Compute $g^{0}_{n\\ell,t}$ by (1) for all links and slots; set $\\tilde{B}_{\\cdot,0}=0$, $r=0$, $\\epsilon_0\\sim\\mathcal{N}(0,1)$",
"for $t=1,\\dots,T$ do",
("  ", "Compute $\\pi_{n\\ell,t}$ by (14) from geometry, state, run length and node coupling"),
("  ", "Draw one $u_{n,t}$ per node; set $\\tilde{B}_{n\\ell,t}$ by (15) on visible links"),
("  ", "Update run lengths; draw $\\tilde{d}_{n\\ell,t}$ by (13); synthesise $\\tilde{x}_{n\\ell,t}$ by (12)"),
"end for",
"Output: gain tensor $\\tilde{x}\\in\\mathbb{R}^{T\\times|\\mathcal{N}|\\times|\\mathcal{L}|}$",
])

ALGO2 = ("Algorithm 2: LUMEN-Ward slot decision", [
"Input: delayed gains $\\hat{x}_{t}$, margins $q_t$, ages $a_t$, weights $w$, pending alarms, $\\theta$",
"Certified gains $c=\\hat{x}-q$; scores $s_{n\\ell,t}$ by (17) for admissible pairs",
"Initialise committed set $\\mathcal{S}=\\emptyset$, interference $I_\\ell=0$",
"repeat",
("  ", "Select the highest-scoring pair $(n,\\ell)$ with free $n$ and $\\ell$ that satisfies (11)"),
("  ", "Commit $(n,\\ell)$; update $I_{\\ell'}$ for all $\\ell'\\neq\\ell$ and the slack of committed receivers"),
"until no admissible pair remains",
"Assign the $K_{\\mathrm{RF}}$ radio slots to the uncommitted nodes with the largest radio scores",
"Transmit; update ages by (6), alarm latencies and margins by (10)",
])
