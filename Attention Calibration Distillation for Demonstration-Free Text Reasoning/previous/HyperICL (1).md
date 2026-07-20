# **Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

**Niloufar Alipour Talemi**<sup>1</sup> **Hossein Kashiani**<sup>1</sup> **Fatemeh Afghah**<sup>1</sup> 

## **Abstract** 

Multimodal In-Context Learning (ICL) has emerged as a practical inference paradigm for Multimodal Large Language Models, where a small set of interleaved image-text In-Context Demonstrations (ICDs) conditions the model to solve new tasks. Despite its flexibility, multimodal ICL incurs high inference latency and suffers from instability due to sensitivity to demonstration formatting, ordering, and content. To address these limitations, we propose HyperICL, a lightweight, training-based framework for demonstration-free multimodal ICL that reconstructs demonstration effects directly without requiring ICDs at inference time. Hyper-ICL learns a parameter-efficient low-rank logit-level adapter that calibrates attention distributions to better match demonstration-induced attention redistribution. To capture how demonstration influence varies across queries, we introduce a queryadaptive modulation mechanism that adaptively controls intervention strength at token level across layers and heads based on the current query. Finally, we propose a layer-wise hyperbolic anchor distillation loss that aligns intermediate student features to a demonstration-conditioned teacher via Lorentz geodesic distance. This loss encourages the student to reconstruct the demonstration–query relationships induced by ICDs. Extensive experiments across six different multimodal benchmarks (including VQAv2, OK-VQA, and COCO Caption) demonstrate that Hyper-ICL consistently improves accuracy and stability over vanilla ICL and existing state-of-the-art methods. 

1Holcombe Department of Electrical and Computer Engineering, Clemson University, Clemson, United States. Correspondence to: Niloufar Alipour Talemi _<_ nalipou@clemson.edu _>_ . 

_Proceedings of the 43_<sup>_rd_</sup> _International Conference on Machine Learning_ , Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s). 

## **1. Introduction** 

Multimodal Large Language Models (MLLMs) (Zhou et al., 2022; Li et al., 2023; Liu et al., 2023) have recently become a strong paradigm for unified vision-language understanding and generation, enabling a single model to solve diverse tasks such as Visual Question Answering (VQA) (Shao et al., 2023; Goyal et al., 2017), image captioning (Chen et al., 2015) and visual reasoning (Xu et al., 2026) with minimal task-specific adaptation. A particularly attractive adaptation mechanism is In-Context Learning (ICL) (Liu et al., 2022), where the model is steered at inference time by prefixing a small set of In-Context Demonstrations (ICDs) into the prompt (Dong et al., 2024). Compared to parameter updating, ICL offers rapid and flexible adaptation, making it a practical interface for deploying MLLMs across diverse downstream applications. 

Despite recent progress in ICL, multimodal ICL still faces several practical and fundamental limitations. Compared with text-only ICL in Large Language Models (LLMs), MLLMs require interleaved image–text demonstrations that substantially increase the effective input length, leading to higher inference latency. Beyond this token overhead, recent studies (Li et al., 2024b; Qin et al., 2024) show that MLLMs can overfit to superficial ICD cues, leading to brittle behavior under formatting and ordering changes (Sun et al., 2025; Zhang et al., 2023; Baldassini et al., 2024). For instance, in VQA, MLLM may mimic the answer format observed in the demonstrations rather than learning the underlying input–output mapping, in contrast to behavior more commonly seen in language-only QA. These instabilities are exacerbated in multimodal contexts, where the inherent complexity of aligning cross-modal features introduces additional layers of variance. 

Recent work has shifted toward vector-based ICL to mitigate these overheads and reproduce in-context behaviors without long, token-heavy prompts. Rather than conditioning on raw ICDs, these methods extract a compact task representation from demonstrations and inject it into the model as an additive shift in the hidden activations, reframing ICL as the induction of an internal task representation by the input context (Brown et al., 2020; Todd et al., 2023). Compressing multiple demonstrations into a single In-Context Vector (ICV) shortens the inference-time prompt and reduces re- 

1 

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

liance on carefully selecting and ordering ICDs. However, heuristic ICV extraction often fails on complex multimodal tasks like VQA, which require richer cross-modal reasoning and tighter visual-language alignment than language-only scenarios. 

To address these challenges, a recent study, LIVE (Peng et al., 2024), proposes a training-based approach that learns a rich shift vector from a large supporting set. LIVE distills task-relevant information from many randomly sampled ICD sets into layer-wise ICVs, achieving superior performance over heuristic methods. However, it remains limited in modeling the complex interactions inherent in multimodal ICL. As shown by both empirical results and theoretical analysis in Section 3.1, not only layers but also attention heads play distinct roles in processing demonstrations, which is an aspect LIVE overlooks. Moreover, to fully exploit crossdomain ICD patterns, such information should be injected into attention logits rather than final-layer hidden states. Lastly, LIVE applies a uniform adaptation across inputs, which can degrade performance when queries require different degrees of adjustment. 

Building on these insights, we propose Hyper-ICL, a lightweight, training-based framework that decomposes and reconstructs multimodal in-context effects directly within the attention mechanism (shown in Figure 1). Instead of injecting a uniform hidden-state shift, Hyper-ICL learns a lightweight low-rank logit-level adapter that steers where the model attends in a way that better matches the attention patterns induced by ICDs. To accommodate the query-adaptive nature of ICL, we further introduce a query-adaptive modulation mechanism that dynamically controls intervention strength across layers and heads, ensuring query-adaptive calibration while suppressing unnecessary interference. Finally, we propose a layer-wise hyperbolic anchor distillation loss that aligns intermediate student representations with a demonstration-conditioned teacher via hyperbolic geodesic distance, allowing the student to recover both highlevel guidance and fine-grained contextual refinements even without access to demonstrations at inference time. Hyperbolic geometry is particularly well suited to this setting, as multimodal ICL induces a dense set of demonstration–query relationships that must remain coherent across intermediate layers. Prior work motivates negative curvature as offering effective representational capacity and mitigate distortion in low-dimensional embeddings (Fish & Bowden, 2025; Ibrahimi et al., 2024). This helps preserve relative similarity ordering when many interactions are embedded in a lowdimensional embedding (Law et al., 2019). In our work, the teacher encodes these demonstration-conditioned relations, whereas the student observes only the query. Aligning their intermediate states with Lorentz geodesic distance, therefore, encourages the student to match the teacher’s multiscale relational geometry, rather than relying on Euclidean proximity that can permute relative distance structure (Law 

et al., 2019; Poppi et al., 2025). In summary, our main contributions are: 

- We introduce **Hyper-ICL** , an efficient multimodal ICL framework that decomposes demonstration effects within self-attention and proposes a logit-level attention intervention that directly calibrates attention distributions rather than approximating demonstration-induced shifts only at the output level. 

- We propose a layer-wise hyperbolic anchor distillation loss that aligns intermediate student features to a demonstration-conditioned teacher via Lorentz geodesic distance, helping preserve relative similarity ordering under dense demonstration–query relationships and enabling demonstration-free inference. 

- Experiments on two large-scale MLLMs across six widely used, challenging benchmarks (including VQAv2 (Goyal et al., 2017), OK-VQA (Marino et al., 2019), and COCO Caption (Chen et al., 2015)) show that Hyper-ICL consistently improves performance and stability over direct ICD prompting, vector-based ICV baselines, and trainingbased alternatives. 

## **2. Related Work** 

### **2.1. Multimodal Large Language Models** 

With advances in LLMs, MLLMs have emerged as a unified framework for joint vision–language understanding and generation (Liu et al., 2023; Alipour Talemi et al., 2025; Chen et al., 2024). A widely adopted formulation augments a pretrained LLM with a vision encoder and a lightweight alignment module, such as a projection layer or a querybased transformer, to map visual features into the LLM embedding space for multimodal reasoning (Liu et al., 2023). Idefics-9b (Laurenc¸on et al., 2023) follows a cross-attention based architecture, where language features attend to image representations through dedicated attention blocks for multimodal integration. In contrast, Idefics2-8b (Laurenc¸on et al., 2024) adopts a fully autoregressive design by converting visual features into token-like embeddings and processing them jointly with text via self-attention. A widely adopted strategy for adapting MLLMs to new tasks is ICL, where task behavior is induced by conditioning on a few multimodal demonstrations at inference. Despite its simplicity, multimodal ICL is less stable than language-only ICL, as cross-modal interactions increase sensitivity to demonstration choice, order, and formatting (Ma et al., 2025). 

### **2.2. In-Context Learning** 

ICL enables rapid task adaptation by conditioning on demonstrations at inference time, without updating model parameters. However, in multimodal ICL, each image introduces substantial token overhead, making high-shot prompting expensive and highly sensitive to demonstration choice and formatting (Doveh et al., 2024; Li et al., 2026; Baldassini et al., 2024; Li et al., 2025c). To reduce this cost, exist- 

2 

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

ing approaches develop efficient ICL mechanisms such as vector-based compression that distills demonstration effects into compact ICVs injected into intermediate activations, as well as lightweight activation-shift strategies that approximate ICL through additive shifts estimated from attention statistics or layerwise representations (Brown et al., 2020; Todd et al., 2023). LIVE (Peng et al., 2024) follows a related direction by learning task-specific, layer-wise ICVs via distillation from demonstration-conditioned outputs, improving VQA performance while lowering inference overhead, but it still fails to precisely recover the true, query-adaptive demonstration shifts. Motivated by these limitations, we propose Hyper-ICL to faithfully decompose the effects of ICDs for effective multimodal ICL. 

### **2.3. Learning in Hyperbolic Space** 

Hyperbolic geometry has emerged as an effective alternative to Euclidean representation learning when data exhibit latent hierarchies, power-law degree distributions, or tree-like structures. Early work such as Poincare embeddings showed´ that spaces with constant negative curvature can represent hierarchical relations with lower distortion and higher capacity than Euclidean spaces of the same dimensionality (Nickel & Kiela, 2017). Follow-up studies demonstrated that the Lorentz model improves numerical stability and optimization efficiency for large-scale taxonomy learning while preserving the same geometric advantages (Nickel & Kiela, 2018). Building on this, hyperbolic deep learning extends neural operations to Riemannian manifolds via exponential and logarithmic maps, enabling hyperbolic analogues of linear layers, recurrent units, and classification heads (Ganea et al., 2018). Recently, hyperbolic objectives have been explored in vision and multimodal learning to encode compositional scene-object hierarchies and structured vision-language alignment (Ge et al., 2023). In this work, we adopt the Lorentz model (Law et al., 2019) as a geometry-aware alignment space, using a layer-wise hyperbolic anchor distillation loss to match intermediate student representations to teacher hidden states via geodesic distance, yielding a structure-preserving regularizer that maintains latent demonstration–query relationships. 

## **3. Proposed Method** 

### **3.1. Motivation and Problem Setup** 

ICL enables LLMs and MLLMs to generalize to new tasks by conditioning on a small set of ICDs provided directly in the input. We define the prompt context as _C_ = _{XD, X}_ , where _XD_ = _{X_ 1 _, X_ 2 _, . . . , Xm} ∈_ R<sup>_TD×d_</sup> denotes the concatenation of _m_ ICDs, and _X ∈_ R<sup>_T ×d_</sup> is the query input. Here, _TD_ and _T_ represent the number of tokens in _XD_ and _X_ , respectively, and _d_ is the embedding dimension. Multi-head self-attention applies the self-attention (SA) mechanism across _Nh_ heads. Each head is parameterized by projection matrices _Wk, Wq, Wv ∈_ R<sup>_d×dh_</sup> , which 

map _C_ to keys _KC_ , queries _QC_ , and values _VC_ . Typically, _dh_ is set to _d/Nh_ so that each head operates in a lowerdimensional subspace, reducing parameter usage while preserving expressivity. For a given head, the key mapping is defined as follows: 



Similarly, we compute the corresponding _QD, Q_ , and _VD, V_ using _Wq_ and _Wv_ , respectively. For a query vector _q ∈ Q_ , the single-head self-attention computation is given by (for clarity, we present the formulation for a single head): 



where _µ_ ( _q, KD, K_ ) = _Z_ 1 _/_ ( _Z_ 1 + _Z_ 2), with _Z_ 1( _q, KD_ ) = � _Ti_ =1 _D_<sup>exp(</sup><sup>_qK_</sup> _D_<sup>_⊤_)</sup><sup>_i_and</sup><sup>_Z_2(</sup><sup>_q, K_)=�</sup><sup>_T_</sup> _j_ =1<sup>exp(</sup><sup>_qK⊤_)</sup><sup>_j_.</sup> Equation 2 shows that the self-attention over the prompt context _C_ can be decomposed into two terms. For the former “standard attention”, it is the self-attention over the query tokens, which is independent of the ICDs. While for the latter “shift vector”, it is the shift effects caused by the ICDs to shift the query space into the answer space, and such effects are calculated as the attention between the ICDs and the query _q_ . In in-context settings, demonstrations primarily affect where the model attends by injecting additional key-value evidence. Approximations that add a vector to the attention output behave like value-stream shifts (Peng et al., 2024; Jiang et al., 2025); however, they do not directly calibrate the attention distribution itself. To address this, we propose an intervention that directly calibrates the attention distribution via a learnable logit-level bias. 

### **3.2. In-Context Attention Calibration** 

**Low-rank Logit-level Adapter.** At layer _l_ , and attention head _h_ , the standard attention logits are computed using the scaled dot product between queries _Q_<sup>(</sup><sup>_l,h_)</sup> and keys _K_<sup>(</sup><sup>_l,h_)</sup> : 



where _Q_<sup>(</sup><sup>_l,h_)</sup> , _K_<sup>(</sup><sup>_l,h_)</sup> _∈_ R<sup>_T ×dh_</sup> and _S_<sup>(</sup><sup>_l,h_)</sup> _∈_ R<sup>_T ×T_</sup> . Therefore, the head output is computed by: 



Unlike prior studies that approximate demonstration effects by adding a vector to the attention output, we intervene 

3 



<!-- Start of picture text -->
' os, 1<br>\ i Truth ~<br>1 1! ~ .<br>H Top-downView ‘- ; m = ~ 0 | va<br>\ fe :<br>3 s, UcvsameeRenchor | NC’: "t Linear<br>oSwk 4<br>3 ——<br>— (ame) | 7 a an Transformation<br>: gun CO<br>a aun<br>' : ohh — Abr<br>3 g a a, bin<br>KA AR A Vik hh Tg ake:<br>—X |\X; Xo --- Xm |x Qe’ (Lh) eRTxd,<br>Query Demonstrations Query<br><!-- End of picture text -->

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

### **3.3. Layer-wise Hyperbolic Anchor Distillation** 

Building on the token-wise modulated logit-bias attention in Equation 5, we introduce an intermediate alignment regularizer that matches the student’s internal representations to those of a frozen teacher model that observes full demonstrations. Concretely, the teacher processes the full prompt context _C_ = [ _XD_ ; _X_ ], while the student processes only the query tokens _X_ and relies on the learned logit intervention to recover the demonstration-conditioned behavior. This design provides a structured distillation signal beyond output-level matching, encouraging consistent layer-wise transformations. As the teacher encodes many demonstration-conditioned relations among query tokens, a Euclidean penalty can prioritize pointwise matching while distorting relative similarity ordering when many relations compete in a low-dimensional embedding. Negative curvature offers higher effective representational capacity and mitigates distortion in low-dimensional embeddings; thus, geodesic alignment can better preserve the multi-scale distances that reflect how demonstrations reshape intermediate features. This is especially relevant when the student must reconstruct those relations without seeing demonstrations. 

**Lorentzian Representation of Layer Features.** Let _h_<sup>(</sup> _i_<sup>_l_)</sup> _∈_ R<sup>_d_</sup> denote the student hidden state of token index _i_ at layer _l ∈{_ 1 _, . . . , L}_ , and let _h_<sup>_′_</sup> _i_<sup>(</sup><sup>_l_)</sup> _∈_ R<sup>_d_</sup> denote the corresponding teacher hidden state for the same query token, computed under the full context _C_ . To capture hierarchical structure across layers in a geometry-aware manner, we align student and teacher representations in a hyperbolic space, whose distances naturally preserve layered, tree-like relationships. Specifically, we adopt the Lorentz model of hyperbolic space with constant negative curvature magnitude _κ >_ 0. The Lorentzian inner product on R<sup>_d_+1</sup> is defined as: 



and the corresponding _d_ -dimensional hyperboloid manifold is expressed as: 



We use the canonical base point _o_ ≜ ��1 _/κ,_ 0 _, . . . ,_ 0� _⊤_ as the origin of the manifold. Next, we map Euclidean hidden states onto H<sup>_d_</sup> _κ_<sup>using the exponential map at</sup><sup>_o_.To</sup> ensure scale stability and consistency with the normalization used in Equation 9, we first apply layer normalization to each hidden state as follows: 



Since the tangent space at _o_ can be identified with vectors whose first coordinate is zero, the padded vector _u_ ¯<sup>(</sup> _i_<sup>_l_)</sup> is a valid tangent direction at _o_ . We then apply the Lorentz exponential map as: 





where _u_<sup>_′_</sup> _i_<sup>(</sup><sup>_l_)</sup> ≜ LN( _h_<sup>_′_</sup> _i_<sup>(</sup><sup>_l_)</sup> ) _∈_ R<sup>_d_</sup> and _u_ ¯<sup>_′_</sup> _i_<sup>(</sup><sup>_l_)</sup> ≜ (0 _, u_<sup>_′_</sup> _i_<sup>(</sup><sup>_l_)</sup> )<sup>_⊤_</sup> _∈_ R<sup>_d_+1</sup> . **Hyperbolic-Based Regularization.** Given the hyperbolic embeddings _Pi_<sup>(</sup><sup>_l_)</sup> and _Pi_<sup>_′_(</sup><sup>_l_)</sup> , we measure alignment using the Lorentz geodesic distance on H<sup>_d_</sup> _κ_<sup>:</sup> 



where the argument _−κ ⟨P_<sup>_′_</sup> _, P ⟩L_ is guaranteed to be greater than or equal to 1 for valid points on the hyperboloid, making the distance well-defined. We then define a layer-wise hyperbolic anchor distillation loss that aligns student and teacher representations across all transformer layers: 



This loss provides explicit intermediate supervision, encouraging the student to reproduce the teacher’s demonstrationconditioned layer-wise transformations, rather than only matching final outputs. Intuitively, since the teacher observes _XD_ while the student does not, minimizing Equation 16 forces the student to recover teacher-like representations using only the learned attention-logit interventions. 

In addition, we employ a standard cross-entropy loss function, _L_ sup, as the main supervised loss objective to enhance student model’s predictions on downstream tasks. Therefore, the final training objective combines these losses as a weighted sum: 



where _λ ≥_ 0 is a hyperparameter that balances task supervision and intermediate alignment. This overall objective ensures that Hyper-ICL learns to apply efficient logit-level calibration while preserving layer-wise behaviors consistent with query-conditioned inference. 

## **4. Experiments** 

### **4.1. Implementation Details** 

We evaluate Hyper-ICL on two large-scale MLLMs, Idefics-9b (Laurenc¸on et al., 2023) and Idefics2-8B-base (Laurenc¸on et al., 2024). Idefics-9b adopts a cross-attention architecture, whereas Idefics2-8B-base follows a fully autoregressive design, representing two widely used architectural paradigms for vision-language modeling. Our experiments span a diverse set of benchmarks, including VQAv2 

5 

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

_Table 1._ Results of VQAv2, OK-VQA, and COCO on Idefics-9b and Idefics2-8B-base. **Bold numbers** represent the best results. In # Params (M), the first value denotes the number of trainable parameters (in millions), and the value in parentheses reports the relative factor compared to Hyper-ICL. 

|**Model **|**Type **|**Method**|**# Params (M) **|**VQAv2 **|**OK-VQA **|**COCO**|
|---|---|---|---|---|---|---|
||ct<br>s|Zero-shot|-|29.25|30.54|63.06|
||Dire<br>ICD|32-shot ICL|-|56.18|48.48|105.89|
|**b**||RICES|-|58.07|51.11|110.64|
|**s-9**|-<br>able|FV|-|30.21|31.02|74.01|
|**defc**|Non<br>earn|TV|-|43.68|32.68|84.72|
|**I**|L<br>le|LoRA|25.0 (_×_21.2)|55.60|47.06|97.75|
||nab|LIVE|0.13 (_×_0.11)|53.71|46.05|112.76|
||ear|MimIC|0.26 (_×_0.22)|59.64|52.05|114.89|
||L|Hyper-ICL|1.18 (_×_1)|**62.08**|**55.31**|**117.44**|
||ct<br>s|Zero-shot|-|55.39|43.08|40.00|
||ire<br>CD|8-shot ICL|-|66.20|57.68|122.51|
|**b**|D<br>I|RICES|-|66.44|55.73|111.44|
|**s2-8**|-<br>able|FV|-|36.47|34.58|75.24|
|**efc**|Non<br>earn|TV|-|47.12|38.27|87.61|
|**Id**|L<br>le|LoRA|17.6 (_×_14.9)|66.54|55.05|116.69|
||nab|LIVE|0.13 (_×_0.11)|67.60|54.86|126.04|
||ear|MimIC|0.26 (_×_0.22)|69.29|58.74|132.87|
||L|Hyper-ICL|1.18 (_×_1)|**71.17**|**62.24**|**135.66**|



(Goyal et al., 2017), OK-VQA (Marino et al., 2019), COCO Caption (Chen et al., 2015), Flickr30k (Young et al., 2014), MME (Fu et al., 2025), and SEED-Bench (Li et al., 2024a). To compare Hyper-ICL against prior SOTA methods, we focus on three widely used datasets: VQAv2, OK-VQA, and COCO Caption. VQAv2 targets open-ended visual question answering and contains 4,437,570 question-answer pairs in the training split, along with 2,143,540 pairs in the validation split. OK-VQA is designed to evaluate models that require external knowledge, comprising 14,055 questionanswer pairs, with 9,009 for training and 5,046 for validation. COCO Caption is a standard benchmark for image captioning built on the MS COCO image collection, providing multiple human-annotated descriptions per image to capture diverse yet semantically consistent views of the visual content. We further extend our evaluation to Flickr30k (Young et al., 2014), which includes 31,000 images each paired with five captions, MME (Fu et al., 2025), a comprehensive benchmark with 14 subtasks assessing perceptual and cognitive capabilities, and SEED-Bench (Li et al., 2024a), a multi-modal benchmark comprising 24,000 multiple-choice questions. 

For the training stage, we randomly sample 1,000 instances from each dataset to form the training set. In addition, we randomly select 32 samples as ICDs for Idefics-9b and 8 samples for Idefics2-8B-base, along with one separate sample used as the query input. Following prior evaluation protocols (Li et al., 2024b; Liu et al., 2023; Peng et al., 

_Table 2._ Results evaluated on more challenging tasks. 

|**Model**|**Method**|**Flickr30k**|**MME**|**SEED**|
|---|---|---|---|---|
||Zero-shot|49.17|55.36|27.56|
|Idefcs-9b|ICL|63.41|52.11|28.30|
||Hyper-ICL|**75.96**|**65.46**|**31.87**|
||Zero-shot|53.04|74.80|12.91|
|Idefcs2-8B-base|ICL|84.57|71.10|47.90|
||Hyper-ICL|**93.79**|**82.13**|**48.31**|



2024), we use 10,000 validation examples from VQAv2 and the full validation splits for OK-VQA and COCO. We employ the AdamW optimizer with a learning rate of 5 _×_ 10<sup>_−_3</sup> , coupled with a cosine annealing scheduler with warmup, allocating 10% of the total steps for warmup. All results are reported from the best-performing epoch. 

### **4.2. Comparison with Existing Methods** 

We compare Hyper-ICL with the following methods: 

**Direct use of ICDs.** It is evaluated under three settings: zero-shot, few-shot, and Retrieval-based In-Context Examples Selection (RICES) (Yang et al., 2022). For few-shot ICL, we use 32/8-shot for Idefics-9B and Idefics2-8B-base, respectively. We also compare Hyper-ICL with RICES, which retrieves visually similar support images for each query by matching features extracted from a frozen pretrained encoder. 

**Non-learnable vector-based ICDs.** Task Vector (TV) (Brown et al., 2020) and Function Vector (FV) (Todd et al., 2023) that extract an in-context vector from examples and inject it into the model’s hidden states during inference. 

**Learnable use of ICDs.** We compare Hyper-ICL with LoRA (Hu et al., 2022), LIVE (Peng et al., 2024), and MimIC (Jiang et al., 2025), where LIVE and MimIC are trainable ICV methods that distill few-shot ICD effects into learnable shift vectors under the same few-shot setting. For LoRA, we follow the standard setup and apply it to all attention layers in both the vision and language encoders. 

Table 1 compares Hyper-ICL against a broad set of baselines on two MLLMs and three datasets. Overall, prior nontrainable ICD selection methods remain consistently below the standard 32-shot ICL setting. For instance, while RICES improves over random selection on all datasets for Idefics9B, it becomes less reliable on Idefics2-8B-base, where its performance on OK-VQA and COCO falls below random selection. This behavior stems from architectural differences: nearest-neighbor retrieval aligns well with Idefics9B’s cross-attention fusion, whereas Idefics2-8B-base’s fully autoregressive decoding benefits more from context diversity than visual similarity. In contrast to non-trainable approaches, trainable methods consistently improve performance across both backbones, often approaching the effectiveness of few-shot ICL. Notably, Hyper-ICL achieves the best results among all learnable baselines on both MLLMs, 

6 



<!-- Start of picture text -->
VQAv?2 (Idefics-9b) OK-VQA (Idefics-9b) VQAv?2 (Idefics2-8b-base) OK-VQA (Idefics2-8b-base)<br>62 56 72 62<br>> 52 684 8-shot ICL —* 58} ----------8-shot ICL 5B<br>Q53 58 ge eeeree32-shot CL 48 bs<7 Hot ICL SS] a = °4 —-<br>< 54 44 60 50 “C4<br>50 1. $——— 40 56 46<br>100 200 300 500 1000 100 200 300 500 1000 100 200 300 500 1000 100 200 300 500 1000<br>Samples Samples Samples Samples<br>—® LoRA —@® LIVE —® MimiC —@® Hyper-ICL<br><!-- End of picture text -->





<!-- Start of picture text -->
65 609. 60.5}62.1 60.5}62.2 | ©°<br>60 59.4 61.5<br>z© 554 53.5] 54.9) 55.1 61.0<br>3 60.5<br>=<br>59 Gl Layer-wise (Static) 60.0 —®—Hyper-ICL, A=0.1<br>45) GG Layer & Head-wise (Static) 50,5 —@®Hyper-ICL, A=0.5<br>gm Hyper-ICL (Dynamic) " —@-Hyper-ICL, A=1<br>2 4 8 0.01 0.1 0.5 1.0<br>(a) Rank (r) (b) Curvature (x)<br><!-- End of picture text -->



**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 



<!-- Start of picture text -->
Question:<br>What does the sign say?<br>LoRA: LIVE:<br><EOS> Bus<br>ICL: Hyper-ICL:<br>Bus Dark Skies<br><!-- End of picture text -->



<!-- Start of picture text -->
Question: Question:<br>What does the sign say? What color is the purse?<br>LoRA: LIVE: LoRA: LIVE:<br><EOS> Bus <EOS> Red<br>ICL: Hyper-ICL: ICL: Hyper-ICL:<br>Bus Dark Skies Red Blue<br><!-- End of picture text -->



<!-- Start of picture text -->
Hyper-ICL:<br>Dark Skies<br><!-- End of picture text -->



<!-- Start of picture text -->
Hyper-ICL:<br>Blue<br><!-- End of picture text -->



<!-- Start of picture text -->
Red<br><!-- End of picture text -->



<!-- Start of picture text -->
Bus<br><!-- End of picture text -->



<!-- Start of picture text -->
Question: Question:<br>Which street is straight ahead? What does the sign say?<br>LoRA: LIVE: LoRA: LIVE:<br><EOS> Hoffman <EOS> No Parking<br>ICL: Hyper-ICL: ICL: Hyper-ICL:<br>Hoffman S.8th ST No Bikes Stop<br><!-- End of picture text -->



<!-- Start of picture text -->
Question:<br>Which street is straight ahead?<br>LoRA: LIVE:<br><EOS> Hoffman<br>ICL: Hyper-ICL:<br>Hoffman S.8th ST<br><!-- End of picture text -->



<!-- Start of picture text -->
Hyper-ICL:<br>Stop<br><!-- End of picture text -->



<!-- Start of picture text -->
Hoffman<br><!-- End of picture text -->



_Figure 4._ Qualitative comparison of hallucination behavior across methods on VQA. Hyper-ICL stays visually grounded on representative queries (e.g., sign text, purse color, and road direction), while standard ICL, LoRA, and LIVE produce ungrounded content. 

_Table 6._ Analysis of inference FLOPs and runtime. 

|**Metric**|**Hyper-ICL **|**0-Shot**|**8-shot**|**16-shot **|**32-shot**|
|---|---|---|---|---|---|
|FLOPs (T)|0.955|0.935|6.375|12.341|23.364|
|Runtime(ms)|59.32|56.69|158.21|266.13|468.55|



benefits while enabling demonstration-free inference with near zero-shot efficiency. 

## **5. Conclusion** 

In this work, we present Hyper-ICL, a training-based framework for demonstration-free multimodal ICL that reconstructs ICD effects inside self-attention. Hyper-ICL introduces a parameter-efficient low-rank logit-level adapter that injects head-wise biases into attention logits, together with a query-adaptive token-wise modulation mechanism to enable query-adaptive calibration across layers and heads. To transfer demonstration-conditioned representations, we propose a layer-wise hyperbolic anchor distillation loss, aligning student representations to a frozen demonstration-conditioned teacher via geodesic distance. By removing ICDs at inference, it shortens context length and reduces latency while retaining the benefits of few-shot prompting. Experiments on Idefics-9B and Idefics2-8B over VQAv2, OK-VQA, and COCO Caption show consistent gains and improved stability over direct ICD prompting, vector-based ICV baselines, and trainable SOTA methods. 

## **Acknowledgment** 

This material is based upon work supported by the National Science Foundation under Grant Numbers CNS-2232048, and CNS-2204445. 

## **Impact Statement** 

This work aims to improve the efficiency and practicality of multimodal ICL by eliminating the need for demonstration tokens at inference time. By reconstructing demonstration effects through a lightweight adapter, Hyper-ICL achieves near zero-shot inference cost while retaining the benefits of few-shot prompting, substantially reducing latency and computational overhead. This efficiency can support greener AI deployment by lowering inference-time energy consumption, which is especially important for large-scale visionlanguage systems. The low-latency profile of Hyper-ICL may also broaden the use of multimodal models in realtime applications such as assistive technologies, medical decision support, and interactive vision-language systems, where long demonstration-based prompts are often impractical. 

A limitation of the proposed framework is that the adapter is learned for a given task or domain setting. Therefore, switching to substantially different tasks, domains, or model backbones may require retraining or re-calibrating the adapter. As with other multimodal learning systems, deployment in high-stakes settings should include careful validation, robustness testing, and human oversight. 

## **References** 

- Alipour Talemi, N., Kashiani, H., Nowdeh, H. R., and Afghah, F. DiSa: Directional saliency-aware prompt learning for generalizable vision-language models. In _Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2_ , pp. 37–46, 2025. 

- Baldassini, F. B., Shukor, M., Cord, M., Soulier, L., and Piwowarski, B. What makes multimodal in-context learning work? In _Proceedings of the IEEE/CVF Conference_ 

9 

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

_on Computer Vision and Pattern Recognition_ , pp. 1539– 1550, 2024. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33: 1877–1901, 2020. 

- Chen, X., Fang, H., Lin, T.-Y., Vedantam, R., Gupta, S., Dollar, P., and Zitnick, C. L.´ Microsoft COCO captions: Data collection and evaluation server. _arXiv preprint arXiv:1504.00325_ , 2015. 

- Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 24185–24198, 2024. 

- Dong, Q., Li, L., Dai, D., Zheng, C., Ma, J., Li, R., Xia, H., Xu, J., Wu, Z., Chang, B., et al. A survey on incontext learning. In _Proceedings of the 2024 conference on empirical methods in natural language processing_ , pp. 1107–1128, 2024. 

- Doveh, S., Perek, S., Mirza, M. J., Lin, W., Alfassy, A., Arbelle, A., Ullman, S., and Karlinsky, L. Towards multimodal in-context learning for vision and language models. In _European Conference on Computer Vision Workshops_ , pp. 250–267. Springer, 2024. 

- Fish, E. and Bowden, R. Geo-Sign: Hyperbolic contrastive regularisation for geometrically aware sign language translation. _Advances in Neural Information Processing Systems_ , 38:99293–99330, 2025. 

- Fournier, H., Ismail, A., and Vigneron, A. Computing the Gromov hyperbolicity of a discrete metric space. _Information Processing Letters_ , 115(6-8):576–579, 2015. 

- Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. MME: A comprehensive evaluation benchmark for multimodal large language models. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2025. 

- Ganea, O., Becigneul, G., and Hofmann, T.´ Hyperbolic neural networks. _Advances in neural information processing systems_ , 31, 2018. 

- Ge, S., Mishra, S., Kornblith, S., Li, C.-L., and Jacobs, D. Hyperbolic contrastive learning for visual representations beyond objects. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 6840–6849, 2023. 

- Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 6904–6913, 2017. 

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. LoRA: Low-rank adaptation of large language models. _ICLR_ , 1(2):3, 2022. 

- Ibrahimi, S., Atigh, M. G., Van Noord, N., Mettes, P., and Worring, M. Intriguing properties of hyperbolic embeddings in vision-language models. _Transactions on Machine Learning Research_ , 2024. 

- Jiang, Y., Fu, J., Hao, C., Hu, X., Peng, Y., Geng, X., and Yang, X. Mimic in-context learning for multimodal tasks. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pp. 29825–29835, 2025. 

- Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., and Lempitsky, V. Hyperbolic image embeddings. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 6418–6428, 2020. 

- Laurenc¸on, H., Saulnier, L., Tronchon, L., Bekman, S., Singh, A., Lozhkov, A., Wang, T., Karamcheti, S., Rush, A., Kiela, D., et al. OBELICS: An open web-scale filtered dataset of interleaved image-text documents. _Advances in Neural Information Processing Systems_ , 36:71683– 71702, 2023. 

- Laurenc¸on, H., Tronchon, L., Cord, M., and Sanh, V. What matters when building vision-language models? _Advances in Neural Information Processing Systems_ , 37: 87874–87907, 2024. 

- Law, M., Liao, R., Snell, J., and Zemel, R. Lorentzian distance learning for hyperbolic representations. In _International Conference on Machine Learning_ , pp. 3672–3681. PMLR, 2019. 

- Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., and Shan, Y. SEED-Bench: Benchmarking multimodal large language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 13299–13308, 2024a. 

- Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., and Li, C. LLaVAonevision: Easy visual task transfer. _Transactions on Machine Learning Research_ , 2025a. ISSN 2835-8856. 

- Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., and Li, C. LLaVA-NeXT-Interleave: Tackling multiimage, video, and 3d in large multimodal models. In _International Conference on Learning Representations_ , volume 2025, pp. 81182–81199, 2025b. 

10 

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

- Li, J., Li, D., Savarese, S., and Hoi, S. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pp. 19730–19742. PMLR, 2023. 

- Li, L., Peng, J., Chen, H., Gao, C., and Yang, X. How to configure good in-context sequence for visual question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 26710– 26720, 2024b. 

- Li, X., Liu, Y., Kang, X., Luo, Z., Lou, F., Wu, X., and Xiong, Z. HIFICL: High-fidelity in-context learning for multimodal tasks. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , 2026. 

- Li, Y., Cao, Y., He, H., Cheng, Q., Fu, X., Xiao, X., Wang, T., and Tang, R. M²IV: Towards efficient and fine-grained multimodal in-context learning via representation engineering. In _Second Conference on Language Modeling_ , 2025c. 

- Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. _Advances in neural information processing systems_ , 36:34892–34916, 2023. 

- Liu, J., Shen, D., Zhang, Y., Dolan, W. B., Carin, L., and Chen, W. What makes good in-context examples for GPT3? In _Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd workshop on knowledge extraction and integration for deep learning architectures_ , pp. 100–114, 2022. 

- Ma, Z., Zhang, S., Wei, L., and Tian, Q. Efficient multimodal long context learning for training-free adaptation. _arXiv preprint arXiv:2505.19812_ , 2025. 

- Marino, K., Rastegari, M., Farhadi, A., and Mottaghi, R. OK-VQA: A visual question answering benchmark requiring external knowledge. In _Proceedings of the IEEE/cvf conference on computer vision and pattern recognition_ , pp. 3195–3204, 2019. 

- Nickel, M. and Kiela, D. Poincare embeddings for learning´ hierarchical representations. _Advances in neural information processing systems_ , 30, 2017. 

- Nickel, M. and Kiela, D. Learning continuous hierarchies in the lorentz model of hyperbolic geometry. In _International conference on machine learning_ , pp. 3779–3788. PMLR, 2018. 

- Peng, Y., Hu, X., Peng, J., Geng, X., Yang, X., et al. LIVE: Learnable in-context vector for visual question answering. _Advances in Neural Information Processing Systems_ , 37: 9773–9800, 2024. 

- Poppi, T., Kasarla, T., Mettes, P., Baraldi, L., and Cucchiara, R. Hyperbolic safety-aware vision-language models. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pp. 4222–4232, 2025. 

- Qin, L., Chen, Q., Fei, H., Chen, Z., Li, M., and Che, W. What factors affect multi-modal in-context learning? an in-depth exploration. _Advances in Neural Information Processing Systems_ , 37:123207–123236, 2024. 

- Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., and Saenko, K. Object hallucination in image captioning. _arXiv preprint arXiv:1809.02156_ , 2018. 

- Shao, Z., Yu, Z., Wang, M., and Yu, J. Prompting large language models with answer heuristics for knowledgebased visual question answering. In _Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition_ , pp. 14974–14983, 2023. 

- Shukor, M., Rame, A., Dancette, C., and CORD, M. Beyond task performance: evaluating and reducing the flaws of large multimodal models with in-context-learning. In _International Conference on Learning Representations_ , volume 2024, pp. 21756–21786, 2024. 

- Sun, Y., Chen, Q., Wang, J., Wang, J., and Li, Z. Exploring effective factors for improving visual in-context learning. _IEEE Transactions on Image Processing_ , 2025. 

- Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., and Bau, D. Function vectors in large language models. _arXiv preprint arXiv:2310.15213_ , 2023. 

- Xu, W., Wang, J., Wang, W., Chen, Z., Zhou, W., Yang, A., Lu, L., Li, H., Wang, X., Zhu, X., et al. VISULOGIC: A benchmark for evaluating visual reasoning in multi-modal large language models. In _International Conference on Learning Representations_ , 2026. 

- Yang, Z., Gan, Z., Wang, J., Hu, X., Lu, Y., Liu, Z., and Wang, L. An empirical study of GPT-3 for few-shot knowledge-based VQA. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 36, pp. 3081– 3089, 2022. 

- Young, P., Lai, A., Hodosh, M., and Hockenmaier, J. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _Transactions of the association for computational linguistics_ , 2:67–78, 2014. 

- Zhang, Y., Zhou, K., and Liu, Z. What makes good examples for visual in-context learning? _Advances in Neural Information Processing Systems_ , 36:17773–17794, 2023. 

- Zhou, K., Yang, J., Loy, C. C., and Liu, Z. Learning to prompt for vision-language models. _International Journal of Computer Vision_ , 130(9):2337–2348, 2022. 

11 

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

## **A. Generalize to Additional MLLMs** 

To further evaluate the robustness and scalability of Hyper-ICL across different multimodal architectures, we additionally conduct experiments on two recent MLLMs: LLaVA-Interleave-7B (Li et al., 2025b) and LLaVA-OneVision (Qwen2-7B) (Li et al., 2025a). Table 7 shows that Hyper-ICL consistently outperforms zero-shot, standard few-shot ICL, and the recent MimIC method (Jiang et al., 2025) on both VQAv2 (Goyal et al., 2017) and OK-VQA (Marino et al., 2019). These results further demonstrate that the proposed logit-level attention calibration and hyperbolic distillation generalize effectively across diverse MLLM architectures. 

_Table 7._ Results evaluated on additional MLLMs. 

|**Backbone**|**Method**|**VQAv2**|**OK-VQA**|
|---|---|---|---|
||Zero-shot|13.02|5.10|
|LLaVA-Interleave-7B|8-shot ICL|68.19|43.84|
||MimIC|74.40|52.29|
||Hyper-ICL|**76.51**|**56.77**|
||Zero-shot|71.75|48.19|
|LLaVA-OneVision|8-shot ICL|78.70|66.59|
|(Qwen2-7B)|MimIC|81.22|69.43|
||Hyper-ICL|**82.24**|**75.32**|



## **B. Held-out Benchmark Transfer** 

We further evaluate whether Hyper-ICL can transfer to unseen benchmarks within the same task family. We conduct two held-out transfer experiments on Idefics2-8B-base (Laurenc¸on et al., 2024). In the first setting, the model is trained on VQAv2 and evaluated on the unseen OK-VQA benchmark. In the second setting, the model is trained on COCO Caption (Chen et al., 2015) and evaluated on Flickr30k (Young et al., 2014). For a fair comparison, the 8-shot ICL baseline also uses demonstrations drawn from the source dataset rather than the target dataset. As shown in Table 8, Hyper-ICL substantially improves over both zero-shot inference and standard 8-shot ICL on unseen benchmarks within the same task family. These results indicate that the learned attention calibration is not purely dataset-specific and can capture transferable demonstration-conditioned behavior across related benchmarks. 

_Table 8._ Held-out benchmark transfer results on Idefics2-8B-base. 

|**Train**|**Test**|**Zero-shot**|**8-shot ICL**|**Hyper-ICL**|
|---|---|---|---|---|
|COCO|Flickr30k|53.04|73.26|**87.42**|
|VQAv2|OK-VQA|43.08|51.43|**59.11**|



## **C. Additional Analysis of Hyperbolic Structure** 

In this section, we provide a direct empirical analysis of the hyperbolic anchor loss to support our assumption that multimodal ICL induces dense demonstration-conditioned relations among query tokens. Specifically, we examine this geometric assumption by measuring whether the intermediate hidden-state representations exhibit a non-uniform, tree-like relational structure. 

We quantify this structure using relative _δ_ -hyperbolicity, which measures how closely a metric space satisfies the Gromov four-point condition. Smaller values indicate that pairwise metric relations are closer to those of a tree-like geometry, while larger values indicate a flatter structure. Following (Khrulkov et al., 2020), we use the scale-invariant score as follows: 



where _δ_ ( _X_ ) is the estimated Gromov hyperbolicity of the metric space _X_ . 

We conduct this analysis on Idefics2-8B-base using 1,000 VQAv2 samples. For each sample, we construct the metric space from query-token hidden states in the four middle transformer layers. We compare three settings: the demonstrationconditioned teacher hidden states, the student hidden states trained with the Euclidean anchor loss, and the student hidden states trained with the proposed hyperbolic anchor loss. For a fair comparison, we use the same sampled four-point tuples across all settings when estimating _δ_ ( _X_ ). Specifically, we estimate _δ_ ( _X_ ) using the efficient four-point procedure adopted 

12 

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

in (Khrulkov et al., 2020; Fournier et al., 2015): we sample 1,000 four-tuples, compute the four-point _δ_ value for each, take the maximum sampled value, normalize by the diameter, and report mean _±_ std. 

_Table 9._ Relative _δ_ -hyperbolicity of intermediate hidden-state representations on Idefics2-8B-base. Lower values indicate a metric geometry closer to a tree-like structure. 

|**Setting**|**_δ_rel** **_↓_**|
|---|---|
|Teacher hidden states|0_._228_±_0_._014|
|Student hidden states + Euclidean anchor|0_._314_±_0_._023|
|Student hidden states + Hyperbolic anchor|0_._247_±_0_._016|



As shown in Table 9, the demonstration-conditioned teacher hidden states exhibit a measurable degree of hyperbolic structure, suggesting that the teacher representations are not merely an unstructured set of hidden vectors. Instead, they encode a non-uniform relational geometry induced by multimodal demonstrations. The student trained with the proposed hyperbolic anchor loss achieves a substantially lower _δ_ rel than the Euclidean-anchor student and more closely matches the teacher geometry. These results provide direct evidence that the gain from the hyperbolic anchor is not only due to adding an intermediate regularizer, but is also consistent with better preserving the demonstration-conditioned relational structure that Hyper-ICL is designed to reconstruct. 

## **D. Interpretability Analysis of Query-Adaptive Modulation** 

In Section 3.2, Hyper-ICL introduces a query-adaptive token-wise modulation vector _gl,h ∈_ (0 _,_ 1)<sup>_T_</sup> to control the strength of the logit-level intervention for each query token across layers and heads. In this section, we provide an additional analysis of the learned modulation behavior to examine whether _gl,h_ provides meaningful, input-dependent control rather than collapsing to a nearly constant scalar. 

For layer _l_ , head _h_ , and query token _t_ , we denote the corresponding modulation value by _gl,h,t_ . Larger values indicate stronger activation of the learned logit-level bias for that token. We analyze the learned gates on Idefics-9B (Laurenc¸on et al., 2023) using subsets of VQAv2 and OK-VQA. First, we compute the mean gate value in early layers (0-7) and late layers (24-31), together with the average per-layer standard deviation over heads and query tokens. This evaluates whether the gate saturates near 0 or 1, or instead preserves meaningful variation across the model. 

Second, we examine whether the learned gate becomes larger when demonstrations induce stronger changes in attention. For each query, we run the model twice: once with the full demonstration-conditioned prompt and once with the query-only prompt. We define the demonstration-induced attention shift as: 



where _A_<sup>ICD</sup> _l,h,t_<sup>and</sup><sup>_A_noICD</sup> _l,h,t_ denote the attention distributions for token _t_ under the demonstration-conditioned and query-only settings, respectively. We then compute the Pearson correlation between _gl,h,t_ and ∆ _Al,h,t_ over all evaluated examples. A positive correlation indicates that the gate assigns stronger intervention weights to tokens, heads, and layers where demonstrations cause larger attention redistribution. 

_Table 10._ Quantitative analysis of query-adaptive token-wise modulation on Idefics-9B. Reported metrics include the mean gate value in early and late layers, the average per-layer standard deviation, and the Pearson correlation between the learned gate _g_ and the demonstration-induced attention shift ∆ _A_ . 

|**Dataset**|**Early-layer mean** _g_|**Late-layer mean** _g_|**Avg. per-layer std** _g_|**Corr(****_g,_ ∆****_A_)**|
|---|---|---|---|---|
|VQAv2|0.38|0.46|0.17|0.41|
|OK-VQA|0.43|0.62|0.23|0.56|



Table 10 supports the intended behavior of the modulation mechanism in three ways. First, the mean gate values do not collapse toward 0 or 1, and the non-trivial per-layer standard deviations show that the gate preserves meaningful variation rather than acting as a nearly constant scalar. Second, the gate values increase in later layers, which is consistent with the design motivation that deeper layers contribute more strongly to compositional reasoning and answer generation. This pattern is more pronounced on OK-VQA, where the late-layer mean gate reaches 0 _._ 62, reflecting the stronger reasoning and knowledge demands of this benchmark. Third, the positive correlations between _g_ and ∆ _A_ on both datasets provide direct evidence that the gate is activated more strongly when demonstrations induce larger attention redistribution. 

13 

**Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning** 

## **E. Output-level Offset vs. Logit-level Attention Calibration** 

We further analyze whether the benefit of Hyper-ICL can be recovered by applying a learnable correction after attention, rather than intervening on the attention logits. Although modifying the attention logits ultimately changes the attention output, the two operations are not equivalent. An output-level offset directly perturbs the resulting vector representation after the attention-weighted summation, whereas our logit-level intervention changes the attention distribution before the value aggregation. Therefore, Hyper-ICL explicitly controls which tokens are attended to and how strongly they compete under the softmax normalization. 

This distinction is important in multimodal ICL, where demonstrations mainly affect the query by redistributing attention over relevant visual and textual tokens. Existing output-level methods, such as LIVE (Peng et al., 2024), inject learned layer-wise shift vectors into hidden activations, but they do not directly calibrate the attention probabilities. As shown in Table 1, Hyper-ICL consistently outperforms LIVE across VQAv2, OK-VQA, and COCO, indicating that output-level shifts are less effective at reconstructing demonstration-induced behavior. 

To further analyze the effect of intervention placement, we additionally implement a variant that uses the same low-rank, query-adaptive module as Hyper-ICL but applies it to the post-attention output _O_ rather than to the pre-softmax logits _S_ . As shown in Table 11, the attention-logit intervention used by Hyper-ICL leads to stronger results than applying the same module at the output level. This confirms that the improvement does not simply come from adding a learnable correction, but from applying the correction at the attention-logit level, where the model can directly reshape token competition and attention redistribution. 

_Table 11._ Comparison between applying the same low-rank, query-adaptive module to the post-attention output and applying it at the attention-logit level in Hyper-ICL. 

|**Method**|**VQAv2**|**OK-VQA**|
|---|---|---|
|Intervention on_O_|58.42|50.18|
|Hyper-ICL|**62.08**|**55.31**|



14 

