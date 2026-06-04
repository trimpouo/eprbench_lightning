# EPRBench: A High-Quality Benchmark Dataset for Event Stream Based Visual Place Recognition

Xiao Wang, Member, IEEE, Xingxing Xiong, Jinfeng Gao, Xufeng Lou, Bo Jiang, Sibao Chen, Yaowei Wang, Member, IEEE, Yonghong Tian, Fellow, IEEE 

Abstract—Event stream-based Visual Place Recognition (VPR) is an emerging research direction that offers a compelling solution to the instability of conventional visible-light cameras under challenging conditions such as low illumination, overexposure, and high-speed motion. Recognizing the current scarcity of dedicated datasets in this domain, we introduce EPRBench, a highquality benchmark specifically designed for event stream–based VPR. EPRBench comprises 10K event sequences and 65K event frames, collected using both handheld and vehicle-mounted setups to comprehensively capture real-world challenges across diverse viewpoints, weather conditions, and lighting scenarios. To support semantic-aware and language-integrated VPR research, we provide LLM-generated scene descriptions, subsequently refined through human annotation, establishing a solid foundation for integrating LLMs into event-based perception pipelines. To facilitate systematic evaluation, we implement and benchmark 15 state-of-the-art VPR algorithms on EPRBench, offering a strong baseline for future algorithmic comparisons. Furthermore, we propose a novel multi-modal fusion paradigm for VPR: leveraging LLMs to generate textual scene descriptions from raw event streams, which then guide spatially attentive token selection, cross-modal feature fusion, and multi-scale representation learning. This framework not only achieves highly accurate place recognition but also produces interpretable reasoning processes alongside its predictions, significantly enhancing model transparency and explainability. The dataset and source code will be released on https://github.com/Event-AHU/Neuromorphic_ReID 

Index Terms—Visual Place Recognition; Event Stream; Benchmark Dataset; Chain-Of-Thought Reasoning; Large Language Model 

# I. INTRODUCTION

V ISUAL Place Recognition (VPR) [1], [2], [3] is a com- puter vision task that aims to determine whether an agent is at a previously visited location solely based on current visual input, fundamentally framed as a large-scale image retrieval problem. As a cornerstone for long-term, GPS-denied localization in mobile and autonomous systems, VPR finds broad applications in loop closure detection for robotic SLAM, high-precision localization in autonomous driving, UAV navigation, and augmented reality. However, real-world challenges, particularly “perceptual aliasing” caused by variations in illumination, season, weather, and dynamic occlusions, pose 

• Xiao Wang, Xingxing Xiong, Jinfeng Gao, Xufeng Lou, Bo Jiang, Sibao Chen are with School of Computer Science and Technology, Anhui University, Hefei, China. (email: {xiaowang, jiangbo, sbchen}@ahu.edu.cn, {e24201089, e24201028}@stu.ahu.edu.cn, gaoelieen750@gmail.com) 

• Yaowei Wang is with Harbin Institute of Technology, Shenzhen, China; Peng Cheng Laboratory, Shenzhen, China. (email: wangyw@pcl.ac.cn) 

• Yonghong Tian is with Peng Cheng Laboratory, Shenzhen, China; School of Computer Science, Peking University, China; Shenzhen Graduate School, Peking University, China. (email: yhtian@pku.edu.cn) 

• Corresponding author: Bo Jiang 

significant robustness hurdles. In addition, advancing VPR requires addressing critical research directions: improving generalization and interpretability in unseen environments (e.g., by integrating semantic priors from large language models and chain-of-thought reasoning), while simultaneously ensuring computational efficiency and scalability for large-scale, longterm deployment. 

Despite significant breakthroughs enabled by deep learning and large foundation models [4], overall performance remains constrained by the widespread reliance on RGB frame-based cameras. Specifically, RGB cameras are highly sensitive to lighting conditions and perform poorly in scenarios involving overexposure, low illumination, or those requiring high dynamic range (HDR) perception. Moreover, conventional RGB cameras are limited by their frame rate (typically around 30 FPS), making them prone to motion blur under highspeed movement. While using high frame rate cameras could mitigate this issue, it would impose significantly higher demands on data transmission bandwidth and computational processing challenges that hinder practical deployment in realworld applications. 

To address these issues, the recently developed event cameras are drawing more and more attention in computer vision [5], [6], such as object detection and tracking [7], human activity recognition [8], attribute recognition [9], person reidentification [10], and video captioning [11]. The widespread attention stems from the fact that this sensor is a novel bioinspired camera capable of asynchronously capturing changes in pixel brightness, thereby directly outputting information about actual motion in the scene, akin to optical flow derived straight from the sensor itself. Event cameras typically offer an exceptionally wide dynamic range of 120–140 dB, enabling superior perception in both high-exposure and lowlight conditions. Their spatial sparsity combined with temporal density allows them to effectively capture high-speed motion; for instance, a rapidly moving drone can acquire clearer, more detailed images of the ground below. Moreover, event cameras have demonstrated distinct advantages in terms of low power consumption and ultra-low latency. Some researchers also incorporate such sensors into the VPR community, e.g., Lee et al. [12] achieve robust VPR in the low-illumination conditions using the textural information from the event cameras. Tobias et al. [13] propose the first event-based dataset for largescale visual place recognition, termed Brisbane-Event-VPR. Building upon these valuable datasets and advances, Wang et al. [14] propose the Focus on Local (FOL) approach, which mines reliable discriminative local regions to improve 

Query : This image proides aclose-up perspective of a modern multi-story building,emphasizing its architectural details.The primary landmark is a prominent rectangular facadecharacterized by a repetitive vertical window grid pattern. 

Candidate 1: This image captures the same modern building but from a significantly farther distance and a wider angle.The ouerall rectangular oolume of thebuilding is now fully visible,，maintaining the samecharacteristic vertical window pattern across its facade. 

![](images/7978a9d252ee7582ec49e8330f4e79fdf3bc9ee63993e09b87a17ec0858c379a.jpg)



(a). An illustration of event stream based visual place recognition


![](images/ee993cd085245430fe8b57b4b93192d4660661193b5424cbec861d62c33d2413.jpg)



(b). Comparison between existing VPR (left) and our newly proposed one (right)



Fig. 1. (a). An illustration of event stream-based VPR; (b). Comparison between existing VPR frameworks and ours.


VPR retrieval and re-ranking performance. Ali-Bey et al. [15] propose Bag-of-Queries method, which learns a set of global learnable queries to probe input features via cross-attention for consistent information aggregation and achieve superior performance. 

The aforementioned works have clearly laid a solid foundation, both in terms of data and algorithms, for the task of eventbased Visual Place Recognition (VPR), and have demonstrated the research value of this direction. However, progress in this field remains constrained by several key challenges: 1) Limited Availability and Scale of Datasets: To date, only three event-based VPR datasets are publicly available in the academic community: Brisbane-Event-VPR [13] (introduced in 2020), NeuroGPR [16] (2023), and NYC-Event-VPR [17] (2024). As summarized in Table I, these datasets are notably limited in both scale and diversity, falling far short of the growing demand for large-scale, high-quality VPR benchmarks. 2) Insufficient Semantic Understanding: Current approaches typically focus on designing powerful neural networks to extract effective feature representations from input images or event streams. This paradigm, centered on representation learning, reflects the dominant methodology of the first half of the third wave of AI, yet it lacks deep exploitation of semantic cues essential for robust place recognition. 3) Poor Model Interpretability: Mainstream deep learning–based VPR models often rely on activation heatmaps (e.g., feature map visualizations) to highlight regions of interest, offering limited visual justification for their place matching decisions. However, such post-hoc visual explanations remain far from achieving truly intelligent, human-aligned interpretability. 

In this paper, we propose a new large-scale, high-definition (1280 × 720) event stream based VPR benchmark dataset, termed EPRBench. In more detail, our dataset contains 10K event sequences, which involve 1K urban scenes and 65K event frames. The data were primarily collected using two methods: handheld and vehicle-mounted setups, covering a variety of viewpoints, weather conditions, and lighting scenarios. To better support the interpretability of VPR algorithms in the era of large models, this paper also leverages large language models (i.e., DouBao [18]) to generate textual descriptions for the provided data and performs manual verification to ensure the accuracy of these semantic descriptions. A comparison between the proposed dataset and existing ones can be found in Table I. In addition, we also retrain and report 15 stateof-the-art VPR algorithms based on our benchmark dataset, including the CNN-based, Transformer-based, and re-ranking based VPR models. 

Based on the VPR benchmark dataset proposed in this paper, we also introduce a novel VPR paradigm that leverages large language models to extract semantic descriptions from visual inputs, thereby transforming the traditional purely vision-based perception task into a vision-language fusion problem, as shown in Fig. 1. Specifically, Fig. 1 (b) contrasts this with existing frameworks: while traditional methods rely solely on opaque visual embeddings, our approach integrates an LLM branch to generate explicit semantic descriptions for more robust matching. Specifically, we adopt DINO V2 and the CLIP model to extract the event and language features. We then compute the cross-modal correlations between semantic and visual tokens, use these correlations to weight the word tokens, and fuse them with the top-k selected visual tokens. Meanwhile, the unselected visual tokens are repurposed to reshape a global feature map representation. Then, multi-scale features are extracted via average pooling as the final semantic enhanced vision features. Note that we also introduce an LLM decoder to generate the chain-of-thought reasoning process to further improve the model interpretability. 

To sum up, we conclude the main contributions of this paper into the following three aspects: 

1) We propose a large-scale and new high-definition benchmark dataset for the event stream based visual place recognition, termed EPRBench. It contains 13,109 video samples and was collected under various weather and light conditions. 

2) We retrain and report the performance of existing 15 open-sourced VPR models based on our EPRBench, which builds a solid foundation for the research of event camerabased place recognition. 

3) We propose a simple yet effective baseline for eventbased visual place recognition (VPR) that leverages semantic cues derived from LLMs, termed SG-VPR. By grounding visual features in LLM-generated semantic descriptions, our approach effectively captures spatially meaningful semantics. This is further enhanced through multi-modal fusion and multi-scale feature integration. Moreover, we incorporate chain-of-thought reasoning to improve model interpretability further. 

The rest of this paper is organized as follows: In section II, we review the existing works related to this paper with a focus on visual place recognition and event-based vision. Then, we dive into the details of the EPRBench benchmark dataset in section III. We describe the baseline approach proposed in this paper in section IV and report the experimental results in section V. Finally, we conclude this paper and propose possible research directions in section VI. 

# II. RELATED WORK

# A. Visual Place Recognition

Visual Place Recognition (VPR) is a task that identifies previously visited locations solely based on visual input, aiming at achieving autonomous localization and relocalization through image information. Traditional VPR algorithms primarily consist of three main components: feature extraction, feature aggregation, and feature matching. Recently, feature extraction methods based on deep learning have made significant progress. Using convolutional neural networks (CNNs) [21]– [23] to extract intermediate layer features, or leveraging the Vision Transformer (ViT) [24] architecture to enhance modeling capabilities, it becomes more effective to extract both local features (e.g., feature points and descriptors) and global features (i.e., a single vector representation used for efficient retrieval). These approaches significantly outperform traditional handcrafted feature methods such as SIFT [25] and SURF [26]. 

In the feature aggregation stage, early statistical aggregation methods [27]–[29] were manually designed, lacked learning capabilities, and struggled to adapt to complex environmental changes. NetVLAD [30] addresses this by aggregating local features in an end-to-end trainable manner. GeM pooling [31] enhances robustness by applying power-average pooling to the CNN output feature maps. MixVPR [32] adopts feature concatenation and compression of multiple convolutional channels as the aggregation method, eliminating the need for local feature point calculations, thus offering faster and more efficient performance. Additionally, SALAD [33] proposes a new approach by formulating feature aggregation as an optimal transport problem, explicitly excluding less informative local features through a “dustbin” mechanism to improve tolerance against dynamic backgrounds and occlusions. There are also methods that incorporate attention mechanisms and transformer architectures, such as TransVPR [34] and R2Former [35], to enhance spatial modeling and context awareness. 

Feature matching primarily involves two strategies: coarse matching based on global descriptors and fine matching based on local features. Global descriptor matching, currently the mainstream due to its efficiency, involves extracting global vectors from each image [30]–[32] and retrieving the most relevant candidate images using vector similarity measures such as L2 norm or cosine similarity. To further improve matching accuracy, many methods incorporate fine matching strategies based on local features. These extract keypoints and local descriptors [25], [36], [37], and perform fine-grained comparisons of candidate images using spatial consistency verification techniques such as RANSAC or geometric transformations. Furthermore, advanced techniques such as sequence matching (e.g., SeqSLAM [38], FAB-MAP with Temporal Priors [39], [40]) match the current image with a continuous sequence of images, leveraging temporal redundancy to improve robustness and accuracy in dynamic environments. Although deep learning-based VPR methods have achieved excellent performance, they still face challenges in extreme scenarios such as strong lighting conditions or high-speed motion, where standard cameras are prone to glare and motion blur. Furthermore, most existing methods are based on static architectures and lack adaptive feature extraction mechanisms, which can limit their effectiveness in rapidly changing or event-rich environments. 

# B. Event-based Vision

Distinct from the synchronous integration of traditional shutters, event sensors operate on a bio-inspired, asynchronous paradigm, enabling robust perception in high-dynamic-range and high-speed scenarios [41]. This mechanism has catalyzed significant progress in fundamental vision tasks, often driven by the co-evolution of large-scale benchmarks and novel algorithms [42], [43]. In the field of object detection, MvHeat-DET [44] constructs the large-scale EvDET200K benchmark and proposes a Mixture-of-Experts (MoE) heat conduction backbone, which effectively models feature extraction as a thermal diffusion process to balance accuracy and efficiency. 


TABLE I COMPARISON OF REPRESENTATIVE EVENT-BASED DATASETS FOR VISUAL PLACE RECOGNITION. # DENOTES THE NUMBER OF THE CORRESPONDING ITEM. CL, RA, DT, NT REPRESENT CLEAR SKY, RAINY DAY, DAY TIME, AND NIGHT TIME, RESPECTIVELY.


<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Year</td><td rowspan="2">Sensor</td><td rowspan="2">Resolution</td><td rowspan="2">#Scene</td><td rowspan="2">Real</td><td rowspan="2">Text</td><td colspan="2">Weather</td><td colspan="2">Lighting</td></tr><tr><td>CL</td><td>RA</td><td>DT</td><td>NT</td></tr><tr><td>DDD17 [19]</td><td>2017</td><td>DAVIS 346</td><td>346×260</td><td>40</td><td>√</td><td>×</td><td>√</td><td>×</td><td>√</td><td>×</td></tr><tr><td>DDD20 [20]</td><td>2020</td><td>DAVIS 346</td><td>346×260</td><td>215</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Brisbane-Event-VPR [13]</td><td>2020</td><td>DAVIS 346</td><td>346×240</td><td>6</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>NeuroGPR [16]</td><td>2023</td><td>DAVIS 346</td><td>346×240</td><td>12</td><td>√</td><td>×</td><td>√</td><td>×</td><td>√</td><td>×</td></tr><tr><td>NYC-Event-VPR [17]</td><td>2024</td><td>Prophesee EVK4-HD</td><td>1280×720</td><td>16</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>EPRBench (Ours)</td><td>2026</td><td>Prophesee EVK4-HD</td><td>1280×720</td><td>1022</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Similarly, for long-term visual object tracking, AMTTrack [7] contributes the FELT dataset and introduces an Associative Memory Transformer incorporating Modern Hopfield Networks, which enables robust cross-modal retrieval and dynamic template updating to handle significant appearance changes. These works demonstrate that establishing highquality datasets coupled with semantically aware algorithms is pivotal for advancing event-based vision. In the specific domain of localization, existing VPR approaches generally fall into reconstruction-based and fusion-based categories. Early works like Ensemble-Event-VPR [13] adopt a reconstruction strategy, converting asynchronous events into intensity frames via variable time windows to leverage established visual descriptors. Conversely, Event-VPR [45] pioneers an end-toend learning framework, utilizing an Event Spike Tensor (EST) voxel grid to structure the raw stream, followed by a deep residual network and a modified VLAD layer for direct feature aggregation. In their work, most researchers adopt feature fusion methods to obtain fused features of traditional frames and events, thereby achieving superior visual place recognition performance. Different from their approaches, in our work, we propose a novel event-based visual place recognition benchmark dataset. 

# III. EPRBENCH DATASET

In this section, we will first introduce the protocols needed to follow when collecting our event videos. Then, we will focus on data collection and statistical analysis. 

# A. Protocols

In this work, we intend to develop an event-based benchmark dataset for visual scene recognition, providing a data foundation for future research. When constructing this benchmark dataset, we follow the following protocols: 1) Largescale: In the context of current human technological and artificial intelligence development, large-scale datasets have become crucial for the advancement of intelligent human life. To promote the progress of intelligent human life, the dataset we collected includes over 10k event sequences, a large-scale dataset comprising more than 1k urban scenes, and over 65k event frames. 2) Different Collection Approaches: We adopt two methods for data collection: the one uses vehicle-mounted event cameras to gather road scenes with wide field views, enabling the capture of rich semantic information in largescale environments; the other relies on manual data collection to capture fine-grained details in small-scale scenarios, supplementing information unattainable by vehicle-mounted systems to ensure comprehensive coverage of urban environments. 3) Multi-viewpoint and Multi-condition: During data acquisition, we captured scene information from multiple viewpoints for each scenario to ensure comprehensive contextual coverage. During the same time, the dataset was collected under various climatic and lighting conditions to introduce challenges, facilitating the construction of a high quality event dataset. 

# B. Data Collection and Statistical Analysis

This EPRBench dataset was acquired with a PROPHESEE EVK4-HD event camera with 1280 × 720 resolution. During the data collection period, we always followed the above protocols to ensure that we provide a dataset with event data that contains rich and challenging urban scenarios. We divide the urban scene data according to three main categories: Campus, Park, and Road; where Campus includes scenes inside closed and dense building groups, such as neighborhoods and schools; Park includes non-building open scenes that are not next to the city’s main roads, such as parks and attractions; and Road denotes scenes along the streets inside the city as well as scenes near the city’s highways. These three types of data almost cover most of the scenes in the city. The EPRBench dataset contains 13,109 event scene samples, where each event sample consists of five event frames, Campus which has 315 scenes containing 2,766 scene samples, Park which has 351 scenes containing 5,176 scene samples, and Road which has 356 scenes containing 5,167 scene samples. The dataset was randomly divided into training, validation, and testing subsets in the ratio 7:1:2, containing 9,220, 1,273, and 2,616 scene samples, respectively. 

# C. Scenario Text Generation

To augment the visual data with rich semantic information, we propose a multi-stage pipeline to generate specialized scenario descriptions, as illustrated in Fig. 3. The process consists of the following three core stages: 

Stage 1. Initial Chinese CoT Generation. In this stage, 50,000 image sample pairs are first constructed based on the original event data, with positive and negative samples each accounting for half. Subsequently, text generation is performed on the constructed image sample pairs based on a template prompting strategy, and the Doubao Pro-1.6 model 1 is selected for this generation task. Comparative verification with the output results of other mainstream generation models shows that the text generated by the Doubao Pro-1.6 model can more fully reflect the scene semantic information and has better finegrained feature capture capability. Finally, the initial Chinese Chain-of-Thought (CoT) [46] dataset is obtained through this stage. 

![](images/9aebe3ac80210e101e4290e30e1358a07165236af368e55d77cc210bc4baa075.jpg)



Fig. 2. Data samples from our newly proposed EPRBench dataset. We visualize two distinct samples (a, b) for each scene category (Campus, Park, Road). For every sample, three images from different viewpoints are presented, demonstrating the significant viewpoint variations contained in our dataset.


Stage 2. Cross-Lingual Translation and Refinement. The purpose of this stage is to perform cross-lingual conversion and quality optimization on the Chinese CoT dataset obtained in Stage 1, which is specifically divided into two sub-steps: 1) Translation stage: The NLLB-200-3B model [47] is adopted as the core translation model to convert the Chinese CoT data into English, obtaining the initial English CoT dataset; 2) Refinement stage: Considering that the initial translation results have quality issues such as expression deviations and grammatical errors, the DeepSeek [48] model is further introduced. Through constructing targeted prompt words, the model is guided to verify and correct the translation accuracy and language fluency of the text, and finally a high-quality English scene judgment CoT dataset is obtained. 

Stage 3. Model Fine-tuning and Inference. To generate scene description text with rich semantic information, this work constructs a scenario-specific generation model (scene model expert) by fine-tuning a basic vision-language model. In the experiment, Qwen2.5-VL-7B [49] is selected as the base model, and the Low-Rank Adaptation (LoRA) [50] finetuning strategy (i.e., Supervised Fine-Tuning, SFT) is adopted to fine-tune the parameters of the basic model using the highquality CoT dataset constructed in Stage 2. The fine-tuned model has efficient scene perception capability and can be used as a scene perception expert model to generate semantically rich description text for the scene images in the dataset. 

# D. Benchmark Baselines

To establish a comprehensive performance reference for event-based visual place recognition, we have selected 15 SOTA or representative VPR algorithms for evaluation on our proposed EPRBench dataset, including: 1) CNN-based Global Descriptors: CosPlace [51], MixVPR [32], and Eigen-Places [52]. These methods focus on extracting robust global representations from single-mode inputs using convolutional neural networks and have established strong benchmarks in traditional large-scale VPR tasks. 2) Transformer-based Local Aggregators: CricaVPR [53], BoQ [15], and SALAD [33]. These models leverage attention mechanisms or advanced matching-based aggregation to capture fine-grained spatial relationships. Specifically, CricaVPR [53] utilizes cross-image correlation, while SALAD [33] employs Sinkhorn-based optimal transport for robust feature alignment. 3) Re-ranking based Methods: R2Former [35], SelaVPR [54] and Pair-VPR [55]. This category includes two-stage pipelines that refine initial retrieval results through sophisticated matching schemes. R2Former [35] employs a global-to-local reranking scheme with Transformers, and SelaVPR [54] performs semantic label alignment to enforce consistency between queries and candidates, effectively filtering out visually similar but semantically distinct distractors. 

By benchmarking these diverse algorithms, we aim to demonstrate the unique challenges posed by the NYC-Event-VPR and EPRBench benchmarks, especially in scenarios involving extreme motion blur and modality shifts. 

# IV. METHODOLOGY

# A. Overview

The overall architecture of our proposed Semantic-Guided Multi-modal VPR (SG-VPR) framework is illustrated in Fig. 4. It operates as a dual-stream system that processes event streams and textual descriptions in parallel. To extract robust scene representations, we introduce a text-guided token selection strategy. This module leverages the semantic features from the text encoder to evaluate the relevance of visual tokens, applying a top-k strategy to sparsely sample significant regions while discarding task-irrelevant redundancy. The selected visual tokens are then fused with textual features to generate the final multi-modal representation for retrieval. Additionally, an auxiliary LLM decoder is employed to enhance feature interpretability. 

![](images/6748208c8673fef6bfac25345fc402937b3192af8f92b80bd4413f91d46e0019.jpg)


<table><tr><td>Chinese CoT
(Stage 1 Output)</td><td>Refined English CoT
(Stage 2 Output)</td><td>Final Application
(Stage 3 Output)</td></tr><tr><td>Input Pair: [image1,image2]
CoT: [&quot;场景描述&quot;: 图1画面中有…,图2描述了...&quot;
场景元素识别&quot;: 图1三个高耸的柱状结构位于前
景左侧...图2...画面右侧及前景底部分布有树木,整
体呈现…”差异分析”...图片1背景右侧为密集高层
建筑群,图片2背景后方形成连续城市天际线...“推
理结论&quot;: 两张图片的核心构成元素高度一致...支
持两张图片为同一场景]</td><td>Input Pair: [image1,image2]
CoT: [&quot;Scene Description&quot;: Image 1 shows..., Image 2 describes... &quot;Scene Element Recognition&quot;: In Image 1, three towering pillar structures are ... Image 2... trees are distributed on the right and bottom of the foreground... &quot;Difference Analysis&quot;: ...the Image 1 features a dense cluster of high-rise buildings, while the background of Image 2 ... &quot;Reasoning &amp; Conclusion&quot;: ... supporting the conclusion that both images represent the same scene.]</td><td>Scene name: @Park@013@09
The picture shows three tall columns, with a dense cluster of low-rise buildings behind them, and a horizontal line structure at the bottom.Overall, the image is composed of vertical and horizontal lines, with a clear distribution of elements in the upper and lower regions, and a sense of hierarchy...</td></tr></table>


Fig. 3. The overall pipeline of constructing the specialized scenario description model.We first synthesize Chinese CoT data from paired images, then rigorously translate and refine it into English using DeepSeek to ensure semantic accuracy. Finally, this curated dataset is used to fine-tune the Qwen model, enabling it to perform robust scene reasoning and description.


# B. Network Architecture

The overall architecture follows a progressive pipeline structured into four stages: Dual-stream Feature Encoding, Text-Guided Semantic Fusion, Multi-modal Spatial Pyramid Aggregation, and Auxiliary Interpretability Reasoning. 

• Dual-stream Feature Encoding. Since the raw event stream $\mathcal { E } ~ = ~ \{ e _ { k } \} _ { k = 1 } ^ { N }$ is asynchronous and spatially sparse, it is incompatible with standard vision backbones. We first employ a temporal aggregation strategy to convert the stream into a synchronous representation. Specifically, distinct events within a fixed time interval ∆T are accumulated based on their polarity pk and coordinates $( x _ { k } , y _ { k } )$ to generate a 2D event frame $\bar { F } \in \mathbb { R } ^ { H \times W }$ . This frame effectively collapses high-temporal motion information into a static spatial structure containing rich edge and texture details. Subsequently, F is fed into the DINOv2 [4] encoder, which is robust against illumination changes and requires no human annotations. The encoder patchifies the frame and extracts discriminative features, resulting in local patch embeddings $v _ { p a t c h } \in \mathbb { R } ^ { N \times D }$ and a global representation $v _ { c l s } \in \mathbb { R } ^ { D }$ , serving as the visual anchor for place recognition. In parallel, the textual stream processes the semantic description $S = \{ w _ { 1 } , \ldots , w _ { M } \}$ (e.g., geometric structures and landmarks) to provide semantic guidance. The raw text is first tokenized and padded to a fixed length L to form an input vector. We then utilize a frozen CLIP Text Encoder [56] to map these discrete tokens into a continuous semantic embedding space. We extract the output from the final transformer layer to obtain the word-level features $t _ { w } \in \mathbb { R } ^ { L \times D }$ and the sentence-level feature $t _ { s } \in \mathbb { R } ^ { D }$ , which are crucial for guiding the network to focus on task-relevant regions in the subsequent fusion stage. 

• Text-Guided Semantic Fusion. To facilitate deep interaction between the visual and textual modalities and ensure robustness against dynamic artifacts, we devise a hierarchical fusion framework. This module synchronizes features at two distinct granularities: Global Context Aggregation for holistic scene definition and Local Sparsification for fine-grained landmark enhancement. 

The global fusion module aims to align the high-level semantic representation of the scene with the holistic visual content. Given the global visual class token $\smash { \boldsymbol { v } _ { c l s } } \in \mathbb { R } ^ { D }$ extracted by the vision backbone and the sentence-level textual embedding $t _ { s } ~ \in \mathbb { R } ^ { D }$ , we project them into a unified latent space. We then employ a concatenation-based fusion mechanism to synthesize the multi-modal global context: 

$$
\mathbf {f} _ {\text { global }} = \mathcal {F} _ {m l p} \left(\operatorname{Concat} (v _ {c l s}, t _ {s})\right), \tag {1}
$$

where Concat(·, ·) denotes the concatenation operation, and $\mathcal { F } _ { m l p } ( \cdot )$ represents a Multi-Layer Perceptron that models the non-linear interaction between the heterogeneous modalities. This resulting vector $\mathbf { f } _ { g l o b a l }$ encapsulates the overall scene atmosphere, ensuring the network is conditioned on the global semantic prior before processing local details. 

While global fusion captures the gist of the scene, event streams often contain significant background clutter and dynamic interference. To address this, the Local Fusion module utilizes fine-grained word-level features $t _ { w }$ to guide the selection and enhancement of informative visual patches $v _ { p a t c h }$ . 

![](images/60e5d7b47eac466a23a5c20d41fa393df13084f7d26dc268323998e178cf150e.jpg)



Fig. 4. An overview of our proposed Semantic Guided VPR framework for event-based place recognition (SG-VPR). The dual-stream system processes event streams and textual descriptions in parallel, adopting a text-guided top-k token selection strategy to extract robust multi-modal representations, with an auxiliary LLM decoder enhancing feature interpretability for retrieval.


We first evaluate the semantic relevance of each visual patch $v _ { i }$ by computing its maximum cosine similarity with the word tokens $t _ { w } .$ , formulated as 

$$
m _ {i} = \max _ {j} (\frac {v _ {i} \cdot w _ {j} ^ {T}}{\| v _ {i} \| \| w _ {j} \|}). \tag {2}
$$

Based on these relevance scores, we employ a ratio-based selection strategy to identify the discriminative regions. Specifically, we retain the top ρ percent of patches (where $\rho = 0 . 2 5$ in our experiments), dynamically determining the selection count $k = \lfloor \rho \cdot N \rfloor$ ⌋. Let K denote the indices of these topk patches, which typically correspond to static landmarks described in the text. Subsequently, we perform semantic injection exclusively on these selected regions to enhance their representation, while keeping unselected background regions unchanged: 

$$
\hat {v} _ {i} = \left\{ \begin{array}{l l} v _ {i} + \alpha \cdot (v _ {i} \odot w _ {i d x _ {i}}) & \text { if } i \in \mathcal {K} \\ v _ {i} & \text { otherwise } \end{array} , \right. \tag {3}
$$

where $w _ { i d x _ { i } }$ denotes the word token most similar to patch $v _ { i } ,$ ⊙ represents element-wise interaction, and α is a learnable scaling factor. This adaptive fusion design ensures the network focuses its computation on the most semantically significant visual content, effectively discarding task-irrelevant noise. 

• Multi-modal Spatial Pyramid Aggregation. To synthesize a unified descriptor, we reshape the patch tokens into a 2D feature map $\mathbf { F } _ { m a p }$ and employ a modified spatial pyramid strategy. Unlike standard approaches, we substitute the global pooling layer with the multi-modal semantic anchor $\mathbf { f } _ { g l o b a l } ,$ while applying Generalized Mean (GeM) pooling [57] to the finer $2 \times 2$ and $3 \times 3$ grids to capture local geometry. The final representation $\mathbf { V } _ { f i n a l }$ is generated via concatenation: 

$$
\begin{array}{c} \mathbf {F} _ {\text {final}} = \operatorname{Concat} \left(\mathbf {f} _ {\text {global}}, \operatorname{GeM} _ {2 \times 2} (\mathbf {F} _ {\text {map}}), \right. \\ \left. \operatorname{GeM} _ {3 \times 3} (\mathbf {F} _ {\text {map}})\right) ^ {\prime}, \end{array} \tag {4}
$$

where Concat(·) denotes the channel-wise concatenation, and $\mathrm { G e M } _ { k \times k } ( \cdot )$ represents the aggregated feature vector derived from the $k \times k$ spatial partition. By integrating the global semantic anchor (aligned with textual descriptions) and the local geometric descriptors (derived from event streams), ${ \bf F } _ { f i n a l }$ effectively encapsulates both high-level linguistic priors and fine-grained visual structures. This comprehensive multi-modal embedding is directly employed for the final retrieval task, ensuring robustness against extreme environmental variations. 

• Auxiliary Interpretability Reasoning. We introduce an optional reasoning branch driven by a trained Scene Description Expert (a specialized LLM decoder). This module translates the aggregated multi-modal representation $\mathbf { F } _ { f i n a l }$ back into natural language, validating that the learned features preserve high-level semantic fidelity for human-understandable interpretation. 

# C. Loss Function

We train the proposed framework using a joint objective that combines metric learning for retrieval and contrastive learning for cross-modal alignment. 

• VPR Metric Loss. To optimize the retrieval performance, we employ the Multi-Similarity (MS) Loss [63], which adaptively weights hard positive and negative pairs. Let $S _ { i k } = \left. \mathbf { f } _ { i } , \mathbf { f } _ { k } \right.$ be the cosine similarity between descriptors. The loss is defined as: 

$$
\begin{array}{l} \mathcal {L} _ {m s} = \frac {1}{B} \sum_ {i = 1} ^ {B} \left\{\frac {1}{\alpha} \log \left[ 1 + \sum_ {k \in \mathcal {P} _ {i}} e ^ {- \alpha (S _ {i k} - \lambda)} \right] \right. \tag {5} \\ \left. + \frac {1}{\beta} \log \left[ 1 + \sum_ {k \in \mathcal {N} _ {i}} e ^ {\beta (S _ {i k} - \lambda)} \right] \right\}, \\ \end{array}
$$


TABLE II COMPARISON WITH OTHER SOTA VISUAL PLACE RECOGNITION MODELS.


<table><tr><td rowspan="2">Method</td><td rowspan="2">Source</td><td colspan="3">NYC-Event-VPR-Event</td><td colspan="3">NYC-Event-VPR-RGB</td><td colspan="3">EPRBench</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>01. Deep visual geo-loc [58]</td><td>CVPR&#x27;22</td><td>25.6</td><td>41.9</td><td>49.1</td><td>84.9</td><td>89.3</td><td>90.4</td><td>63.2</td><td>77.9</td><td>82.8</td></tr><tr><td>02. CosPlace [51]</td><td>CVPR&#x27;22</td><td>32.1</td><td>48.1</td><td>55.3</td><td>85.4</td><td>88.8</td><td>89.7</td><td>73.2</td><td>88.9</td><td>92.0</td></tr><tr><td>03. MixVPR [32]</td><td>WACV&#x27;23</td><td>45.7</td><td>63</td><td>68.9</td><td>88.3</td><td>91.0</td><td>91.5</td><td>90.2</td><td>93.7</td><td>95.6</td></tr><tr><td>04. EigenPlaces [52]</td><td>ICCV&#x27;23</td><td>28.5</td><td>43.4</td><td>51.2</td><td>81.4</td><td>91.7</td><td>93.9</td><td>81.1</td><td>86.0</td><td>89.0</td></tr><tr><td>05. R2Former [35]</td><td>CVPR&#x27;23</td><td>61.2</td><td>75.0</td><td>78.5</td><td>86.3</td><td>88.9</td><td>89.9</td><td>80.0</td><td>94.8</td><td>96.7</td></tr><tr><td>06. CricaVPR [53]</td><td>CVPR&#x27;24</td><td>30.3</td><td>48.2</td><td>55.0</td><td>88.3</td><td>92.2</td><td>92.9</td><td>92.6</td><td>95.3</td><td>96.5</td></tr><tr><td>07. SelaVPR [54]</td><td>ICLR&#x27;24</td><td>43.3</td><td>57.6</td><td>62.9</td><td>87.8</td><td>91.3</td><td>91.7</td><td>89.9</td><td>93.6</td><td>95.2</td></tr><tr><td>08. EMVP [59]</td><td>NeurIPS&#x27;24</td><td>45.7</td><td>64.1</td><td>70.9</td><td>85.5</td><td>90.4</td><td>91.4</td><td>88.7</td><td>93.9</td><td>95.7</td></tr><tr><td>09. SALAD [33]</td><td>CVPR&#x27;24</td><td>39.6</td><td>57.8</td><td>64.1</td><td>89.1</td><td>91.8</td><td>92.3</td><td>91.3</td><td>93.4</td><td>95.3</td></tr><tr><td>10. SuperVLAD [60]</td><td>NeurIPS&#x27;24</td><td>39.7</td><td>59.2</td><td>66.6</td><td>88.8</td><td>91.5</td><td>92.4</td><td>93.3</td><td>95.0</td><td>96.2</td></tr><tr><td>11. BoQ [15]</td><td>CVPR&#x27;24</td><td>38.9</td><td>56.7</td><td>63.6</td><td>83.2</td><td>88.1</td><td>89.1</td><td>93.1</td><td>94.7</td><td>95.7</td></tr><tr><td>12. ImAge [61]</td><td>NeurIPS&#x27;25</td><td>42.4</td><td>59.0</td><td>65.0</td><td>85.7</td><td>90.4</td><td>91.0</td><td>76.2</td><td>89.7</td><td>92.7</td></tr><tr><td>13. PRGS [62]</td><td>PR&#x27;25</td><td>43.8</td><td>62.1</td><td>69.3</td><td>82.8</td><td>88.6</td><td>89.9</td><td>87.7</td><td>93.9</td><td>96.9</td></tr><tr><td>14. Pair-VPR [55]</td><td>RA-L&#x27;25</td><td>64.0</td><td>75.2</td><td>78.4</td><td>89.4</td><td>91.9</td><td>92.4</td><td>91.9</td><td>95.7</td><td>95.8</td></tr><tr><td>15. FOL [14]</td><td>AAAI&#x27;25</td><td>55.5</td><td>68.2</td><td>72.5</td><td>85.1</td><td>89.3</td><td>89.9</td><td>93.8</td><td>96.0</td><td>97.1</td></tr><tr><td>16. SG-VPR (Ours)</td><td>-</td><td>57.5</td><td>74.1</td><td>78.9</td><td>86.5</td><td>90.9</td><td>92.0</td><td>94.3</td><td>96.1</td><td>97.1</td></tr></table>

where $\mathcal { P } _ { i }$ and ${ \mathcal { N } } _ { i }$ denote the positive and negative sample sets for anchor i. We set $\alpha = 1 . 0 , \beta = 5 0 , \lambda = 1 . 0$ . 

• Cross-modal Alignment Loss. To enforce semantic consistency, we apply a symmetric InfoNCE Loss [64] between the global visual token $v _ { c l s }$ and global textual token $t _ { s } \mathrm { : }$ 

$$
\begin{array}{l} \mathcal {L} _ {\text {con}} = - \frac {1}{2 B} \sum_ {i = 1} ^ {B} \left(\log \frac {e ^ {\langle \mathbf {v} _ {i} , \mathbf {t} _ {i} \rangle / \tau}}{\sum_ {j} e ^ {\langle \mathbf {v} _ {i} , \mathbf {t} _ {j} \rangle / \tau}} \right. \tag {6} \\ \left. + \log \frac {e ^ {\langle \mathbf {t} _ {i} , \mathbf {v} _ {i} \rangle / \tau}}{\sum_ {j} e ^ {\langle \mathbf {t} _ {i} , \mathbf {v} _ {j} \rangle / \tau}}\right) \\ \end{array}
$$

where $\tau = 0 . 0 7$ is the temperature parameter. The final loss is a weighted sum: 

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {m s} + \gamma \cdot \mathcal {L} _ {\text { con }}, \tag {7}
$$

where $\gamma = 0 . 1 5$ balances the two terms. 

# V. EXPERIMENTS

# A. Dataset and Evaluation Metric

To evaluate the effectiveness of our proposed method, we conducted extensive experiments on two datasets: NYC-Event-VPR [17] and EPRBench dataset. Specifically, considering that NYC-Event-VPR contains multi-modal data (both RGB and Event streams), we performed separate experiments on both its RGB and Event modalities to ensure a comprehensive assessment. Specifically, EPRBench is the first event-based VPR dataset to provide rich semantic annotations for each scene, facilitating the cross-modal alignment between sparse event streams and high-level textual descriptions. By incorporating complex environmental noise and drastic appearance changes, EPRBench establishes a rigorous and comprehensive benchmark for evaluating the robustness, interpretability, and viewpoint-invariance of multi-modal localization algorithms in challenging real-world scenarios. 

We employ Recall@N as the primary metric to evaluate the performance of our proposed method. For each query, we retrieve the N nearest candidates from the database based on the similarity of their feature descriptors. A retrieval is considered a true positive if at least one of the top-N retrieved samples shares the same location label as the query. The Recall@N score is then calculated as the percentage of successfully identified queries over the entire query set. Following the standard protocol in VPR research, we report the results for $N \in \{ 1 , 5 , 1 0 \}$ . This label-based evaluation effectively measures the model’s ability to maintain semantic and spatial consistency across diverse modalities such as text and event streams. 

# B. Implementation Details

We implement our framework using PyTorch on a server equipped with a single NVIDIA RTX 3090 GPU. During the training phase, all input images are resized to $2 2 4 \times 2 2 4$ and normalized using ImageNet statistics. The model is optimized using the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of $1 \times 1 0 ^ { - 3 }$ . To stabilize convergence, we employ a StepLR scheduler, decaying the learning rate by a factor of 0.5 every 3 epochs, with a batch size of 24. Following the efficient fine-tuning protocol of CricaVPR [53], we freeze the parameters of the DINOv2 [4] backbone and the CLIP [56] Text Encoder to preserve their pre-trained generalized knowledge. Only the lightweight Adapter modules, the task-specific projection layers, and our proposed Crossmodal Fusion module are updated. This strategy allows for rapid domain adaptation without compromising the robust pretrained feature space. More details can be found in our source code. 

# C. Comparison with Other SOTA Algorithms

As shown in Table II, it reports the retrieval performance on two benchmark datasets. Our proposed SG-VPR achieves consistent improvements over existing state-of-the-art methods across most metrics. 

![](images/84ddf944c415b25471cf13437ccbd68857fd0c30f05977d68d0264cc89293e6c.jpg)



Fig. 5. Visualization of qualitative results. The leftmost column shows the queries, followed by the retrieval results from our SG-VPR and four baselines. Green/Red borders denote success/failure. Our method achieves correct retrieval across all samples, outperforming competitors that struggle with large viewpoint shifts and environmental noise.


• Performance on EPRBench Dataset. We first evaluate the performance on our newly introduced EPRBench dataset to establish a baseline for future research. As shown in the rightmost column, our SG-VPR sets a strong state-of-the-art with 94.3% R@1. It outperforms competitive methods like SuperVLAD [60] (93.3%) and CricaVPR [53] (92.6%). This result not only validates the effectiveness of our semanticguided design but also demonstrates the high quality and distinctiveness of the proposed benchmark. 

• Performance on NYC-Event-VPR Dataset. The advantages of our method are more pronounced on the challenging NYC-Event-VPR dataset (Event modality), which features complex urban environments and severe sparsity. While R2Former [35] and Pair-VPR [55] achieve top performance using a two-stage retrieval method, our SG-VPR secures the second-best position with 57.5% R@1, strictly outperforming specialized VPR methods like CricaVPR [53] (30.3%) and MixVPR [32] (45.7%) by substantial margins of +27.2% and +11.8%, respectively. This result highlights that our method effectively mitigates the impact of noise and sparsity where traditional geometric-based methods fail. Furthermore, on the NYC-Event-VPR (RGB modality) split, SG-VPR maintains competitive performance (86.5% R@1), comparable to leading RGB-based methods, indicating the versatile generalization capability of our learned representations. 

# D. Ablation Study

• Analysis on Global and Local Semantics Fusion. As shown in Table III, we investigate the impact of semantic guidance at different scales. Introducing Global Fusion yields a 1.1% gain in R@1, indicating that holistic context helps disambiguate visually similar scenes. Meanwhile, Local Fusion provides a 1.2% improvement, proving its ability to 


TABLE III ABLATION STUDY OF THE PROPOSED FUSION MODULES. EXPERIMENTS ARE CONDUCTED ON OUR SELF-COLLECTED EPRBENCH DATASET. THE BEST RESULTS ARE HIGHLIGHTED IN BOLD.


<table><tr><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>Baseline (Vision Only)</td><td>92.6</td><td>95.3</td><td>96.5</td></tr><tr><td>+ Global Fusion (cls-fusion)</td><td>93.7</td><td>95.4</td><td>96.8</td></tr><tr><td>+ Local Fusion (patch-fusion)</td><td>93.8</td><td>95.8</td><td>96.8</td></tr><tr><td>+ Full Fusion (Global + Local)</td><td>94.3</td><td>96.1</td><td>97.1</td></tr></table>

mitigate dynamic interference in event streams by focusing on text-described landmarks. Most importantly, the Full Model achieves the best performance of 94.3%. This confirms that global alignment and local sparsification are complementary, collectively ensuring robustness against the extreme conditions in the EPRBench dataset. 

• Analysis on Top-k Selection Strategy. As reported in Table IV, we investigate the sensitivity of the selection threshold $\rho .$ Setting ρ to a low value (e.g., 0.20) causes a significant performance drop to 93.1%, indicating that an overly aggressive sparsification strategy results in the loss of critical structural cues. Conversely, increasing ρ to 0.30 leads to a slight decline (94.2%), suggesting that retaining excessive patches reintroduces task-irrelevant dynamic noise. The model achieves its peak performance of 94.3% at $\rho = 0 . 2 5$ , confirming that this ratio strikes the optimal balance between filtering event artifacts and preserving semantic landmarks. 

• Analysis Study on Loss Weight γ. The impact of the loss weight γ is summarized in Table V. By varying γ from 0.10 to 0.25, we observe that the retrieval performance reaches its peak at γ = 0.15 (94.3% R@1). This suggests that incorporating a moderate auxiliary contrastive loss enhances feature discriminability. Conversely, further increasing 

# Input

![](images/d0ed28834537d2a311123e5f303e2060fbca0a8562103e705cef8da3816f4a7b.jpg)


You are a scene reasoning assistant that analyzes images. Compare the two frames below and determine if they depict the same scene. Always respond in the exact format: <think>[your reasoning process]</think><d>yes/ no</d><answer>[final conclusion]</answer> 

![](images/f1f9c80a215b14471d3f0abc081ef078d38c4d4597a917a73911a275a2d657ac.jpg)



Frame1


![](images/959efb799ffcc4e6594d1aee281f44c78f3187d02edd57c00e585b52500ff66f.jpg)



Frame 2


<think>Okay, let's compare the two frames. Frame 1 shows a house with a sloping roof and a portico, with trees on the left.Frame 2 shows a similar house with a sloping roof and a chimney,with trees also on the left.The main structural elements (house type,roof,a roof shape,roof,and the location) are consistent.The differences,like the visibility of the chimney,the exact perspectioe of the portico,are likely due to changeS in viewpoint. The high-level semantic features align strongly. Therefore, they depict the same scene.</think><d>yes</d> 

<answer>Same Scene (T)</answer> 

![](images/c52c6d7e9a68494612ca1dcaa55f9a4f54e2466c938002d7b6d79435bdfcb386.jpg)


# Input

![](images/910a4601c53cf4ce508b33024f5dc854ece85a7b3567885d1e31033d47ebf9b1.jpg)


You are a scene reasoning assistant that analyzes images. Compare the two frames below and determine if they depict the same scene. Always respond in the exact format: <think>[your reasoning process]</think><d>yes/ no</d><answer>[final conclusion]</answer> 

![](images/6e22ef655fd34df999e421b04b1e747a0ae85d132835a3e02b02b44004c8a41e.jpg)



Frame1


![](images/0ae654332e67df78bbad1abce50534a9b075a4c7c60f5d78dd0d430ef6b78257.jpg)



Frame 2


<think>Okay, let's compare the two frames.Frame 1 depicts a residential house with a sloping roof and a visible portico.In contrast,Frame 2 shows a modern multi-story building with a flat roof and a repetitioe window grid pattern. Although both frames contain trees in the background, their main structural elements (building type,roof shape,and architectural style)are fundamentaly different. The high-level semantic features do not align. Therefore, they depict diferent scenes</think><d>no</ d> 

<answer>Different Scene (F)</answer> 

![](images/0ff4a6d91634d46ce7e53cca885d2f552ef55b7030c4e436b276036a997edb06.jpg)



Fig. 6. Visualization of the auxiliary reasoning process. As shown in the “think” blocks, the model provides step-by-step semantic analysis to justify the retrieval results. By identifying shared landmarks (top) or structural mismatches (bottom), it offers human-understandable explanations for both successful matches and rejected candidates.



TABLE IV SENSITIVITY ANALYSIS OF THE SELECTION RATIO ρ. WE EVALUATE THE IMPACT OF DIFFERENT TOP-k RATIOS ON THE EPRBENCH DATASET. THE DEFAULT SETTING IS ρ = 0.25.


<table><tr><td>Ratio ρ</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>0.15</td><td>93.9</td><td>95.9</td><td>96.6</td></tr><tr><td>0.20</td><td>93.1</td><td>95.6</td><td>96.7</td></tr><tr><td>0.25</td><td>94.3</td><td>96.1</td><td>97.1</td></tr><tr><td>0.30</td><td>94.2</td><td>95.6</td><td>96.4</td></tr></table>

γ to 0.20 or 0.25 results in performance drops to 93.8% and 93.6%, respectively. We attribute this to the auxiliary loss overwhelming the primary retrieval supervision at higher weights. Based on these findings, we fix γ = 0.15 for all subsequent experiments. 


TABLE V ANALYSIS STUDY ON THE HYPERPARAMETER γ. WE INVESTIGATE THE IMPACT OF THE WEIGHT γ IN THE LOSS FUNCTION (EQ. 7) ON THE EPRBENCH DATASET. THE BEST PERFORMANCE IS ACHIEVED WHEN γ = 0.15.


<table><tr><td>Hyperparameter γ</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>0.10</td><td>92.9</td><td>95.7</td><td>96.9</td></tr><tr><td>0.15</td><td>94.3</td><td>96.1</td><td>97.1</td></tr><tr><td>0.20</td><td>93.8</td><td>95.6</td><td>96.9</td></tr><tr><td>0.25</td><td>93.6</td><td>95.6</td><td>96.7</td></tr></table>

# E. Visualization

To provide an intuitive understanding of the proposed benchmark and our method’s superiority, we present qualitative visualizations in Fig. 2, Fig. 5, and Fig. 6. Fig. 2 first highlights the diversity of EPRBench, which features significant viewpoint variations across different scene categories, creating a rigorous testing ground. Complementing this, Fig. 5 confirms the effectiveness of our SG-VPR under these difficult conditions. As observed, while baseline methods frequently fail due to environmental noise or large viewpoint shifts, our model successfully leverages high-level textual priors to anchor retrieval on stable semantic landmarks, consistently achieving correct matches (green borders) where others fail. Furthermore, Fig. 6 demonstrates the interpretability of our proposed Scene Expert. As shown in the visualization, the model employs a Chain-of-Thought (CoT) mechanism to perform a step-by-step comparison between the query and the retrieved candidate. Instead of relying solely on opaque feature distances, the system explicitly identifies and aligns shared high-level semantic structures to verify semantic consistency. This process not only validates the correctness of the match under significant viewpoint changes but also provides transparent, human-understandable justifications for the retrieval decision. 

# F. Limitation Analysis

While the proposed EPRBench benchmark effectively mitigates the environmental constraints faced by RGB sensors (e.g., variations in lighting conditions), its reliance on a specifically fine-tuned scene expert model for text generation remains a core bottleneck in improving generalization capability. The currently used expert model is tailored and optimized for our specific data distribution, which greatly limits its direct transferability to open-world RGB scenarios. 

# VI. CONCLUSION

In this paper, in order to address the issue of lacking challenging benchmarks for Event-based VPR, we propose a large-scale Event-based VPR dataset named EPRBench, which includes 13k samples equipped with fine-grained scene descriptions. Complementing the data, we propose SG-VPR, which integrates high-level textual semantics into the visual retrieval pipeline. Through global alignment and local aggregation, our method effectively resolves the ambiguity caused by sparse event streams. Experimental results validate that SG-VPR achieves state-of-the-art performance and exceptional robustness against geometric deformations. Moving forward, we plan to extend our approach by exploring a General-Purpose Scene Expert, aiming to achieve unified scene understanding for broader cross-modal applications. 

# ACKNOWLEDGMENT

This work was supported by the National Natural Science Foundation of China under Grant U24A20342, 62102205. Anhui Provincial Natural Science Foundation-Outstanding Youth Project, 2408085Y032. The authors acknowledge the Highperformance Computing Platform of Anhui University for providing computing resources. 

# REFERENCES



[1] P. Yin, J. Jiao, S. Zhao, L. Xu, G. Huang, H. Choset, S. Scherer, and J. Han, “General place recognition survey: Toward real-world autonomy,” IEEE Transactions on Robotics, vol. 41, pp. 3019–3038, 2025. 





[2] Y. Wang, Y. Qiu, P. Cheng, and J. Zhang, “Hybrid cnn-transformer features for visual place recognition,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 3, pp. 1109–1122, 2022. 





[3] T. Jin, F. Lu, S. Hu, C. Yuan, and Y. Liu, “Edtformer: An efficient decoder transformer for visual place recognition,” IEEE Transactions on Circuits and Systems for Video Technology, 2025. 





[4] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research Journal, pp. 1–31, 2024. 





[5] F. Wang, F. Zhang, X. Wang, M. Wang, D. Huang, and J. Tang, “Evraindrop: Hypergraph-guided completion for effective frame and event stream aggregation,” arXiv preprint arXiv:2511.21439, 2025. 





[6] X. Wang, L. Jin, X. Lou, S. Wang, L. Chen, B. Jiang, and Z. Zhang, “Reasoningtrack: Chain-of-thought reasoning for long-term visionlanguage tracking,” arXiv preprint arXiv:2508.05221, 2025. 





[7] X. Wang, X. Lou, S. Wang, J. Huang, L. Chen, and B. Jiang, “Long-term visual object tracking with event cameras: An associative memory augmented tracker and a benchmark dataset,” arXiv preprint arXiv:2403.05839, 2025. 





[8] X. Wang, S. Wang, P. Shao, B. Jiang, L. Zhu, and Y. Tian, “Event stream based human action recognition: a high-definition benchmark dataset and algorithms,” arXiv preprint arXiv:2408.09764, 2024. 





[9] X. Wang, S. Wu, X. Cheng, C. Bi, J. Tang, and B. Luo, “Pedestrian attribute recognition via hierarchical cross-modality hypergraph learning,” arXiv preprint arXiv:2509.22331, 2025. 





[10] X. Wang, Q. Zhu, S. Wu, B. Jiang, and S. Zhang, “When person reidentification meets event camera: A benchmark dataset and an attributeguided re-identification framework,” arXiv preprint arXiv:2507.13659, 2025. 





[11] X. Wang, J. Jiang, Q. Chen, L. Chen, L. Zhu, Y. Wang, Y. Tian, and J. Tang, “Estr-cot: Towards explainable and accurate event stream based scene text recognition with chain-of-thought reasoning,” arXiv preprint arXiv:2507.02200, 2025. 





[12] A. J. Lee and A. Kim, “Eventvlad: Visual place recognition with reconstructed edges from event cameras,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021, pp. 2247– 2252. 





[13] T. Fischer and M. Milford, “Event-based visual place recognition with ensembles of temporal windows,” IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 6924–6931, 2020. 





[14] C. Wang, S. Chen, Y. Song, R. Xu, Z. Zhang, J. Zhang, H. Yang, Y. Zhang, K. Fu, S. Du et al., “Focus on local: Finding reliable discriminative regions for visual place recognition,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025, pp. 7536–7544. 





[15] A. Ali-bey, B. Chaib-draa, and P. Giguère, “BoQ: A place is worth a bag of learnable queries,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 17 794–17 803. 





[16] F. Yu, Y. Wu, S. Ma, M. Xu, H. Li, H. Qu, C. Song, T. Wang, R. Zhao, and L. Shi, “Brain-inspired multimodal hybrid neural network for robot place recognition,” Science Robotics, vol. 8, no. 78, p. eabm6996, 2023. 





[17] T. Pan, J. He, C. Chen, Y. Li, and C. Feng, “Nyc-event-vpr: A largescale high-resolution event-based visual place recognition dataset in dense urban environments,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 4657–4664. 





[18] ByteDance, “Doubao AI,” https://www.doubao.com/, 2026, [Online]. Available: https://www.doubao.com/ (Accessed on: 09-Feb-2026). 





[19] J. Binas, D. Neil, S.-C. Liu, and T. Delbruck, “Ddd17: End-to-end davis driving dataset,” arXiv preprint arXiv:1711.01458, 2017. 





[20] Y. Hu, J. Binas, D. Neil, S.-C. Liu, and T. Delbruck, “Ddd20 endto-end event camera driving dataset: Fusing frames and events with deep learning for improved steering prediction,” in 2020 IEEE 23rd international conference on intelligent transportation systems (ITSC). IEEE, 2020, pp. 1–6. 





[21] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778. 





[22] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in International Conference on Learning Representations (ICLR), 2015. 





[23] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in International Conference on Machine Learning (ICML). PMLR, 2019, pp. 6105–6114. 





[24] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, 





J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations (ICLR), 2021, pp. 1–23. 





[25] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International journal of computer vision, vol. 60, no. 2, pp. 91–110, 2004. 





[26] H. Bay, T. Tuytelaars, and L. Van Gool, “Surf: Speeded up robust features,” in European Conference on Computer Vision (ECCV). Springer, 2006, pp. 404–417. 





[27] J. Sivic and A. Zisserman, “Video google: A text retrieval approach to object matching in videos,” in IEEE International Conference on Computer Vision (ICCV). IEEE, 2003, pp. 1470–1477. 





[28] H. Jégou, M. Douze, C. Schmid, and P. Pérez, “Aggregating local descriptors into a compact image representation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2010, pp. 3304–3311. 





[29] F. Perronnin, J. Sánchez, and T. Mensink, “Improving the fisher kernel for large-scale image classification,” in European Conference on Computer Vision (ECCV). Springer, 2010, pp. 143–156. 





[30] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic, “Netvlad: Cnn architecture for weakly supervised place recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 5297–5307. 





[31] F. Radenovic, A. Iscen, G. Tolias, Y. Avrithis, and O. Chum, “Revisiting oxford and paris: Large-scale image retrieval benchmarking,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 5706–5715. 





[32] A. Ali-Bey, B. Chaib-Draa, and P. Giguere, “Mixvpr: Feature mixing for visual place recognition,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2023, pp. 2998–3007. 





[33] R. Mahjoub and S. Kasaei, “Salad: Self-supervised aggregation learning for adverse conditions in visual place recognition,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 11 518–11 524. 





[34] R. Hou, H. Zhao, J. Shi, G. Yu, and C. Wang, “Transvpr: Transformerbased visual place recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10 437–10 446. 





[35] J. Xu, Q. Liu, W. Huang, J. Wang, and Z. Li, “R2former: Robust and reliable transformer for visual place recognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 12 251–12 260. 





[36] D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superpoint: Selfsupervised interest point detection and description,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 224–236. 





[37] M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler, “D2-net: A trainable cnn for joint detection and description of local features,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 8092– 8101. 





[38] M. J. Milford and G. F. Wyeth, “Seqslam: Visual route-based navigation for everyday mobile robots,” in 2012 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2012, pp. 1643–1649. 





[39] M. Cummins and P. Newman, “Fab-map: Probabilistic localization and mapping in the space of appearance,” The International Journal of Robotics Research, vol. 27, no. 6, pp. 647–665, 2008. 





[40] M. Labbé and F. Michaud, “Appearance-based loop closure detection for online large-scale and long-term operation,” IEEE Transactions on Robotics, vol. 29, no. 3, pp. 734–745, 2013. 





[41] Z. Chen, J. Wu, J. Hou, L. Li, W. Dong, and G. Shi, “Ecsnet: Spatiotemporal feature learning for event camera,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 2, pp. 701–712, 2022. 





[42] X. Wang, C. Wang, S. Wang, X. Wang, Z. Zhao, L. Zhu, and B. Jiang, “Mambaevt: Event stream based visual object tracking using state space model,” IEEE Transactions on Circuits and Systems for Video Technology, 2025. 





[43] D. Liu, T. Wang, and C. Sun, “Voxel-based multi-scale transformer network for event stream processing,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 4, pp. 2112–2124, 2023. 





[44] X. Wang, Y. Jin, W. Wu, W. Zhang, L. Zhu, B. Jiang, and Y. Tian, “Object detection using event camera: A moe heat conduction based detector and a new benchmark dataset,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 29 321–29 330. 





[45] D. Kong, Z. Fang, K. Hou, H. Li, J. Jiang, S. Coleman, and D. Kerr, “Event-vpr: End-to-end weakly supervised deep network architecture 





for visual place recognition using event-based vision sensor,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–18, 2022. 





[46] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in neural information processing systems, vol. 35, pp. 24 824–24 837, 2022. 





[47] “Scaling neural machine translation to 200 languages,” Nature, vol. 630, no. 8018, pp. 841–846, 2024. 





[48] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan et al., “Deepseek-v3 technical report,” arXiv preprint arXiv:2412.19437, 2024. 





[49] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025. 





[50] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022. 





[51] G. Berton, C. Masone, and B. Caputo, “Rethinking visual geolocalization for large-scale applications,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 4868– 4878. 





[52] G. Berton, G. Trivigno, B. Caputo, and C. Masone, “Eigenplaces: Training viewpoint robust models for visual place recognition,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 11 046–11 056. 





[53] F. Lu, X. Lan, L. Zhang, D. Jiang, Y. Wang, and C. Yuan, “Cricavpr: Cross-image correlation-aware representation learning for visual place recognition,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 772–16 782. 





[54] F. Lu, L. Zhang, X. Lan, S. Dong, Y. Wang, and C. Yuan, “Towards seamless adaptation of pre-trained models for visual place recognition,” arXiv preprint arXiv:2402.14505, 2024. 





[55] S. Hausler and P. Moghadam, “Pair-vpr: Place-aware pre-training and contrastive pair classification for visual place recognition with vision transformers,” IEEE Robotics and Automation Letters, vol. 10, no. 4, pp. 4013–4020, 2025. 





[56] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763. 





[57] F. Radenovic, G. Tolias, and O. Chum, “Fine-tuning cnn image retrieval ´ with no human annotation,” IEEE transactions on pattern analysis and machine intelligence, vol. 41, no. 7, pp. 1655–1668, 2018. 





[58] G. Berton, R. Mereu, G. Trivigno, C. Masone, G. Csurka, T. Sattler, and B. Caputo, “Deep visual geo-localization benchmark,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5386–5397. 





[59] Q. Qiu, S. Zhang, H. Gao, H. Yang, H. Ying, W. Wang, and X. He, “Emvp: Embracing visual foundation model for visual place recognition with centroid-free probing,” Advances in Neural Information Processing Systems, vol. 37, pp. 120 928–120 950, 2024. 





[60] F. Lu, X. Zhang, C. Ye, S. Dong, L. Zhang, X. Lan, and C. Yuan, “Supervlad: Compact and robust image descriptors for visual place recognition,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 5789–5816. 





[61] F. Lu, T. Jin, C. Ye, X. Lan, Y. Liu, and C. Yuan, “Towards implicit aggregation: Robust image representation for place recognition in the transformer era,” in The Annual Conference on Neural Information Processing Systems, 2025. 





[62] W. Zuo, L. Liu, Y. Li, Y. Shen, F. Xiang, J. Xin, and N. Zheng, “Prgs: Patch-to-region graph search for visual place recognition,” Pattern Recognition, p. 111673, 2025. 





[63] X. Wang, X. Han, W. Huang, D. Dong, and M. R. Scott, “Multisimilarity loss with general pair weighting for deep metric learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 5022–5030. 





[64] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018. 

