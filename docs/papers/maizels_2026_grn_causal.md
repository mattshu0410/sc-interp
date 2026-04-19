Perspective

## Gene regulatory networks: from correlative models to causal explanations

Rory J. Maizels 1,2,3 &amp; James Briscoe 1

## Abstract

Gene regulatory networks (GRNs) explain how the genome controls cellular behaviour and tissue morphogenesis, serving to connect molecular mechanism to functional output. Single-cell technologies now provide descriptions of these networks with unprecedented detail, but this advance has also revealed gene regulatory systems that are too complex for our existing conceptual frameworks. GRNs, which should provide mechanistic explanations, are increasingly reduced to statistical correlations - 'hairballs' that fail to capture molecular causation. Here, we explore why this dilemma exists and propose a path forward. We argue that methods in 'representation learning' can be used to model GRNs, without needing to capture every molecular detail. For this framework, we advocate three linked principles: models must be inherently mechanistic, with structures grounded in cellular and evolutionary biology; molecular principles and constraints must be used to reduce the solution space for learning GRN models; and more sophisticated forms of experimental perturbation and synthetic biological engineering are needed to train models and test predictions. By reimagining GRNs through these principles, we can bridge the gap from data abundance to new conceptual understanding.

1 The Francis Crick Institute, London, UK. 2 EMBL-EBI, Hinxton, UK. 3 Wellcome Sanger Institute, Hinxton, UK.

[e-mail: James.Briscoe@crick.ac.uk](mailto:James.Briscoe@crick.ac.uk)

![Image](figure-omitted)

Sections

Introduction

The shortcomings of GRN models

Inherent challenges of modelling GRNs

Representational solutions

Solutions in fewer dimensions

Experimental solutions

The role of machine learning

Conclusion

## Perspective

## Introduction

'No observations on single genes can ever illuminate the overall mechanisms of development of the body plan or of body parts except at the minute and always partial, if not wholly illusory, level of the worm's eye view.' - Eric H. Davidson

When Nusslein-Volhard and Wieschaus first performed their systematic mutational screen of segmentation in Drosophila embryos 1-3 , they revealed that developmental processes such as body segmentation can be orchestrated by a surprisingly small number of genes. In the following decades, genetic knockout approaches have continued to build a deeper and more detailed understanding of the genetic toolbox driving tissue formation, leading to a call for a Perturbation Cell and Tissue Atlas 4 that will extend genetic screening to its logical conclusion of documenting the role of every gene in every tissue.

However, it is clear that studying development gene-by-gene is not enough: genes are not fixed entities with singular roles. Genes are not independent units of causality 5-7 ; they interact and communicate forming networks. It is through the dynamics of these networks, rather than through individual genes, that developmental form emerges.

Over the past half century, the concept of the gene regulatory network (GRN) has become central to developmental biology 6,8,9 . Broadly, a GRN is a system of molecular genetic regulators that act in concert to drive particular cellular outcomes. The basic form of a GRN is a set of transcription factors (TFs) that act at the cis -regulatory elements of other genes. Fundamentally, the GRN concept describes how regulatory genes operate together to control cell function and tissue morphogenesis, providing a dynamic map from genotype to phenotype 5 . These holistic descriptions can explain behaviours that could not be produced by a single gene, such as oscillations 10-12 , stripes 13-17 and switch-like responses 18-20 .

The original concept of GRNs was inherently causal, constructed through iterative experimental interventions on the genome 21 . However, as advances in genome-wide sequencing revealed the size and complexity of developmental GRNs - with dozens of TFs 22,23 binding thousands of sites across the genome 24,25 - the challenge of constructing GRNs purely through experimental intervention has become more evident. This obstacle has contributed to the rise of 'statistical' GRN models 26-29 that are built to infer genetic relationships from statistical patterns in the data, such as co-variance between genes across cells or samples. Switching focus from experimental intervention to statistical inference has led to methods that can better handle the complexity of development, but it has moved the field away from seeking a mechanistic understanding of how GRNs map from genotype to phenotype.

In their current states, the two modes of developmental genetics have converse strengths and weaknesses: classical single-gene knockouts provide strong causal evidence of genetic function but ignore the interconnected reality of developmental systems; by contrast, GRN-based approaches capture this vital context but increasingly provide only correlational evidence with minimal explanatory power. In this Perspective, we argue for a return to the view of GRNs as mechanistic explanations. We begin by overviewing the current challenges associated with computational GRN models from genomics datasets. To overcome these challenges while still embracing the big-data methods required to capture the full complexity of GRNs, we then outline how to potentially distill these high-dimensional datasets into clear explanations. First, we propose that causal representation learning 30 techniques can provide descriptions of gene regulation that are low-dimensional and inherently mechanistic. Second, we consider how existing knowledge of cellular and evolutionary biology can be used to ensure meaningful biological information is learned with representational models. Third, we explore experimental opportunities that can create new data for these models by engineering synthetic GRNs and generating cis -regulatory perturbations that expand the space of regulatory systems that we can explore. Last, we discuss how our efforts can be guided by previous successes in biological machine learning. Through the combined opportunities considered here, we can create models of gene regulation that can explain how, from complex molecular networks, the elegance of organismal form and function emerges.

## The shortcomings of GRN models

Conventionally, the construction of GRN models is an iterative process of systematically identifying regulatory genes and elements, charting out expression patterns and establishing regulatory interactions through genetic perturbation 31-33 (Fig. 1a,b). The advent of high-throughput sequencing technologies led to GRN inference algorithms that expedited this process (Fig. 1c) by capturing correlative patterns between genes across many samples (or many single cells), circumventing the need for piecewise reconstruction of regulatory connections (Fig. 1d).

Early methods, such as GENIE3 (ref. 34) and MRNET 35 ,  were designed for use with microarray or bulk RNA sequencing data, and they were benchmarked using synthetic datasets or simple bacterial datasets wherein a true network structure was known. The performance of these methods was quantified through comparison to these ground truths , with limited application to real biological questions. With the advent of single-cell RNA sequencing, which revealed the true extent of dynamism, stochasticity and heterogeneity of GRNs 36-39 , came a host of new methods, applying techniques from Bayesian modelling 40 to dynamical systems theory 41 , sometimes integrating pseudo-temporal 42 or RNA velocity 43 information. These methods saw greater biological application, for example, by identifying novel genes that appeared to be involved in disease states. However, independent benchmarking studies of GRN inference tools have revealed widespread poor performance 44 , with methods often performing no better than simple baselines or random guesswork 45 . Indeed, a recent study has demonstrated that, even with single-cell resolution, gene expression data alone is insufficient to control for false discoveries in GRN inference 46 .

The advent of multiomics approaches that simultaneously measure gene expression and chromatin accessibility promised exciting improvements 26 . Methods such as SCENIC+ 29 , Dictys 28 and CellOracle 27 use multimodal single-cell data to model GRNs, and they have been used to identify key TFs and important enhancer regions. In some cases, they have been used to predict the effect of perturbing well-studied differentiation driver genes or to find new important cell fate TFs 27 . However, despite the promise of multimodality, these data bring more issues for modelling. Inferring GRNs from chromatin accessibility requires two additional inference tasks for each chromatin region: inferring which TF binds to it and inferring which downstream gene it regulates. Predicting TF binding from sequence alone is not straightforward 47,48 , as binding motifs can be highly degenerate (for example, one TF can bind a range of sequences, whereas one motif can be bound by many TFs 49 ). Furthermore, an enhancer can be tens to hundreds of kilobases away from the gene that it regulates 50 . Perhaps as a result, recent independent benchmarking has revealed that

## Perspective

![Image](figure-omitted)

multimodal GRN inference methods have limited robustness, high sensitivity to user-supplied parameters, and performed poorly at perturbation-based causal predictions 51 .

## Inherent challenges of modelling GRNs

The basic formulation of a GRN model is a network graph, wherein each node is a gene, and each edge is an interaction between two genes. The challenge of GRN inference is clear from the nature of these graph structures: if interactions are directional and self-interactions included, the number of possible interactions in a network of n genes is n 2 , and the number of possible network topologies is 2 n 2 . As such, a ten-gene network has 100 possible interactions and more than 10 30 possible topologies. Systematically deleting each gene in the network generates only 11 observational conditions (10 knockouts and a wild type) but 100 interaction parameters need to be learnt. If, after experimentally testing each of these 100 interactions, one is 95% confident of each interaction estimation, this outcome still only gives 0.6% confidence in overall network structure. Making matters worse is the fact that GRNs are dynamic processes with time-dependent interactions and feedback loops 52 , which means that any static representation of a network (for example, a conventional graph of nodes and edges) would not necessarily capture the behaviour of the network 53,54 .

## Perspective

Theoretical work in systems biology and dynamical modelling has revealed a number of challenges faced when constructing models of complex systems (such as GRNs) that are only partially observed (as is the case with all biological datasets). Mathematical models can suffer from a phenomenon known as structural non-identifiability 55,56 , wherein different sets of model parameters generate the same output, making it impossible to determine the 'correct' parameter solution. A related but distinct 57 phenomenon is that of ' sloppy models' 58,59 , wherein certain parameters in a model can change by orders of magnitude without impacting the output of the model (Fig. 2c).

These phenomena are related to a more general problem of 'dynamical equivalence' 60,61 wherein different models generate equivalent dynamics, making the task of identifying the correct model structure intractable. It has been shown that many different GRN structures are capable of generating the same patterning behaviour 53,62,63 (Fig. 2a). Similarly, slight parameter variations with a single GRN structure can create very different model behaviours 10,64 (Fig. 2b). One cannot expect a one-to-one mapping between the structure of genetic interactions within a GRN and the behaviour of the GRN as a whole. These problems worsen as the number of parameters in the model increases.

Even an accurate GRN graph model would not provide a complete, objective depiction of the reality of gene regulation. Many aspects are ignored in these models - from spatiotemporal dynamics to epigenetic regulation to TF co-operativity, depending on the model. These aspects are deliberately ignored; they are abstracted with the assumption that they will be sufficiently captured by the parameters of the model. This abstraction is necessary: it makes the system   tractable for analysis.

Understanding this act of simplification allows us to ask whether our chosen level of abstraction captures the biological phenomena we wish to study. The choice to abstract molecular details and represent GRNs at the level of genetic interactions is based on the assumption that genes are the fundamental units of causality in cellular systems 5 . However, biological function can be 'emergent', arising from the dynamics and global structure of the GRN system itself 65-68 . Oscillations, switch-like behaviours and Turing patterns are examples of emergent properties that are not evident from individual genes.

There is no reason why the emergent properties of a GRN could not be described by an explicit and detailed model of every genetic component of the network. However, these explicit representations may not provide the most informative depiction of these complex systems 69,70 . Understanding every individual interaction may not provide a clear explanation of the cellular function of the GRN any more than a complete understanding of the structure of amino acids explains the folding of proteins. Indeed, given the evident challenges associated with constructing GRN models, it is worth considering whether abstracting away the details of genetic interactions would help us to learn more about the functions that GRNs perform in cells.

What molecular organization might be captured in an 'emergent property' of a GRN? It could be as simple as a quantification of the ratio of activities between two genes, rather than of the genes' activities themselves (for example, the erythroid-myeloid fate decision depends on the stoichiometric balance between GATA1 and PU.1 (ref. 71)). Other examples of emergent properties or mechanisms could include the following: many co-expressed genes that are induced in concert to engender a particular phenotype (such as the pigmentation gene module in melanocytes 72 ); sets of related or duplicated components (such as the different Gli proteins that transduce Sonic hedgehog signals 73 or the different enhancers that act cooperatively in the α-globin super-enhancer 74 ); components that are distinct molecules (for example, DNA sequence and proteins) but act collectively to drive a specific cell-level phenotype (such as the sharp, position-specific stripe of expression driven by the even-skipped stripe 2 system in Drosophila 75 or the interferon-β enhanceosome that integrates NF-κB and IRF signals in viral infection 76 ); or a sub-circuit, within a wider network of TFs, that is responsible for a specific phenotype (such as the four-gene network of Pax6 , Olig2 , Nkx2 -2 and Irx3 that drives ventral spinal cord patterning 73 ).

In each example, molecular components form a larger functional unit such that we could model the behaviour of the functional unit and abstract molecular details away. This coarse-grained approach could provide more robust models, revealing new forms of biological mechanism. It would provide a multi-perspectival way to study multi-scale biological systems, with different abstractions for different questions.

Fig. 2 | Challenges of GRN modelling. a , Different gene regulatory network (GRN) structures (three-node networks) can produce the same pattern of expression and tissue phenotype (represented as box of expression values through time and space). b , Conversely, the same GRN structure can produce different patterns depending on its parametrizations (strength of genegene interactions), contexts (boundary conditions) and initial conditions.

![Image](figure-omitted)

c , The challenge of 'sloppy models'. Top: in these cases, the model responds very sharply to changes in some parameters (stiff parameters) while hardly responding at all to others (sloppy parameters). Bottom: in parameter space, sloppy parameters can be visualized as directions in which the model output does not change; the contour map is unchanging (in this example, from bottom-left to top-right).

## Perspective

The challenge is figuring out how to do so in a flexible, generic way such that a single modelling approach would be suitable for all examples given above.

## Representational solutions

The challenge of modelling across different levels of granularity has been addressed in various scientific contexts. For example, in chemistry, coarse-grained modelling is used to produce molecular simulations that abstract away atomic information 77 , replacing particles with 'pseudo-particles' that only retain details relevant at the molecular or macromolecular level 78 . Comparably, AlphaFold2 (ref. 79) abstracts amino acid chains as 'triangular gases', retaining only the core   geometrical information required for modelling global protein structure.

However, coarse-graining analysis to focus on emergent properties can do more than just remove extraneous details. In studying signal processing system, higher levels of analysis can reveal broader design principles and functional structure. In neuroscience, Marr's levels of analysis 80,81 proposes three levels at which information processing systems can be understood. The highest level is the 'computational' level, which describes what problem is being solved by the system: the goals, constraints and success criteria. Next is the 'algorithmic' or 'representational' level, which describes how the system achieves its goal: how it processes inputs, builds useful representations, and uses these representations to create outputs. The lowest level is the 'implementational' level, which describes the physical realization of the system. Applying these levels of analysis to a radio 82 , we might say that the computational level of a radio is to deliver an audio programme to a listener. The implementational level will involve antennae, electronics, speakers, buttons, a power supply and so on. Linking these components is the algorithmic or representational level, which might describe how the radio selects a particular frequency with bandpass filtering, reads this signal with frequency discrimination, then processes this signal into data to send to speakers. The computational level describes why one might use a radio, the implementational level describes what a radio is composed of, but understanding how a radio works requires an understanding of the representational level.

This framework can similarly be applied to the signal processing performed by GRNs 83,84 : first, a cell-level, tissue-level or organism-level description of what processes a GRN controls, the phenotypes it drives, and the contexts in which it functions; second, a molecular level that describes the proteins, cis -regulatory elements and epigenetic components that construct the GRN; and last, to connect the first two, a representational level that describes how these different components organize, how input signals are mapped to output expression programmes 83 , and how this input-output mapping creates organismic function out of molecular components (Fig. 3a).

To build this representational layer, understanding how molecular components are organized is crucial. In this respect, GRNs have been shown to possess structure and organization 85,86 : task-specific sub-circuits provide a form of modularity 87,88 . These sub-circuits are organized in a hierarchical fashion that reflects the evolutionary and functional structure of the GRN 89 , whereas the sequential progression of metastable cell states through development creates another form of hierarchy between cell-state-specific sub-circuits. That hierarchy and modularity exist in this functional way, connecting to how GRNs operate to drive cellular decisions, suggests that GRN architectures are reducible and decomposable. Grouping genes into modules and structuring cell states into hierarchies naturally provides a bridge from genetic to cellular scales of function.

Next comes the question of how the system behaves. At the molecular level, behaviour is just the dynamics of components through time, perhaps extending to include interactions between components. However, a more systemic, representational idea of GRN behaviour must map the inputs to outputs of the system. What is the simplest model that can recapitulate outputs based on inputs? How might the activity of components link to this input-output map?

The third question (more relevant to GRNs than radios) relates to how the system evolved 90-92 . Throughout evolution, neutral or even mildly deleterious mutations can accrue 93,94 and cis -regulatory sequences vary considerably between species 95,96 , but mutations that impact the representational behaviour (and, thus, computational function) of GRN have higher tendency to have negative fitness effects. Many developmental GRNs are built around ancient, stable cores of TFs ('kernels' 9 or 'ChINs' 97,98 ) that drive tissue-specific developmental programmes. Feeding into these kernels are signalling input modules (termed 'plug-ins' or 'I/O switches'), which show greater variability across species, although they are often repeated across different contexts within an organism 9 . Responding to kernel activity are 'differentiation batteries', downstream effector genes that execute the output of the network without feeding back into the regulatory system 9 . These genes show the highest variability across species, as they are not constrained by downstream regulatory logic and can evolve to execute species-specific 'character states' of a tissue. Thus, although the components of a GRN can diverge between species, the systemic logic of the GRN system can remain conserved so that the GRN acts as the molecular and mechanistic basis of homology 99 . The human hand and the bird wing have distinct morphologies and functions; it is their homologous GRN kernels that reflects their shared evolutionary origin.

Just as the evolutionary dynamics of residues in a protein can provide structural information 100-102 , the evolutionary dynamics of genetic components could describe their role within the wider functional context of the GRN 103 . Over evolutionary time, GRN circuits can be co-opted into new developmental contexts 104 , whereas developmental systems drift through different network configurations to produce equivalent outputs, rewiring connections while maintaining overall function 105 . The constraints that guide this drift process, and the correlative patterns that are created by it, could provide valuable prior information for modelling the structure and function of GRNs in development 106-108 .

Taking a representational approach to modelling gene regulatory systems would help to reduce the solution space for models, but the benefit could be more fundamental. The approach moves from asking 'what is a GRN made of?' or 'what is the structure of a GRN?' to instead asking 'how does a GRN map inputs to outputs, and how does this achieve the cell's broader function?' In doing so, this approach can shift focus towards design principles, cellular function and evolutionary dynamics, connecting the study of GRN structure with fundamental questions of biological purpose and origin.

The parameters of such a representational model of gene regulation would not necessarily capture distinct molecular entities or properties, such as proteins or reaction kinetics. The challenge, then, is to find biological constraints that ensure that these models can be trained in a robust, principled way.

## Solutions in fewer dimensions

In fields such as single-cell genomics, it is already common practice to visualize biological systems with abstract representations. Dimensionality reduction approaches such as principal component analysis, uniform manifold approximation and projection 109 , and t -distributed

## Perspective

![Image](figure-omitted)

Fig. 3 | Representational descriptions of GRNs. a , Information processing description: Marr's levels of analysis 80,81 can be applied to the study of gene regulatory networks (GRNs). An implementational level captures the physical realization of the system, which, in the instance of GRNs, is the explicit description of transcription factors and enhancers that mediate genetic interactions. Above this level, a representational level describes the logic of how this physical realization functions to interpret signals (signal 1 and signal 2) and achieve the goals of the system, which is visualized here as logic gate connecting abstract cell-type factors. Finally, a computational level describes the computational process being performed, in this instance the decoding of input signals to form a striped tissue pattern. b , Cellular signal interpretation descriptions: from a cellular perspective, GRNs can be thought of as processes that take signalling dynamics as input and then output cell-type proportions. Constructing mechanistic models at the level of signal interpretation could describe the cellular function of GRNs without needing to explicitly model the underlying genetic interactions 174-176 . c , Evolutionary kernel description: GRNs consist of 'plug-ins', which are reusable modules, such as signalling pathways that provide inputs; 'kernels' that contain the core functional logic of the GRN; and 'differentiation batteries' that are responsible for executing the downstream consequences of the GRN. The different functions of these modules are reflected in their evolutionary dynamics: kernels are highly conserved across species owing to being functionally critical to tissue formation (visualized here as the unchanging blue network across species). Plug-ins display higher variability, particularly in the contexts in which they are deployed across tissues in the organism (demonstrated here as the changing size and strength of the different modules). Differentiation batteries display the highest level of variability; they do not feedback into the GRN and so are free to evolve and adapt to provide speciesspecific outputs. Capturing the evolutionary dynamics of GRN components could, thus, inform the functional role the components perform in the network.

## Perspective

stochastic neighbor embedding 110 condense the thousands of variables and observations into a more digestible two-dimensional depiction. Dimension reduction is also a common step in machine learning pipelines for statistical tasks such as multimodal data integration 111-114 , batch correction 115-117 and perturbation prediction 118-121 .

Although low-dimensional visualizations can introduce distortions into analysis 122 , the principle motivating these approaches is that the number of variables required to properly describe biological systems is considerably less than the number of features one can measure of it. In other words, biology exists in a lower-dimensional space than the full dimension of observable features (known beyond biology as the 'manifold hypothesis' 123 ). Biological features are correlated and interdependent, as the system is constrained to fewer degrees of freedom than observed variables. This phenomenon is a necessary feature of organized biological systems: the manifold hypothesis simply implies the presence of organization.

Low-dimensional representations of biology capture the correlations and patterns in biological systems that result from interactions between components. Building mechanistic models into these representations can, thus, learn the mechanisms by which these correlations and patterns are generated. Basic implementations of this more mechanistic form of dimension reduction exist already for single-cell data. Such cases include algorithms for describing the gene modules that capture the correlations between genes 124 , 'meta-cells' that  capture  the  coarse-grained  patterns  of  cell  states 125 ,  and pseudo-time and trajectory analysis tools 126 that can model the path of cellular differentiation that explains the observed patterns of cell types in a dataset.

As another example of mechanistic dimension reduction, methods of causal representation learning 30 aim to disentangle complex phenotypes into distinct biological processes and learn causal relationships from the data 127-136 . These approaches have been used with diverse datasets, including both simulated and real single-cell genomics data, and offer the prospect of more explainable, generalizable models of complex biological systems that are grounded in theory of causal discovery. These methods have thus far been applied largely to the problem of perturbation prediction: learning the causal effect of genetic and chemical interventions on cells.

Future work applying these methods could provide mechanistic representation of how gene regulation systems drive cell-level and tissue-level outcomes during development. Such a model would look to abstract away molecular details, shifting the onus of these complexities to the abstract parameters of a neural network, freeing the meaningful parameters of the model to learn a smaller number of latent causal factors that connect with or drive particular cellular phenotypes (Fig. 4). The challenge here would be to constrain the model such that what is being learnt is meaningful for the question. For example, one could enforce latent variables to map to genes of particular Gene Ontology terms, or to particular chromosomes, or that the mapping passes through a dynamical model of transcription. Model parameters could be defined to capture the observed dynamics of evolutionary sequence data, to represent the effects of specific perturbations or signalling conditions, and to capture cellular qualities such as proliferative rate or cell cycle stage, or they could be designed to capture particular GRN sub-circuits or motifs. The constraints determine what the model learns. If the evolution, cellular function or organismal application

Fig. 4 | Towards mechanistic abstract representations. a , In principal component analysis (PCA), each projected data point is a linear transformation of the original data point by a transformation matrix C . Accordingly, each component can be described as a linear combination of variables in the original dataset (for example, genes in RNA sequencing data). b , Dimension reduction methods such as autoencoders, uniform manifold approximation and projection, or t -distributed stochastic neighbor embedding provide a generalization of this idea beyond linear mappings, wherein each data point is mapped through a nonlinear function to a latent representation (to use the nomenclature of autoencoders). Hence, each latent variable can be described as a nonlinear function of the variables (genes) in the original dataset. c , A mechanistic adaptation in which the latent representation is constructed

![Image](figure-omitted)

from a mechanistic model ( f mech ) that captures the causal relationships between latent factors that can explain the dynamics of data points (for example, sequenced cells). This mechanistic model could be a dynamical system describing the time-dependent progression of cells transitioning between cell states. Concurrently, data variables (genes) are mapped to latent variables by a function that is subject to biological constraints ( f bio ), ensuring that the mapping is biologically meaningful. The structure of the latent representation and the mapping from variables to latent factors are interdependent but not equivalent (as with PCA). The mechanistic model can capture the causal relationships driving cell-level behaviours, whereas the latent variable mapping learns how genes connect to these causal relationships.

## Perspective

Fig. 5 | Establishing molecular principles governing GRNs. a , The structure of the English language can help constrain the solution space when determining whether a letter sequence is a valid word: both 'protein' and 'pertino' follow valid rules of phonetics, whereas the sequence 'rnpetoi' contains the invalid phonotactic cluster 'rnp', so it can be ruled out. Similarly, understanding the structure of allowed configurations at the molecular level can help constrain the solution space for modelling gene regulatory networks (GRNs) (here demonstrated with the example of transcription factor complexes forming at the promoter of a gene). b , Relationships can seem from the interaction between seemingly non-predictive or flexible variables to create a form of 'emergent

![Image](figure-omitted)

of GRNs dictates anything about their form or structure, building these details into our models can serve to reduce the vast solution space. Doing so in a representation learning framework can provide the 'higher-level' constraints that allow coarse-grained models to abstract the noise from the signal.

## Experimental solutions

Building a high-level understanding of gene regulatory function may not require mapping out every molecular component of a network. However, just as the challenge of dynamical equivalence means that many different model parameterizations can generate the same dynamics, many different molecular systems could generate equivalent mechanisms. Modelling gene regulation from measurements of molecular components in a way that is robust and generalizable across cell types, tissues, organisms and species requires understanding the molecular 'rules' that dictate gene regulatory function. The set of possible solutions for a gene regulatory model spans the space of possible molecular instantiations of the system, so to constrain the solution space for GRN models, we must understand what can and cannot happen at a molecular level.

As an example, consider the challenge of learning the ' cis -regulatory code' that maps the sequence of an enhancer to its function. An unbiased approach faces an intractably large solution space: there are more possible 200 base-pair sequences than the number of atoms in the universe. Defining biological principles and design constraints reduces this space to that of biologically plausible mechanisms. Just as knowing linguistic structures, such as syllables and phonemes, helps identify valid words in a language, understanding molecular principles organizing gene regulatory interactions provides a framework to link rigidity': here, neither variable 1 nor variable 2 show good correlation with the input variable; however, the product of these two variables resolves into a clear linear relationship. c , Variation in biological mechanism (for example, cis -regulatory enhancer activity) could be generated through two processes. Top: different enhancer activities or affinities can create dosage control between genes and between contexts, creating a functionally different output between contexts. Bottom: different enhancer activities or affinities can arise as an adaptation to an evolutionary event, such as the duplication of a gene. In this case, the variation does not create functional differences, but it compensates for the previous evolutionary event.

basic physical units (DNA bases and TFs) to their broader functional meaning (Fig. 5a).

'Big-data' methods such as single-cell genomics will be vital for describing patterns of gene regulation across contexts. However, understanding gene regulation requires moving beyond cataloguing and correlating these observations. It will be important to also capture how different regulatory layers connect.

Individual layers of regulation can give the impression of sloppy or noisy mechanisms: TF binding motifs are degenerate 49 , whereas enhancers are often redundant 137,138 . TFs seem to bind thousands of sites in the genome, often in complexes of interchangeable membership, sometimes binding alongside other TFs that they also directly antagonize 139,140 . The role of other forms of regulation, such as histone modifications, DNA methylation and non-coding RNAs, also seems to display context-dependent function 141 . This observation leads to an impression of extreme, almost unlimited flexibility, from which remarkable robustness and precision emerge.

Robustness may be achieved through interactions between regulatory layers. Variations in one layer may be coupled to, complemented by or counteracted by variations in another layer to produce a form of 'emergent rigidity' (Fig. 5b). For example, one set of observations might indicate that a gene is regulated by both distal and proximal enhancers such that the genomic distance of these regulatory elements is not predictive of their relative activities. Another set of observations might find that this gene is contained within a 3D topology that is remodelled between different cell types. Integrating these observations might lead to the finding that the 3D organization and genomic location of enhancers together resolve into a predictive model of cell-type-specific

## Perspective

enhancer activity 142 . Alternatively, one assay might document the sequence variability of motifs bound by a TF, whereas another assay might show that the TF induces the expression of targets that are not specific to a single cell type. Together, these findings could resolve into a model of dose-responsive TF binding, wherein motifs of different affinities drive the expression of different cell-type programmes (as has previously been shown to be the case for Sox2 (ref. 143)). Examining only one layer of regulation may never provide mechanistic understanding: the cis -regulatory code 140,144 of gene regulation may only exist in a vague sense when viewed solely from the lens of DNA sequence or TF activity; examining relationships between layers may allow a form of code to emerge more clearly.

Multimodal experimental designs linking perturbations in one layer to measurements of another will be valuable for dissecting a cis -regulatory code. Such experimental designs may include engineering sequence variation while measuring chromatin conformation with Hi-C, measuring chromatin accessibility changes in response to TF over-expression 145-148 , or recording enhancer activities across histone-code perturbations 149 . In combination, experimentally linking regulatory processes may reveal how the many facets of gene regulation act in concert to constrain cell behaviour.

Importantly, however, not all regulatory variation is necessarily functional. Variability can provide functional benefit (for example, binding motif degeneracy may allow the intensity of the effect of a TF to be modulated across contexts 143 ) but could also occur through evolutionary chance. For example, the duplication of a TF gene could lead to evolutionary changes in cis -regulatory sequences that accommodate two redundant regulators, rather than to removal of the duplicated gene (Fig. 5c). Equally, variation can be driven by stochastic processes of mutation and recombination that do not generate strong-enough selective forces to be eliminated through evolution 150 .

To test how structural aspects of a GRN maps to its function, we will need to rearrange the structure of these networks ourselves. By altering the composition and combination of cis -regulatory elements in a cell, we can begin to alter the strength, polarity or presence of interactions between genes, providing an experimental exploration of how different network structures generate various phenotypes. Work in this direction is ongoing: high-throughput enhancer mutagenesis 151 , engineered rearrangement of the human genome 152 and of enhancer landscapes 153 , high-throughput enhancer knockout 154 and TF induction screens 155 exemplify this direction. Concurrently, methods for designing cis -regulatory sequences towards specific regulatory functions are rapidly maturing, with enhancer screens capable of testing the cell-type-specific regulatory activity of thousands of DNA sequences simultaneously 156,157 , with detailed analyses of how structural recombination alters enhancer function 158 , with saturation genome editing methods that measure functional readouts of exhaustively altered regulatory regions 159 , and with machine learning methods for the de novo generation of cell-type-specific and function-specific enhancer sequences 160-162 .

These developments point towards a future of synthetic gene regulatory engineering, wherein cis -regulatory sequences and TFs can be edited or introduced to produce targeted alterations to the structure, and thus function, of GRNs. This, in turn, could create the possibility of designing entirely synthetic GRNs as objects of investigation, building on existing work in synthetic biology that has created programmable protein circuits 163 and protein-level neural networks 164 in mammalian cells, or ribocomputing devices 165 in bacteria, or genetic logic circuits 166 in yeast. Such efforts could provide gold-standard ground truth systems for building and benchmarking GRN modelling frameworks but, more broadly, would open up a vast space of GRN structures beyond that of naturally occurring systems.

The experimental progression outlined above moves from outlining molecular rules and the connections between layers of regulation in a network, to rearranging, redesigning and ultimately constructing new gene regulatory systems. This pathway offers the opportunity to build perturbation frameworks of commensurate sophistication to match the scale of our data-collecting abilities and the complexity of the systems we study. For theoreticians, it offers a particular enticing prospect: the dynamics of GRNs are, to a certain degree, written into the sequence of the genome. Editing GRNs through genome engineering, thus, offers a novel prospect for causal discovery, namely, that we can systematically alter the structure of causal interactions in the system we wish to understand.

This opportunity is reminiscent of a recently proposed strategy for understanding cis -regulatory DNA code. This strategy suggests that we should 'hold out the genome' 140 : the totality of naturally observed regulatory sequences is only a tiny proportion of the total possible sequence space, so to understand cis -regulatory sequences, we must train models on larger libraries of synthetic sequences, expanding beyond what is observed in nature. A similar argument applies to GRNs: the total space of possible network structures is far larger than is observed in biological systems. By building new synthetic systems and re-engineering the structures of existing ones, we can generate a far deeper and more extensive exploration of GRNs than is currently possible.

## The role of machine learning

The rapid progress of biological technologies, from genomics techniques to computational methods, has created considerable excitement regarding our understanding of gene regulatory mechanisms and cellular function (see Box 1 for open challenges). This momentum has led some to call for the construction of 'virtual cells' 167,168 and foundation models that provide 'universal representations' of cellular biology 4,169 .

One driving force for this excitement is the success of AlphaFold 79,170 , arguably the first bona fide biological foundation model.

## Box 1 | Open questions and challenges

- What forms of hierarchy, modularity and organization can be used to coarse-grain gene regulatory networks (GRNs)?
- What phenotypic and evolutionary data can inform on the structure and function of GRNs?
- At what levels of abstraction do causal relationships between GRNs and cell phenotypes become most apparent, and can we empirically detect such scale-dependent causality?
- How can we validate and benchmark abstract representational models of biological systems?
- How do perturbations at one layer of regulation affect the dynamics of other layers?
- Can we use synthetic biology to engineer ground truth synthetic GRN systems?
- Can we develop tools to engineer regulatory network structure with the precision we currently achieve for DNA and protein sequences?
- How does our picture of regulation change with single-molecular and time-resolved data?

## Perspective

## Glossary

## Bayesian modelling

A probabilistic modelling approach that combines prior beliefs (prior distributions) with observed data (likelihoods) to estimate an updated belief (posterior distributions), allowing inference and prediction with robust uncertainty quantification.

## Coarse-graining

A method commonly used in statistical mechanics and chemical simulation methods; details at a lower scale of resolution (for example, atomic level) are removed or averaged such that only features that are essential to preserve macroscopic behaviour at a higher scale (for example, bio-molecular) are retained.

## Dimension reduction

A class of methods used to reduce the number of variables in a dataset while retaining the major sources of variation in the data. Methods such as principal component analysis, uniform manifold approximation and projection, and t -distributed stochastic neighbor embedding are commonly used to reduce the dimension of sequencing data.

## Dynamical systems

Models of systems whose state evolves through time, often expressed in the form of differential equations; often used to describe biological systems such as population dynamics, biochemical reactions and gene regulatory networks.

## Ground truths

Real-world models or outcome of a system, serving as a benchmark against which trained models can be compared to assess performance.

## Marr's levels of analysis

A framework introduced by David Marr to describe cognitive and computational systems that breaks these systems into three layers: the computational level (what the system does and why), the algorithmic or representational level (organization of the system required to fulfill the computational purpose), and the implementational level (the physical material and substrates used to construct the algorithmic organization).

Creating a comparable model for gene regulatory systems requires one to shift scales from the molecular to the cellular. To appreciate the challenges associated with this shift, one must consider the differences between these contexts: crystal structure prediction is a static and clearly definable problem. It benefits from ground truths and robust metrics of success. Moreover, the Protein Data Bank (PDB) remains one of the cleanest and most consistent biological datasets ever constructed. Proteins display structural 'degeneracy', wherein structures contain commonly repeated motifs, such as α-helices and β-sheets,   making the challenge of structure prediction considerably more tractable.

By contrast, 'cell function' is a context-dependent concept with no objective definition, and 'cell states' can only ever be partially observed. Single-cell sequencing data are noisy, sparse and suffer from batch effects, which means that collected databases require considerable levels of data processing and transformation to be integrated 171 . Unlike protein structure, cellular decision-making is dynamic and context-dependent such that seemingly identical cells can behave differently owing to unobserved differences (for example, clonal dynamics, variation in cell culture conditions, and stochasticity of gene expression).

These differences highlight key areas wherein progress is required. For example, protein structure prediction models are not trained on raw data; X-ray diffraction densities are integrated, scaled and

## Multi-perspectival

The idea that the representation of a phenomenon may depend on the perspective of the viewer or the question of the researcher; the most useful representation of a gene regulatory system may differ, for example, for questions concerning global cell-type control versus questions concerning regulation of specific genes.

## Multi-scale

Models or analysis that represent phenomena across different scales of resolution in time, space or organization, for example, the atomic, molecular, biomolecular, genetic, cellular, organismal and population scales of biology.

## Representation learning

Related to dimension reduction, a field of machine learning focused on learning meaningful, compact representations of data rather than using only the variables observed in the original data; an example of representation learning is the variational autoencoder, a type of deep generative model that encodes data to a latent representation.

processed into atomic coordinates, which are then the input to these models. This process involves the incorporation of biological knowledge, specialized algorithms and human oversight to produce a standardized representation. Moving genomics harmonization methods, which currently focus only on removing batch effects, towards producing refined and biologically and biophysically motivated cell-state data representations, analogous to atomic coordinate data, may be key to building foundation model-worthy datasets. Moreover, as genomics methods become cheaper and more widely available, we must shift focus from cells-per-experiment towards samples-per-experiment to capture denser sampling of timepoints, signalling contexts and perturbation conditions. When we can record thousands of samples per experiment, rather than thousands of cells, we may start to collect datasets that display the same degeneracy of repeated patterns that has proven so fruitful for protein structure prediction and evolutionary sequence modelling 172 .

The continued growth of multimodal sequencing technologies could minimize the problem of cell states only ever being partially observed 26 , but these methods must provide robust, consistent datasets that can be continually reused, as we have seen with the PDB. It is notable that the year in which the PDB began was the same year as the publication of Perceptrons 173 , a pessimistic appraisal of the limited utility of neural nets as statistical models. In this year (1969), deep learning did not exist, the most sophisticated computers had only kilobytes of

## Sloppy models

Systems biology models that show extreme sensitivity to a small number of parameters, with the majority of parameters having no effect on model performance.

## Structural non-identifiability

When wide variation in the parameters of a model produce small changes in model output such that there is not a unique solution for fitting a model to a dataset.

## Perspective

memory, and there was no sense that PDB would provide the data for a computational solution to protein structure. We must similarly plan to generate datasets with the size, scope and quality to be used for years to come with computational methods that do not yet exist.

Even as we construct robust datasets and clear modelling objectives to replicate the environment of AlphaFold, we must recognize the fundamental differences between protein structure prediction and GRN modelling. AlphaFold is a predictive method; the goal is not to learn the biophysical principles that govern protein folding but to predict structure from amino acid sequence. The objective of a GRN model should be not only to predict the phenotypic consequence of a particular molecular genetic state but also to learn the principles that connect these two scales.

A black-box model predicting cell phenotypes is insufficient because interpretability is required. The challenge is that our conception of interpretability in biology exists largely at the molecular level: proteins and genes do things; they are the de facto mechanistic units in the cell 5 . Building an interpretable understanding at a systems-level will require new conceptual frameworks that define what a meaningful systems-level mechanism can look like. Such frameworks will need to exploit the organizational features of GRNs, such as their hierarchy and modularity. Unlike protein folding, GRNs may possess a representational layer that captures not only the molecular implementation but also the logic of how this implementation organizes in response to inputs to deliver outputs. Parsing this representational logic has the potential to reveal new design principles, entirely new forms of mechanism, and new views of how information is controlled in biology.

These features of GRNs - hierarchy, modularity, DNA rules and regulatory layers - point us towards how GRN models could be learnt. Hierarchy and modularity imply that GRNs are reducible, allowing molecular complexity to be abstracted away. If rules exist in DNA sequence, then sequence editing can redesign the rules, allowing exploration of a vast space of re-engineered systems. The organization that emerges from interactions between different regulatory layers may explain how robust cell fate decisions arise from seemingly noisy molecular  processes.  Evolutionary  analysis  can  discriminate between historically contingent patterns and fundamental rules of regulatory logic.

## Conclusion

Our current view of the GRN - a static graph of lines and nodes provides limited insight into the cellular function of regulatory networks. This fact is in stark contrast to the original formulation of GRNs as causal molecular explanations 85 . However, we are well positioned to return to a mechanistic view: single-cell genomics can measure biological phenotypes at huge scale and high resolution, machine learning can convert these data into simpler representations, and by building cellular and evolutionary constraints into these representations, we can distill the complexity of gene regulatory systems into the core logic of developmental processes. Ensuring that these models are meaningful will require experimental developments that set out the syntax and structure of molecular mechanisms, both in the principles of how they operate across contexts and the relationships that govern how different mechanisms interact. Synthetic biology methods that can build and manipulate gene regulatory systems will be transformative for this effort, providing a depth of understanding that would be impossible through examination of natural regulatory systems alone. Our goal in gathering more detailed datasets and building more sophisticated models should be to dismantle the complexity of GRNs and reveal the underlying design principles of developmental form and function.

Published online: xx xx xxxx

## References

1. Nüsslein-Volhard, C., Wieschaus, E. &amp; Kluding, H. Mutations affecting the pattern of the larval cuticle Indrosophila melanogaster : I. Zygotic loci on the second chromosome. Wilhelm Roux Arch. Dev. Biol. 193 , 267-282 (1984).
2. Jürgens, G., Wieschaus, E., Nüsslein-Volhard, C. &amp; Kluding, H. Mutations affecting the pattern of the larval cuticle Indrosophila melanogaster : II. Zygotic loci on the third chromosome. Wilhelm Roux Arch. Dev. Biol. 193 , 283-295 (1984).
3. Wieschaus, E., Nüsslein-Volhard, C. &amp; Jürgens, G. Mutations affecting the pattern of the larval cuticle Indrosophila melanogaster : III. Zygotic loci on the X-chromosome and fourth chromosome. Wilhelm Roux Arch. Dev. Biol. 193 , 296-307 (1984).
4. Rood, J. E., Hupalowska, A. &amp; Regev, A. Toward a foundation model of causal cell and tissue biology with a perturbation cell and tissue atlas. Cell 187 , 4520-4545 (2024).
5. DiFrisco, J. &amp; Jaeger, J. Genetic causation in complex regulatory systems: an integrative dynamic perspective. BioEssays 42 , e1900226 (2020).
6. Davidson, E. H. in Genomic Regulatory Systems (ed. Davidson, E. H.) Ch. 1 (Academic, 2001).
7. Davidson, E. H. &amp; Peter, I. S. in Genomic Control Process (eds Davidson, E. H. &amp; Peter, I. S.) Ch. 2 (Academic, 2015).
8. Davidson, E. H. &amp; Erwin, D. H. Gene regulatory networks and the evolution of animal body plans. Science 311 , 796-800 (2006).

This paper has established the foundational GRN concept within an evolutionary context and hierarchical organization.

9. Davidson, E. H. &amp; Erwin, D. H. Gene regulatory networks and the evolution of animal body plans. Science 311 , 796-800 (2006).
10. Perez-Carrasco, R. et al. Combining a toggle switch and a repressilator within the AC-DC circuit generates distinct dynamical behaviors. Cell Syst. 6 , 521-530.e3 (2018).
11. Pourquié, O. &amp; Goldbeter, A. Segmentation clock: insights from computational models. Curr. Biol. 13 , R632-R634 (2003).
12. Hirata, H. et al. Oscillatory expression of the bHLH factor Hes1 regulated by a negative feedback loop. Science 298 , 840-843 (2002).
13. Jaeger, J. The gap gene network. Cell. Mol. Life Sci. 68 , 243-74 (2011).
14. Sick, S., Reinker, S., Timmer, J. &amp; Schlake, T. WNT and DKK determine hair follicle spacing through a reaction-diffusion mechanism. Science 314 , 1447-1450 (2006).
15. Briscoe, J. &amp; Small, S. Morphogen rules: design principles of gradient-mediated embryo patterning. Development 142 , 3996-4009 (2015).
16. Raspopovic, J., Marcon, L., Russo, L. &amp; Sharpe, J. Modeling digits. Digit patterning is controlled by a Bmp-Sox9-Wnt Turing network modulated by morphogen gradients. Science 345 , 566-570 (2014).
17. Kondo, S. &amp; Asal, R. A reaction-diffusion wave on the skin of the marine angelfish Pomacanthus . Nature 376 , 765-768 (1995).
18. Collier, J. R., Monk, N. A., Maini, P. K. &amp; Lewis, J. H. Pattern formation by lateral inhibition with feedback: a mathematical model of delta-notch intercellular signalling. J. Theor. Biol. 183 , 429-446 (1996).
19. Li, C., Hong, T. &amp; Nie, Q. Quantifying the landscape and kinetic paths for epithelialmesenchymal transition from a core circuit. Phys. Chem. Chem. Phys. 18 , 17949-17956 (2016).
20. Tapscott, S. J. The circuitry of a master switch: Myod and the regulation of skeletal muscle gene transcription. Development 132 , 2685-2695 (2005).
21. Davidson, E. H. et al. A provisional regulatory gene network for specification of endomesoderm in the sea urchin embryo. Dev. Biol. 246 , 162-190 (2002).
22. Delile, J. et al. Single cell transcriptomics reveals spatial and temporal dynamics of gene expression in the developing mouse spinal cord. Development 146 , dev173807 (2019).
23. Rayon, T., Maizels, R. J., Barrington, C. &amp; Briscoe, J. Single-cell transcriptome profiling of the human developing spinal cord reveals a conserved genetic programme with human-specific features. Development 148 , dev199711 (2021).
24. Nishi, Y. et al. A direct fate exclusion mechanism by Sonic hedgehog-regulated transcriptional repressors. Development 142 , 3286-3293 (2015).
25. Peterson, K. A. et al. Neural-specific Sox2 input and differential Gli-binding affinity provide context and positional information in Shh-directed neural patterning. Genes Dev. 26 , 2802-2816 (2012).
26. Badia-I-Mompel, P. et al. Gene regulatory network inference in the era of single-cell multi-omics. Nat. Rev. Genet . 24 , 739-754 (2023).
27. Kamimoto, K. et al. Dissecting cell identity via network inference and in silico gene perturbation. Nature 614 , 742-751 (2023).
28. Wang, L. et al. Dictys: dynamic gene regulatory network dissects developmental continuum with single-cell multiomics. Nat. Methods 20 , 1368-1378 (2023).
29. González-Blas, C. B. et al. SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. Nat. Methods 20 , 1355-1367 (2023).

This paper is a key example of modern GRN modelling from multiomics single-cell data.

30. Scholkopf, B. et al. Toward causal representation learning. Proc. IEEE 109 , 612-634 (2021). This paper introduces the methods and applications of the field of causal representation learning.

## Perspective

31. Davidson, E. H. &amp; Levine, M. S. Properties of developmental gene regulatory networks. Proc. Natl Acad. Sci. USA 105 , 20063-20066 (2008).
32. Davidson, E. H. et al. A genomic regulatory network for development. Science 295 , 1669-1678 (2002).
33. Balaskas, N. et al. Gene regulatory logic for reading the Sonic Hedgehog signaling gradient in the vertebrate neural tube. Cell 148 , 273-84 (2012).
34. Huynh-Thu, V. A., Irrthum, A., Wehenkel, L. &amp; Geurts, P. Inferring regulatory networks from expression data using tree-based methods. PLoS ONE 5 , e12776 (2010).
35. Meyer, P. E., Kontos, K., Lafitte, F. &amp; Bontempi, G. Information-theoretic inference of large transcriptional regulatory networks. Eur. J. Bioinform. Syst. Biol. 2007 , 79879 (2007).
36. Maizels, R. J., Snell, D. M. &amp; Briscoe, J. Reconstructing developmental trajectories using latent dynamical systems and time-resolved transcriptomics. Cell Syst. 15 , 411-424.e9 (2024).

This paper combines time-resolved single-cell transcriptomics and deep learning models to reconstruct cell differentiation trajectories and cell fate decisions.

37. Cao, J. et al. The single-cell transcriptional landscape of mammalian organogenesis. Nature 566 , 496-502 (2019).
38. Mittnenzweig, M. et al. A single-embryo, single-cell time-resolved model for mouse gastrulation. Cell 184 , 2825-2842.e22 (2021).
39. Pijuan-Sala, B. et al. A single-cell molecular map of mouse gastrulation and early organogenesis. Nature 566 , 490-495 (2019).
40. Sanchez-Castillo, M., Blanco, D., Tienda-Luna, I. M., Carrion, M. C. &amp; Huang, Y. A Bayesian framework for the inference of gene regulatory networks from time and pseudo-time series data. Bioinformatics 34 , 964-970 (2018).
41. Matsumoto, H. et al. SCODE: an efficient regulatory network inference algorithm from single-cell RNA-Seq during differentiation. Bioinformatics 33 , 2314-2321 (2017).
42. Specht, A. T. &amp; Li, J. J. LEAP: constructing gene co-expression networks for single-cell RNA-sequencing data using pseudotime ordering. Bioinformatics 33 , 764-766 (2017).
43. Qiu, X. et al. Inferring causal gene regulatory networks from coupled single-cell expression dynamics using scribe. Cell Syst. 10 , 265-274.e11 (2020).
44. Pratapa, A., Jalihal, A. P., Law, J. N., Bharadwaj, A. &amp; Murali, T. M. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nat. Methods 17 , 147-154 (2020).
45. Chen, S. &amp; Mar, J. C. Evaluating methods of inferring gene regulatory networks highlights their lack of performance for single cell gene expression data. BMC Bioinformatics 19 , 232 (2018).
46. Kernfeld, E., Keener, R., Cahan, P. &amp; Battle, A. Transcriptome data are insufficient to control false discoveries in regulatory network inference. Cell Syst. 15 , 709-724.e13 (2024).
47. Wanniarachchi, D. V., Viswakula, S. &amp; Wickramasuriya, A. M. The evaluation of transcription factor binding site prediction tools in human and Arabidopsis genomes. BMC Bioinformatics 25 , 371 (2024).
48. Keilwagen, J., Posch, S. &amp; Grau, J. Accurate prediction of cell type-specific transcription factor binding. Genome Biol. 20 , 9 (2019).
49. Zandvakili, A., Campbell, I., Gutzwiller, L. M., Weirauch, M. T. &amp; Gebelein, B. Degenerate Pax2 and senseless binding motifs improve detection of low-affinity sites required for enhancer specificity. PLoS Genet 14 , e1007289 (2018).
50. Lettice, L. A. et al. A long-range Shh enhancer regulates expression in the developing limb and fin and is associated with preaxial polydactyly. Hum. Mol. Genet. 12 , 1725-1735 (2003).
51. Badia-i Mompel, P. et al. Comparison and evaluation of methods to infer gene regulatory networks from multimodal single-cell data. Preprint at bioRxiv https://doi.org/10.1101/ 2024.12.20.629764 (2025).

## This recent review identifies shortcomings and challenges of GRN modelling from multiome data.

52. Maizels, R. J. A dynamical perspective: moving towards mechanism in single-cell transcriptomics. Philos. Trans. R. Soc. Lond. B Biol. Sci. 379 , 20230049 (2024).
53. Cotterell, J. &amp; Sharpe, J. An atlas of gene regulatory networks reveals multiple threegene mechanisms for interpreting morphogen gradients. Mol. Syst. Biol. 6 , 425 (2010). This work demonstrates dynamical network structures that produce identical developmental patterns.
54. Isalan, M. Gene networks and liar paradoxes. BioEssays 31 , 1110-1115 (2009).
55. Wieland, F.-G., Hauber, A. L., Rosenblatt, M., Tönsing, C. &amp; Timmer, J. On structural and practical identifiability. Curr. Opin. Syst. Biol. 25 , 60-69 (2021).
56. Raue, A. et al. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. Bioinformatics 25 , 1923-1929 (2009).
57. Chis, O.-T., Villaverde, A. F., Banga, J. R. &amp; Balsa-Canto, E. On the relationship between sloppiness and identifiability. Math. Biosci. 282 , 147-161 (2016).
58. Gutenkunst, R. N. et al. Universally sloppy parameter sensitivities in systems biology models. PLoS Comput. Biol. 3 , e189 (2007).
59. Transtrum, M. K. et al. Sloppiness and emergent theories in physics, biology, and beyond. J. Chem. Phys. 143 , 010901 (2015).
60. Gábor, A., Hangos, K. M., Banga, J. R. et al. Reaction network realizations of rational biochemical systems and their structural properties. J. Math. Chem. 53 , 1657-1686 (2015).
61. Babtie, A. C., Kirk, P. &amp; Stumpf, M. P. H. Topological sensitivity analysis for systems biology. Proc. Natl Acad. Sci. USA 111 , 18507-18512 (2014).
62. Schaerli, Y. et al. Synthetic circuits reveal how mechanisms of gene regulatory networks constrain evolution. Mol. Syst. Biol. 14 , e8102 (2018).
63. Otero-Muras, I., Perez-Carrasco, R., Banga, J. R. &amp; Barnes, C. P. Automated design of gene circuits with optimal mushroom-bifurcation behavior. iScience 26 , 106836 (2023).
64. Jiménez, A., Cotterell, J., Munteanu, A. &amp; Sharpe, J. A spectrum of modularity in multi-functional gene circuits. Mol. Syst. Biol. 13 , 925 (2017).
65. Bhalla, U. S. &amp; Iyengar, R. Emergent properties of networks of biological signaling pathways. Science 283 , 381-387 (1999).

## This paper provides an example of emergent functionality in biological networks.

66. Yurchenko, S. B. Is information the other face of causation in biological systems? BioSystems 229 , 104925 (2023).
67. Artime, O. &amp; De Domenico, M. From the origin of life to pandemics: emergent phenomena in complex systems. Philos. Trans. R. Soc. A 380 , 20200410 (2022).
68. Ellis, G. F. Top-down causation and emergence: some comments on mechanisms. Interface Focus 2 , 126-140 (2012).
69. Klein, B. &amp; Hoel, E. The emergence of informative higher scales in complex networks. Complexity https://doi.org/10.1155/2020/8932526 (2020).
70. Hoel, E. When the map is better than the territory. Entropy 19 , 188 (2017).
71. Zhang, P. et al. Negative cross-talk between hematopoietic regulators: GATA proteins repress PU.1. Proc. Natl Acad. Sci. USA 96 , 8705-8710 (1999).
72. Levy, C., Khaled, M. &amp; Fisher, D. E. MITF: master regulator of melanocyte development and melanoma oncogene. Trends Mol. Med. 12 , 406-414 (2006).
73. Frith, T . J., Briscoe, J. &amp; Boezio, G. L. in Current Topics in Developmental Biology Vol. 159 (ed. Mallo, M.) Ch. 5 (Elsevier, 2024).
74. Kassouf, M. T. et al. The α -globin super-enhancer acts in an orientation-dependent manner. Nat. Commun. 16 , 1033 (2025).
75. Small, S., Blair, A. &amp; Levine, M. Regulation of even-skipped stripe 2 in the Drosophila embryo. EMBO J. 11 , 4047-4057 (1992).
76. Panne, D., Maniatis, T. &amp; Harrison, S. C. An atomic model of the interferon-β enhanceosome. Cell 129 , 1111-1123 (2007).
77. Kmiecik, S. et al. Coarse-grained protein models and their applications. Chem. Rev. 116 , 7898-7936 (2016).
78. Ingólfsson, H. I. et al. The power of coarse graining in biomolecular simulations. Wiley Interdiscip. Rev. Comput. Mol. Sci. 4 , 225-248 (2014).
79. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596 , 583-589 (2021).
80. Marr, D. Vision : A Computational Approach (MIT Press, 1982).
81. Marr, D. C. &amp; Poggio, T. From Understanding Computation to Understanding Neural Circuitry (MIT Artificial Intelligence Laboratory, 1976).
17. Lazebnik, Y. Can a biologist fix a radio? - or, what I learned while studying apoptosis.
82. Cancer Cell 2 , 179-182 (2002).
83. Pezzotta, A. &amp; Briscoe, J. Optimal control of gene regulatory networks for morphogen-driven tissue patterning. Cell Syst. 14 , 940-952.e11 (2023).
84. Tkačik, G. &amp; Wolde, P. R. T. Information processing in biochemical networks. Annu. Rev. Biophys. 54 , 249-274 (2025).
85. Peter, I. S. &amp; Davidson, E. H. Evolution of gene regulatory networks controlling body plan development. Cell 144 , 970-985 (2011).
86. Hatleberg, W. L. &amp; Hinman, V. F. in Current Topics in Developmental Biology Vol. 141 (ed. Gilbert, S. F.) Ch. 2 (Elsevier, 2021).

## This paper explores modularity and hierarchy in GRNs, which can help understand their developmental function.

87. Alon, U. Network motifs: theory and experimental approaches. Nat. Rev. Genet. 8 , 450-461 (2007).
88. Lorenz, D. M., Jeng, A. &amp; Deem, M. W. The emergence of modularity in biological systems. Phys. Life Rev. 8 , 129-160 (2011).
89. Erwin, D. H. &amp; Davidson, E. H. The evolution of hierarchical gene regulatory networks. Nat. Rev. Genet. 10 , 141-148 (2009).
90. Krumlauf, R. Hox genes in vertebrate development. Cell 78 , 191-201 (1994).
91. Shubin, N., Tabin, C. &amp; Carroll, S. Deep homology and the origins of evolutionary novelty. Nature 457 , 818-823 (2009).
92. Carroll, S. B. Evo-devo and an expanding evolutionary synthesis: a genetic theory of morphological evolution. Cell 134 , 25-36 (2008).
93. Ohta, T. Slightly deleterious mutant substitutions in evolution. Nature 246 , 96-98 (1973).
94. Lynch, M. &amp; Conery, J. S. The origins of genome complexity. Science 302 , 1401-1404 (2003).
95. Wray, G. A. The evolutionary significance of cis-regulatory mutations. Nat. Rev. Genet. 8 , 206-216 (2007).
96. Schmidt, D. et al. Five-vertebrate ChIP-seq reveals the evolutionary dynamics of transcription factor binding. Science 328 , 1036-1040 (2010).
97. Wagner, G. P. The developmental genetics of homology. Nat. Rev. Genet. 8 , 473-479 (2007).
98. Wagner, G. P. Homology, Genes, and Evolutionary Innovation (Princeton Univ. Press, 2014).
99. DiFrisco, J., Love, A. C. &amp; Wagner, G. P. Character identity mechanisms: a conceptual model for comparative-mechanistic biology. Biol. Philos. 35 , 44 (2020).
100.  Marks, D. S. et al. Protein 3D structure computed from evolutionary sequence variation. PLoS ONE 6 , 1-20 (2011).
101.  Marks, D. S., Hopf, T. A. &amp; Sander, C. Protein structure prediction from sequence variation. Nat. Biotechnol. 30 , 1072-1080 (2012).
102.  Hopf, T. A. et al. Mutation effects predicted from sequence co-variation. Nat. Biotechnol. 35 , 128-135 (2017).

## Perspective

103.  Hecker, N. et al. Enhancer-driven cell type comparison reveals similarities between the mammalian and bird pallium. Science 387 , eadp3957 (2025).
104.  McQueen, E. &amp; Rebeiz, M. in Gene Regulatory Networks (ed. Peter, I. S.) 375-405 (Academic, 2020).
105.  Halfon, M. S. Perspectives on gene regulatory network evolution. Trends Genet. 33 , 436-447 (2017).
106.  Booth, H. &amp; Hadjivasiliou, Z. Gene network organization, mutation, and selection collectively drive developmental pattern evolvability and predictability. PRX Life 3 , 033023 (2025).
107.  Pavlicev, M., Cheverud, J. M. &amp; Wagner, G. P. A model of developmental evolution: selection, pleiotropy and compensation. Trends Ecol. Evol. 27 , 316-322 (2012).
108.  McColgan, Á &amp; DiFrisco, J. Understanding developmental system drift. Development 151 , dev203054 (2024).
109.  McGinnis, C. S. et al. MULTI-seq: sample multiplexing for single-cell RNA sequencing using lipid-tagged indices. Nat. Methods 16 , 619-626 (2019).
110. van der Maaten, L. J. P. &amp; Hinton, G. E. Visualizing data using t-SNE. J. Mach. Learn. Res. 9 , 2579-2605 (2008).
111. Ternes, L. et al. A multi-encoder variational autoencoder controls multiple transformational features in single-cell image analysis. Commun. Biol. 5 , 255 (2022).
112. Wani, S. A., Khan, S. A. &amp; Quadri, S. scJVAE: a novel method for integrative analysis of multimodal single-cell data. Comput. Biol. Med. 158 , 106865 (2023).
113. Kalafut, N. C., Huang, X. &amp; Wang, D. Joint variational autoencoders for multimodal imputation and embedding. Nat. Mach. Intell. 5 , 631-642 (2023).
114. Gong, B., Zhou, Y. &amp; Purdom, E. Cobolt: integrative analysis of multimodal single-cell sequencing data. Genome Biol. 22 , 351 (2021).
115. Danino, R., Nachman, I. &amp; Sharan, R. Batch correction of single-cell sequencing data via an autoencoder architecture. Bioinform. Adv. 4 , vbad186 (2024).
116. Tran, H. T. N. et al. A benchmark of batch-effect correction methods for single-cell RNA sequencing data. Genome Biol. 21 , 12 (2020).
117. Zhang, Z. et al. scDisInFact: disentangled learning for integration and prediction of multi-batch multi-condition single-cell RNA-sequencing data. Nat. Commun. 15 , 912 (2024).
118. Kana, O. et al. Generative modeling of single-cell gene expression for dose-dependent chemical perturbations. Patterns 4 , 100817 (2023).
119. Yang, Y. et al. The Manatee variational autoencoder model for predicting gene expression alterations caused by transcription factor perturbations. Sci. Rep. 14 , 11794 (2024).
120.  Tang, Z., Zhou, M., Zhang, K. &amp; Song, Q. scPerb: predict single-cell perturbation via style transfer-based variational autoencoder. J . Adv . Res . 75 , 189-198 (2024).
121. Rampášek, L., Hidru, D., Smirnov, P., Haibe-Kains, B. &amp; Goldenberg, A. Dr.VAE: improving drug response prediction via modeling of drug perturbation effects. Bioinformatics 35 , 3743-3751 (2019).
122.  Chari, T. &amp; Pachter, L. The specious art of single-cell genomics. PLoS Comput. Biol. 19 , e1011288 (2023).
123.  Gorban, A. N. &amp; Tyukin, I. Y. Blessing of dimensionality: mathematical foundations of the statistical physics of data. Philos. Trans. R. Soc. A 15 , 20170237 (2018).
124.  Kunes, R. Z. et al. Supervised discovery of interpretable gene programs from single-cell data. Nat. Biotechnol. 42 , 1084-1095 (2024).
125.  Li, Y . et al. MetaQ: fast, scalable and accurate metacell inference via single-cell quantization. Nat. Commun. 16 , 1205 (2025).
126.  Saelens, W., Cannoodt, R., Todorov, H. &amp; Saeys, Y. A comparison of single-cell trajectory inference methods. Nat. Biotechnol. 37 , 547-554 (2019).
127. Lopez, R. et al. Learning causal representations of single cells via sparse mechanism shift modeling. In Proc. 2nd Conference on Causal Learning and Reasoning (eds van der Schaar, M. et al.) 662-691 (PMLR, 2023).

## This paper is an early example of the application of causal representation learning in biology.

128.  Zhang, J. et al. Identifiability guarantees for causal disentanglement from soft interventions. In Proc. Advances in Neural Information Processing Systems 36 (eds Oh, A. et al.) (NeurIPS, 2023).
129.  Gao, Y., Dong, K., Shan, C., Li, D. &amp; Liu, Q. Causal disentanglement for single-cell representations and controllable counterfactual generation. Nat. Commun. 16 , 6775 (2025).
130.  An, S. et al. scCausalVI disentangles single-cell perturbation responses with causality-aware generative model. Cell Syst. 16 , 101443 (2025).
131. Bereket, M. &amp; Karaletsos, T. Modelling cellular perturbations with the sparse additive mechanism shift variational autoencoder. In Proc. Advances in Neural Information Processing Systems 36 (eds Oh, A. et al.) (NeurIPS, 2023).
132.  Baek, S. et al. CRADLE-VAE: enhancing single-cell gene perturbation modeling with counterfactual reasoning-based artifact disentanglement. In Proc. Thirty-Ninth AAAI Conference on Artificial Intelligence 15445-15452 (AAAI Press, 2025).
133.  Mao, H. et al. Learning identifiable factorized causal representations of cellular responses. In Proc. Advances in Neural Information Processing Systems 37 (eds Globerson, A. et al.) (NeurIPS, 2024).
134.  Tejada-Lapuerta, A. et al. Causal machine learning for single-cell genomics. Nat. Genet. 57 , 797-808 (2025).
135.  Aliee, H., Kapl, F., Hediyeh-Zadeh, S. &amp; Theis, F. J. Conditionally invariant representation learning for disentangling cellular heterogeneity. Preprint at https://doi.org/10.48550/ arXiv.2307.00558 (2023).
136.  Lopez, R., Hütter, J.-C., Pritchard, J. K. &amp; Regev, A. Large-scale differentiable causal discovery of factor graphs. In Proc. Advances in Neural Information Processing Systems 35 (eds Koyejo, S. et al.) (NeurIPS, 2022).
137.  Kvon, E. Z., Waymack, R., Gad, M. &amp; Wunderlich, Z. Enhancer redundancy in development and disease. Nat. Rev. Genet. 22 , 324-336 (2021).
138.  Hong, J. W., Hendrix, D. A. &amp; Levine, M. S. Shadow enhancers as a source of evolutionary novelty. Science 321 , 1314 (2008).
139.  Thompson, J. J. et al. Extensive co-binding and rapid redistribution of NANOG and GATA6 during emergence of divergent lineages. Nat. Commun. 13 , 4257 (2022).
140.  de Boer, C. G. &amp; Taipale, J. Hold out the genome: a roadmap to solving the cis-regulatory code. Nature 625 , 41-50 (2024).
141. Gibney, E. R. &amp; Nolan, C. M. Epigenetics and gene expression. Heredity 105 , 4-13 (2010).
142.  Bolt, C. C. &amp; Duboule, D. The regulatory landscapes of developmental genes. Development 147 , dev171736 (2020).
143.  Blassberg, R. et al. Sox2 levels regulate the chromatin occupancy of WNT mediators in epiblast progenitors responsible for vertebrate body formation. Nat. Cell Biol. 24 , 633-644 (2022).
144.  Kim, S. et al. Deciphering the multi-scale, quantitative cis-regulatory code. Mol. Cell 83 , 373-392 (2023).
145.  Mayran, A. et al. Pioneer and nonpioneer factor cooperation drives lineage specific chromatin opening. Nat. Commun. 10 , 3807 (2019).
146.  Liu, B. B. et al. An automated ATAC-seq method reveals sequence determinants of transcription factor dose response in the open chromatin. Preprint at bioRxiv https://doi.org/10.1101/2025.07.24.666684 (2025).
147. Li, D. et al. Chromatin accessibility dynamics during iPSC reprogramming. Cell Stem Cell 21 , 819-833.e6 (2017).
148.  Chronis, C. et al. Cooperative binding of transcription factors orchestrates reprogramming. Cell 168 , 442-459.e20 (2017).
149.  Narita, T. et al. Acetylation of histone H2B marks active enhancers and predicts CBP/p300 target genes. Nat. Genet. 55 , 679-692 (2023).
150.  Lynch, M. The evolution of genetic networks by non-adaptive processes. Nat. Rev. Genet. 8 , 803-813 (2007).
151. Kosicki, M. et al. In vivo mapping of mutagenesis sensitivity of human enhancers. Nature 643 , 839-846 (2025).
152.  Koeppel, J. et al. Randomizing the human genome by engineering recombination between repeat elements. Science 387 , eado3979 (2025).
153.  Koeppel, J. et al. Resolution of a human super-enhancer by targeted genome randomisation. Preprint at bioRxiv https://doi.org/10.1101/2025.01.14.632548 (2025).
154.  Gasperini, M. et al. A genome-wide framework for mapping gene regulation via cellular genetic screens. Cell 176 , 377-390.e19 (2019).
155.  Liu, W. et al. Dissecting the impact of transcription factor dose on cell reprogramming heterogeneity using scTF-seq. Nat. Genet. 57 , 2522-2535 (2025).
156.  Frömel, R. et al. Design principles of cell-state-specific enhancers in hematopoiesis. Cell 188 , 3202-3218 (2025).
157. Lalanne, J.-B. et al. Multiplex profiling of developmental cis-regulatory elements with quantitative single-cell expression reporters. Nat. Methods 21 , 983-993 (2024).
158.  Cornwall-Scoones, J. et al. Predictable engineering of signal-dependent cis-regulatory elements. Preprint at bioRxiv https://doi.org/10.1101/2025.03.07.642002 (2025).
159.  Buckley, M. et al. Saturation genome editing maps the functional spectrum of pathogenic VHL alleles. Nat. Genet. 56 , 1446-1455 (2024).
160.  Zhang, P. et al. Deep flanking sequence engineering for efficient promoter design using DeepSEED. Nat. Commun. 14 , 6309 (2023).
161. Taskiran, I. I. et al. Cell-type-directed design of synthetic enhancers. Nature 626 , 212-220 (2024).

## This study exemplifies the emerging capability to engineer cis-regulatory sequences, supporting the vision of synthetic GRN construction.

162.  Gosai, S. J. et al. Machine-guided design of cell-type-targeting cis-regulatory elements. Nature 634 , 1211-1220 (2024).
163.  Gao, X. J., Chong, L. S., Kim, M. S. &amp; Elowitz, M. B. Programmable protein circuits in living cells. Science 361 , 1252-1258 (2018).
164.  Chen, Z. et al. A synthetic protein-level neural network in mammalian cells. Science 386 , 1243-1250 (2024).
165.  Green, A. A. et al. Complex cellular logic computation using ribocomputing devices. Nature 548 , 117-121 (2017).
166.  Chen, Y. et al. Genetic circuit design automation for yeast. Nat. Microbiol. 5 , 1349-1360 (2020).
167. Bunne, C. et al. How to build the virtual cell with artificial intelligence: priorities and opportunities. Cell 187 , 7045-7063 (2024).
168.  Johnson, G. T. et al. Building the next generation of virtual cells to understand cellular biology. Biophys. J. 122 , 3560-3569 (2023).
169.  Heimberg, G. et al. A cell atlas foundation model for scalable search of similar human cells. Nature 638 , 1085-1094 (2024).
170.  Abramson, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 630 , 493-500 (2024).
171. Booeshaghi, A. S., Galvez-Merchán, Á. &amp; Pachter, L. Algorithms for a Commons Cell Atlas. Preprint at bioRxiv https://doi.org/10.1101/2024.03.23.586413 (2024).
172.  Brixi, G. et al. Genome modeling and design across all domains of life with Evo 2. Preprint at bioRxiv https://doi.org/10.1101/2025.02.18.638918 (2025).
173.  Minsky, M. &amp; Papert, S. Perceptrons (MIT Press, 1969).

## Perspective

174.  Sáez, M. et al. Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions. Cell Syst. 13 , 12-28.e3 (2022).

175.  Sáez, M., Briscoe, J. &amp; Rand, D. A. Dynamical landscapes of cell fate decisions. Interface Focus 12 , 20220002 (2022).

176.  Rand, D. A., Raju, A., Sáez, M., Corson, F. &amp; Siggia, E. D. Geometry of gene regulatory dynamics. Proc. Natl Acad. Sci. USA 118 , e2109729118 (2021).

## Acknowledgements

The authors are grateful to T. Brown, J. DiFrisco, D. Erwin, F. Fröhlich, L. Parts, P. Badia-iMompel and the members of the Briscoe Lab for their constructive comments. This work was supported by the Wellcome Trust (220379/D/20/Z) and the Francis Crick Institute, which receives its core funding from Cancer Research UK, the UK Medical Research Council and the Wellcome Trust (all under FC001051).

## Author contributions

R.J.M. and J.B. researched the literature, R.J.M. wrote the article, and R.J.M. and J.B. contributed substantially to the discussion of the content, and reviewed and edited the manuscript before submission.

## Competing interests

R.J.M. is a consultant for Omnipotent Biotechnologies.

## Additional information

Peer review information Nature Reviews Genetics thanks Mohammad Lotfollahi and the other, anonymous, reviewer(s) for their contribution to the peer review of this work.

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

© Springer Nature Limited 2026