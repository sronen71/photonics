# LLE physics benchmark references

These are the publicly available author manuscripts used to define and check
the scalar and generalized Lugiato--Lefever equation benchmarks. Canonical
source links and checksums are recorded so the exact local versions remain
identifiable. Copyright remains with the respective authors and publishers.

## Godey et al. (2014), normal dispersion

- Local PDF: [Godey2014.pdf](Godey2014.pdf)
- Author manuscript: [arXiv:1308.2539v1](https://arxiv.org/abs/1308.2539v1)
- Published article: [Phys. Rev. A 89, 063814](https://doi.org/10.1103/PhysRevA.89.063814)
- Benchmarks: homogeneous response, normal-dispersion dark cavity soliton
- SHA-256: `5e5341f40ee263346f887182a5ee1e234b5bafefa62b9b3ce4095ae548570340`

## Godey et al. (2014), anomalous dispersion

- Local PDF: [Godey2014-anomalous.pdf](Godey2014-anomalous.pdf)
- Author manuscript: [arXiv:1308.2542v1](https://arxiv.org/abs/1308.2542v1)
- Published article: [Phys. Rev. A 89, 063814](https://doi.org/10.1103/PhysRevA.89.063814)
- Benchmark: anomalous-dispersion modulational-instability gain and mode selection
- SHA-256: `90835e30daab51c890c6cee515df49165315c93072d54a8a49bb3f96ccbbac16`

The two arXiv manuscripts are the normal- and anomalous-dispersion parts that
were combined in the published Physical Review A article.

## Coen and Erkintalo (2013)

- Local PDF: [CoenErkintalo2013.pdf](CoenErkintalo2013.pdf)
- Author manuscript: [arXiv:1303.7078v1](https://arxiv.org/abs/1303.7078v1)
- Published article: [Opt. Lett. 38, 1790](https://doi.org/10.1364/OL.38.001790)
- Benchmarks: bistability cubic and large-detuning bright-soliton scaling laws
- SHA-256: `fd2baee387f83d5644bc670ea9c8ed152f20556c2ccff62cad677e87bc5653e0`

## Chembo and Menyuk (2013)

- Local PDF: [ChemboMenyuk2013.pdf](ChemboMenyuk2013.pdf)
- Author manuscript: [arXiv:1210.8210v1](https://arxiv.org/abs/1210.8210v1)
- Published article: [Phys. Rev. A 87, 053852](https://doi.org/10.1103/PhysRevA.87.053852)
- Benchmarks: dimensional field normalization and generalized modal dispersion
- SHA-256: `aeb7ce5f74e4a46ddb65a706ca63d20f3d0b84026dee10bfe698d3ae9591e260`

## Parra-Rivas et al. (2014)

- Local PDF: [ParraRivas2014.pdf](ParraRivas2014.pdf)
- Author manuscript: [arXiv:1403.0903v1](https://arxiv.org/abs/1403.0903v1)
- Published article: [Opt. Lett. 39, 2971](https://doi.org/10.1364/OL.39.002971)
- Benchmark: third-order-dispersion stabilization of a moving cavity soliton
- SHA-256: `570c02a5347a307f5bc2dfa90ead45c1960e7e021ae2160da3e2a81139ff8f86`

## Zang et al. (2025), bidirectional PhCR circuit

- Local author manuscript: [Jizhao2401.16740v1.pdf](Jizhao2401.16740v1.pdf)
- Local published supplement:
  [Jizhao2024-supplemental.pdf](Jizhao2024-supplemental.pdf)
- Author manuscript: [arXiv:2401.16740v1](https://arxiv.org/abs/2401.16740v1)
- Published article: [Nature Photonics 19, 510--517](https://doi.org/10.1038/s41566-025-01624-1)
- Benchmarks: bidirectional coupled LLE, finite-band reflector, input-output
  power conservation, one-sided high-efficiency soliton, and reflector-phase
  contrast in Figs. 1e and S2
- Main-manuscript SHA-256:
  `ea624b3a2e9af9b4dad66737455a3b36565531a6fb0d0276a5bfe42b9eb21b17`
- Supplement SHA-256:
  `9c7fa29440756c35d798754d7b8d1ee81ef2695ab6bf490c37f7f6963a0dd3a1`

The arXiv manuscript predates the 2025 version of record. The local supplement
is the publisher-provided supplementary information for the final article.

## Liu et al. (2024), PhCR phase-convention cross-check

- NIST author copy: [publication PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=936538)
- Published article: [Phys. Rev. Lett. 132, 023801](https://doi.org/10.1103/PhysRevLett.132.023801)
- Cross-check: the displayed coupled LLE gives an antisymmetric red split mode,
  $E^f_0\simeq-E^b_0$, and the analysis identifies maximum saturated conversion
  at $r=-1$, or equation phase $\Phi=\pi$

This fixes the modal phase convention independently of the soliton benchmark.
The 2025 paper plots its constructive device-relative phase as zero while
retaining the same equation signs and $r=\sqrt R\exp(i\phi)$, so its plotted
phase axis is shifted by $\pi$ relative to the literal equation phase.

## Liu et al. (2025), solitons in photonic-crystal resonators

- Author manuscript: [arXiv:2508.13393v1](https://arxiv.org/abs/2508.13393v1)
- Benchmark: the normal-dispersion primary-comb threshold from Eq. (4) and
  Fig. 4 for $F=3$ and pump-mode splitting $\Gamma=9.3$
- Reproduction: the counter-propagating CW intensity reaches its normalized
  threshold of one at $\alpha=5.58$, compared with the paper's quoted
  $\alpha\simeq5.5$

The threshold depends on the homogeneous pump fields and not on the magnitude
of $D_2$. The later nonlinear evolution and spectral bandwidth are not used as
quantitative benchmarks because the manuscript does not specify that magnitude
for the plotted simulation.

## Sakaue and Kuse (2026), coupled-LLE analysis of PhCR solitons

- Open-access article: [Opt. Express 34, 22095](https://doi.org/10.1364/OE.599771)
- Benchmarks: the direction-resolved modulation-instability gain in Fig. 3 and
  the effective-LLE soliton existence boundaries in Fig. 6
- Reproduction: for $F^2=6$, $\Gamma=4$, and $d_2=0.02$, the unstable forward
  sidebands peak at $|\mu|=20$ with normalized gain 0.586; for $\Gamma=6.75$,
  the calculated forward interval is $[3.86,9.63]$ and the backward interval
  is $[2.18,6.36]$

Both papers couple only the forward and backward pump modes. In this repository
that limit is represented by zero external reflectivity and
`epsilon_phc` $=\Gamma$. The coupling sign differs by the harmless global phase
choice $E_b\mapsto-E_b$, while intensities, MI gains, and existence boundaries
are unchanged. Sakaue and Kuse use positive $d_2$ for anomalous dispersion;
with this repository's retained dispersion convention their case is
`beta = -d_2`.
