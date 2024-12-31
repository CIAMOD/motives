# Motivic Decompositions of Vector Bundle and L-Higgs Bundle Moduli Spaces

![lambda image](https://github.com/CIAMOD/motivic_decomposition/assets/94676306/8c7f232a-6fe7-44df-9300-761e9003186c)

Grothendieck's motive of a variety is an invariant that provides extensive information about its geometry. Manipulating motivic formulas and understanding when two distinct expressions can represent the same variety is complex but mathematically interesting. The goal of this package is to provide a comprehensive tool that allows efficient manipulation and comparison of these motives, as well as
expressions in other λ-rings.

In particular, the package contas one of our goals with this package is to study study open questions on the geometry of moduli spaces, like finding low-rank proofs of the Mozgovoy Conjecture on the motive of the L-Higgs moduli.

## Methodology

The simplification algorithm is based on a refinement of the abstract simplification algorithm for λ-ring expressions developed in [Alfaya '22]. See [Sanchez, Alfaya, Pizarroso '24] for the complete mathematical description of the algorithms and equations used by the package.

The package works on λ-rings under the following assumptions:
- R is a unital abelian ring with no additive torsion.
- There are two mutually oposite λ-ring structures on R, λ and σ.
- σ is a special λ-ring structure, and has associated Adams operations ψ. λ is not assumed to be special.

Based on these assumptions, the algorithm uses certain universal algebraic relations allowing to write λ-ring operations λ, σ and ψ in terms on each other and the fact that Adams operations are ring homomorphisms to convert any expression tree in a λ-ring as a polynomial in terms of a finite set of Adams operations ψ. This equivalent Adams polynomial can then be used for comparison, or be transformeed into a polynomial in λ-powers instead, which allows to obtain smaller polynomial representations of an expression when it depends of elements of small dimension in the σ λ-ring structure.

## References
- [Alfaya '22] David Alfaya. Simplification of λ-ring expressions in the Grothendieck ring of Chow motives. __Applicable Algebra in Engineering, Communication and Computing__, 33:599–628, 2022.
- [Alfaya, Oliveira '21] David Alfaya and André Oliveira. Lie algebroid connections, twisted Higgs bundles and motives of moduli spaces.
__Journal of Geometry and Physics__, 201:105195–1 – 105195–55, 2024.
- [Sanchez, Alfaya, Pizarroso '24] Daniel Sanchez, David Alfaya and Jaime Pizarroso. Motives meet SymPy: studying λ-ring expressions in Python. __in preparation__, 2024.
- [Mozgovoy '12] Sergey Mozgovoy. Solutions of the motivic ADHM recursion formula. __Int. Math. Res. Not. IMRN__, 2012(18):4218–
4244, 2012.


## Getting Started

### Prerequisites

- Python >= 3.10.6
- NumPy >= 1.26.2
- SymPy >= 1.12
- gmpy2 == 2.2.1
- tqdm == 4.66.5
- typeguard == 4.3.0

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/CIAMOD/motivic_decomposition.git
   ```
2. Install the required packages
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Instructions on how to run the software or code will be updated soon.

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE.md](LICENSE.md) file in each repository for details.

## Authors
- Daniel Sánchez Sánchez, Student of the Degree in Mathematical Engineering and Artificial Intelligence, ICAI, Comillas Pontifical University
- David Alfaya Sánchez, Department of Applied Mathematics and Institute for Research in Technology, ICAI, Comillas Pontifical University
- Jaime Pizarroso Gonzalo, Department of Telematics and Computing and Institute for Research in Technology, ICAI, Comillas Pontifical University

## Acknowledgments
This research was supported by project CIAMOD (Applications of computational methods and artificial intelligence to the study of moduli spaces, project PP2023_9) funded by Convocatoria de Financiación de Proyectos de Investigación Propios 2023, Universidad Pontificia Comillas, and by grants PID2022-142024NB-I00 and RED2022-134463-T funded by MCIN/AEI/10.13039/501100011033.

Find more about the CIAMOD project in the [project webpage](https://ciamod.github.io/) and the [IIT proyect webpage](https://www.iit.comillas.edu/publicacion/proyecto/en/CIAMOD/Aplicaciones_de_m%c3%a9todos_computacionales_y_de_inteligencia_artificial_al_estudio_de_espacios_de_moduli).

Special thanks to everyone who contributed to the project:
- David Alfaya Sánchez (PI), Department of Applied Mathematics and Institute for Research in Technology, ICAI, Comillas Pontifical University
- Javier Rodrigo Hitos, Department of Applied Mathematics, ICAI, Comillas Pontifical University
- Luis Ángel Calvo Pascual, Department of Quantitative Methods, ICADE, Comillas Pontifical University
- Anitha Srinivasan, Department of Quantitative Methods, ICADE, Comillas Pontifical University
- José Portela González, Department of Quantitative Methods, ICADE, IIT, Comillas Pontifical University
- Jaime Pizarroso Gonzalo, Department of Telematics and Computing and Institute for Research in Technology, ICAI, Comillas Pontifical University
- Tomás Luis Gómez de Quiroga, Institute of Mathematical Sciences, UAM-UCM-UC3M-CSIC
- Daniel Sánchez Sánchez, Student of the Degree in Mathematical Engineering and Artificial Intelligence, Institute for Research in Technology, ICAI, Comillas Pontifical University
- Alejandro García Martínez de Guinea, Student of the Degree in Mathematical Engineering and Artificial Intelligence, Institute for Research in Technology, ICAI, Comillas Pontifical University
- Sergio Herreros Pérez, Student of the Degree in Mathematical Engineering and Artificial Intelligence, Institute for Research in Technology, ICAI, Comillas Pontifical University
