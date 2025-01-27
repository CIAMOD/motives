# Motives 0.2.0

<p align="center">
  <img src="https://github.com/user-attachments/assets/aa6f0c16-2ae5-4d86-92f7-86d47ac6596f" />
</p>

[![pypi](https://img.shields.io/pypi/v/motives.svg)](https://pypi.python.org/pypi/motives)
[![PyPI Downloads](https://static.pepy.tech/badge/motives)](https://pepy.tech/projects/motives)
[![python](https://img.shields.io/badge/python-%5E3.10-blue)]()
[![os](https://img.shields.io/badge/OS-Ubuntu%2C%20Mac%2C%20Windows-purple)]()

Motives is a symbolic manipulation package based on SymPy, which handles motivic expressions in the Grothendieck ring of Chow
motives and other types of λ-rings. It is an easy to use library aimed to help researchers verify equations, simplify, and handle motivic expressions. Check the paper [Motives meet SymPy: studying λ-ring expressions in Python](https://arxiv.org/abs/2501.00563) for a more comprehensive description of the package.

Grothendieck's motive of a variety is an invariant that provides extensive information about its geometry. Manipulating motivic formulas and understanding when two distinct expressions can represent the same variety is sometimes complex but mathematically interesting. The goal of this package is to provide a comprehensive tool that allows efficient manipulation and comparison of motives, as well as expressions in other λ-rings. Furthermore, it contains the equations for the motives of some commonly used moduli spaces, so that the package can be applied easily to test hypothesis and conjectural results on their geometry.

## Methodology

The simplification algorithm is based on a refinement of the abstract simplification algorithm for λ-ring expressions developed in [Alfaya '22]. See [Sanchez, Alfaya and Pizarroso '24] for the complete mathematical description of the algorithms and equations used by the package.

The package works on λ-rings (R,λ,σ) under the following assumptions:

- R is a unital abelian ring with no additive torsion.
- There are two mutually oposite λ-ring structures on R, λ and σ.
- σ is a special λ-ring structure, and has associated Adams operations ψ. λ is not assumed to be special.

Based on these assumptions, the algorithm uses certain universal algebraic relations allowing to write λ-ring operations λ, σ and ψ in terms on each other and the fact that Adams operations are ring homomorphisms to convert any expression tree in a λ-ring as a polynomial in terms of a finite set of Adams operations ψ. This equivalent Adams polynomial can then be used for comparison, or be transformeed into a polynomial in λ-powers instead, which allows to obtain smaller polynomial representations of an expression when it depends of elements of small dimension in the σ λ-ring structure.

The package includes modules for working with the Grothendieck ring of motives, with its natural λ-ring structures yield by symmetric and alternated products. It contains modules for working with several commonly used motives, including the following:

- Complex algebraic curves. Implements the algebraic relations from [Heinloth '07].
- Jacobian and Picard varieties of curves. Implements the equations from [Heinloth '07].
- Symmetric and alternated products of any variety given its motive.
- Moduli spaces of vector bundles on curves. Implements the equations from [García-Prada, Heinloth and Shmitt '14], [Sánchez '14] and [del Baño '01].
- Moduli spaces of L-twisted Higgs bundles on curves. Implements the theorems from [Alfaya and Oliveira '24] and the conjectural formula from [Mozgovoy '12].
- Moduli spaces of chain bundles and variations of Hodge structure on curves in low rank. Implements the results in [García-Prada, Heinloth and Shmitt '14] and [Sánchez '14].
- Algebraic groups. Implements the formula from [Behrend and Dhillon '07].
- Moduli stacks of vector bundles and principal G-bundles on curves. Uses the conjectural formulas from [Behrend and Dhillon '07] (proven for SL(n,C))
- Classifying stacks BG for several groups G. Applies results from [Behrend and Dhillon '07], [Bergh '16] and [Dhillon and Young '16].

## Citation

If you use the "motives" package in your work, please cite the paper

Daniel Sanchez, David Alfaya and Jaime Pizarroso. [Motives meet SymPy: studying λ-ring expressions in Python](https://arxiv.org/abs/2501.00563). _arXiv:2501.00563_, 2025.

## Getting Started

### Prerequisites

- Python >= 3.10.6
- pytest==7.4.4
- sympy>=1.12
- tqdm==4.66.1
- typeguard==4.3.0

### Installation

1. Install the package from PyPI

   ```sh
   pip install motives
   ```

2. Import the package in your Python code
   ```python
   import motives
   ```

Alternatively, you can install the package from the source code:

1. Clone the repository
   ```sh
   git clone https://github.com/CIAMOD/motives.git
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
- Alejandro Martínez de Guinea García, Student of the Degree in Mathematical Engineering and Artificial Intelligence, Institute for Research in Technology, ICAI, Comillas Pontifical University
- Sergio Herreros Pérez, Student of the Degree in Mathematical Engineering and Artificial Intelligence, Institute for Research in Technology, ICAI, Comillas Pontifical University

## References

- [Alfaya '22] David Alfaya. Simplification of λ-ring expressions in the Grothendieck ring of Chow motives. _Applicable Algebra in Engineering, Communication and Computing_, 33:599–628, 2022.
- [Alfaya, Oliveira '24] David Alfaya and André Oliveira. Lie algebroid connections, twisted Higgs bundles and motives of moduli spaces.
  _Journal of Geometry and Physics_, 201:105195–1 – 105195–55, 2024.
- [Behrend and Dhillon '07] Kai Behrend and Ajneet Dhillon. On the motivic class of the stack of bundles. _Advances in Mathematics_,
  212(2):617–644, 2007.
- [Bergh '16] Daniel Bergh. Motivic classes of some classifying stacks. _Journal of the London Mathematical Society_, 93(1):219–
  243, 2016.
- [del Baño '01] Sebastian del Baño. On the chow motive of some moduli spaces. _Journal für die reine und angewandte Mathematik_,
  2001(532):105–132, 2001
- [Dhillon and Young '16] Ajneet Dhillon and Matthew B. Young. The motive of the classifying stack of the orthogonal group. _Michigan Math. J._, 65(1):189–197, 2016.
- [García-Prada, Heinloth and Schmitt '14] Oscar García-Prada, Jochen Heinloth, and Alexander Schmitt. On the motives of moduli of chains and Higgs bundles. _Journal of the European Mathematical Society_, 16:2617–2668, 2014.
- [Heinloth '07 ] Franziska Heinloth. A note on functional equations for zeta functions with values in Chow motives. _Ann. Inst. Fourier (Grenoble)_, 57(6):1927–1945, 2007.
- [Sánchez '14] Jonathan S´anchez. Motives of moduli spaces of pairs and applications. _PhD thesis, Universidad Complutense de Madrid_, Madrid, 2014.
- [Sanchez, Alfaya and Pizarroso '24] Daniel Sanchez, David Alfaya and Jaime Pizarroso. Motives meet SymPy: studying λ-ring expressions in Python. _arXiv:2501.00563_, 2025.
- [Mozgovoy '12] Sergey Mozgovoy. Solutions of the motivic ADHM recursion formula. _Int. Math. Res. Not. IMRN_, 2012(18):4218–4244, 2012.
