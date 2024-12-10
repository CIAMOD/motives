# Motivic Decompositions of Vector Bundle and L-Higgs Bundle Moduli Spaces

![lambda image](https://github.com/CIAMOD/motivic_decomposition/assets/94676306/8c7f232a-6fe7-44df-9300-761e9003186c)

Grothendieck's motive of a variety is an invariant that provides extensive information about its geometry. Manipulating motivic formulas and understanding when two distinct expressions can represent the same variety is complex but mathematically interesting. The goal is to develop a software package that allows efficient manipulation and comparison of these motives, and apply it to low-rank proofs of the Mozgovoy Conjecture on the motive of the L-Higgs moduli.

## Methodology

In the first phase, classical techniques of symbolic computing and parallelization will be used to refine the expression simplification algorithm in λ-rings developed in [Alfaya '22], and to produce a comprehensive package for handling expressions in λ-rings. Subsequently, the software will be further enhanced with numerical and artificial intelligence techniques. On one hand, various numerical methods will be used to develop algorithms capable of ensuring the equality between complex motivic expressions without the need to fully expand them. The use of reinforcement learning and GANs will also be explored to develop an intelligent agent that can selectively apply λ and Adams operator transformations to two given expressions in order to prove their equivalence. These new systems will be applied, on one hand, to verify the Mozgovoy Conjecture [Mozgovoy '12] for the motive of the L-Higgs moduli space in previously unknown cases of rank 2 and 3, based on the formulas obtained in [Alfaya, Oliveira '21]. On the other hand, they will be used to construct geometric decompositions of the motive of vector bundle moduli spaces. In these research lines, collaboration is also expected with André Oliveira (U. Porto) and Kyoung-Seog Lee (U. Miami).

## Getting Started

### Prerequisites

- Python >= 3.11
- NumPy == 1.26.2
- SymPy == 1.12
- multipledispatch == 1.0.0
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

## Acknowledgments

Special thanks to everyone who contributed to the project:

- David Alfaya Sánchez (PI), Department of Applied Mathematics, ICAI, IIT
- Javier Rodrigo Hitos, Department of Applied Mathematics, ICAI
- Luis Ángel Calvo Pascual, Department of Quantitative Methods, ICADE
- Anitha Srinivasan, Department of Quantitative Methods, ICADE
- José Portela González, Department of Quantitative Methods, ICADE, IIT
- Jaime Pizarroso Gonzalo, Department of Telematics and Computing, ICAI
- Tomás Luis Gómez de Quiroga, Institute of Mathematical Sciences, UAM-UCM-UC3M-CSIC
- Daniel Sánchez Sánchez, Student of the Degree in Mathematical Engineering and Artificial Intelligence, ICAI
- Alejandro García Martínez de Guinea, Student of the Degree in Mathematical Engineering and Artificial Intelligence, ICAI
- Sergio Herreros Pérez, Student of the Degree in Mathematical Engineering and Artificial Intelligence, ICAI
