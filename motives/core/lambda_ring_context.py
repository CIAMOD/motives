import sympy as sp
from ..utils import all_partitions, multinomial_coeff, SingletonMeta
from functools import reduce


class LambdaRingContext(metaclass=SingletonMeta):
    """
    A class representing a lambda-ring context. This singleton class contains methods
    for computing the universal polynomials relating the lambda, sigma, and Adams operations
    of a lambda-ring, as well as keeping a cache of the already computed polynomials.

    Attributes:
    -----------
    lambda_vars : list[sp.Expr]
        A list of SymPy expressions representing the lambda variables.
    sigma_vars : list[sp.Expr]
        A list of SymPy expressions representing the sigma variables.
    adams_vars : list[sp.Expr]
        A list of SymPy expressions representing the Adams variables.
    _lambda_2_sigma_pols : dict[int, sp.Expr]
        A dictionary mapping lambda variable indices to sigma variable polynomials.
    _adams_2_lambda_pols : dict[int, sp.Expr]
        A dictionary mapping Adams variable indices to lambda variable polynomials.
    _adams_2_sigma_pols : dict[int, sp.Expr]
        A dictionary mapping Adams variable indices to sigma variable polynomials.
    _sigma_2_adams_pols : dict[int, sp.Expr]
        A dictionary mapping sigma variable indices to Adams variable polynomials.
    _lambda_2_adams_pols : dict[int, sp.Expr]
        A dictionary mapping lambda variable indices to Adams variable polynomials.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the lambda-ring context.
        """
        self.lambda_vars: list[sp.Expr] = [sp.Integer(1)]
        self.sigma_vars: list[sp.Expr] = [sp.Integer(1)]
        self.adams_vars: list[sp.Expr] = [sp.Integer(1)]
        self._lambda_2_sigma_pols: dict[int, sp.Expr] = {}
        self._adams_2_lambda_pols: dict[int, sp.Expr] = {}
        self._adams_2_sigma_pols: dict[int, sp.Expr] = {}
        self._sigma_2_adams_pols: dict[int, sp.Expr] = {}
        self._lambda_2_adams_pols: dict[int, sp.Expr] = {}

        self.mode = "partitions"

    def _generate_variables(self, n: int):
        """
        Generates the lambda, sigma, and Adams variables up to degree n.

        Parameters:
        -----------
        n : int
            The maximum degree to consider.

        Returns:
        --------
        None
        """
        if n < len(self.lambda_vars):
            return

        self.lambda_vars += [
            sp.Symbol(f"λ_{i}") for i in range(len(self.lambda_vars), n + 1)
        ]
        self.sigma_vars += [
            sp.Symbol(f"σ_{i}") for i in range(len(self.sigma_vars), n + 1)
        ]
        self.adams_vars += [
            sp.Symbol(f"ψ_{i}") for i in range(len(self.adams_vars), n + 1)
        ]

    def get_lambda_2_sigma_pol(self, n: int) -> sp.Expr:
        """
        Returns the polynomial that maps lambda variables to sigma variables and viceversa.
        That is, the following polynomial: p(λ_0, λ_1, ..., λ_n) = σ_n || p(σ_0, σ_1, ..., σ_n) = λ_n.

        Parameters:
        -----------
        n : int
            The degree of the polynomial to retrieve.

        Returns:
        --------
        sp.Expr
            The polynomial mapping lambda variables to sigma variables.
        """
        self._generate_variables(n)
        self._compute_lambda_2_sigma_pols(n)

        return self._lambda_2_sigma_pols[n]

    def get_adams_2_sigma_pol(self, n: int) -> sp.Expr:
        """
        Returns the polynomial that maps Adams variables to sigma variables.
        That is, the following polynomial: p(ψ_0, ψ_1, ..., ψ_n) = σ_n.

        Parameters:
        -----------
        n : int
            The degree of the polynomial to retrieve.

        Returns:
        --------
        sp.Expr
            The polynomial mapping Adams variables to sigma variables.
        """
        self._generate_variables(n)
        if self.mode == "recurrent":
            self._compute_adams_2_sigma_pols_recurrent(n)
        elif self.mode == "partitions":
            self._compute_adams_2_sigma_pols_partitions(n)
        else:
            self._compute_adams_2_sigma_pols_recurrent(n)

        return self._adams_2_sigma_pols[n]

    def get_sigma_2_adams_pol(self, n: int) -> sp.Expr:
        """
        Returns the polynomial that maps sigma variables to Adams variables.
        That is, the following polynomial: p(σ_0, σ_1, ..., σ_n) = ψ_n.

        Parameters:
        -----------
        n : int
            The degree of the polynomial to retrieve.

        Returns:
        --------
        sp.Expr
            The polynomial mapping sigma variables to Adams variables.
        """
        self._generate_variables(n)
        if self.mode == "recurrent":
            self._compute_sigma_2_adams_pols_recurrent(n)
        elif self.mode == "partitions":
            self._compute_sigma_2_adams_pols_partitions(n)
        else:
            self._compute_sigma_2_adams_pols_recurrent(n)

        return self._sigma_2_adams_pols[n]

    def get_adams_2_lambda_pol(self, n: int) -> sp.Expr:
        """
        Returns the polynomial that maps Adams variables to lambda variables.
        That is, the following polynomial: p(ψ_0, ψ_1, ..., ψ_n) = λ_n.

        Parameters:
        -----------
        n : int
            The degree of the polynomial to retrieve.

        Returns:
        --------
        sp.Expr
            The polynomial mapping Adams variables to lambda variables.
        """
        self._generate_variables(n)
        if self.mode == "recurrent":
            self._compute_adams_2_lambda_pols_recurrent(n)
        elif self.mode == "partitions":
            self._compute_adams_2_lambda_pols_partitions(n)
        else:
            self._compute_adams_2_lambda_pols_old(n)

        return self._adams_2_lambda_pols[n]

    def get_lambda_2_adams_pol(self, n: int) -> sp.Expr:
        """
        Returns the polynomial that maps lambda variables to Adams variables.
        That is, the following polynomial: p(λ_0, λ_1, ..., λ_n) = ψ_n.

        Parameters:
        -----------
        n : int
            The degree of the polynomial to retrieve.

        Returns:
        --------
        sp.Expr
            The polynomial mapping lambda variables to Adams variables.
        """
        self._generate_variables(n)
        if self.mode == "partitions":
            self._compute_lambda_2_adams_pols_partitions(n)
        elif self.mode == "recurrent":
            self._compute_lambda_2_adams_pols_recurrent(n)
        else:
            self._compute_lambda_2_adams_pols_recurrent(n)

        return self._lambda_2_adams_pols[n]

    def _compute_lambda_2_sigma_pols(self, n: int):
        """
        Computes the lambda to sigma polynomials up to degree n.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        self._lambda_2_sigma_pols[0] = 1

        for k in range(1, n + 1):
            if k in self._lambda_2_sigma_pols:
                continue

            self._lambda_2_sigma_pols[k] = sp.Add(
                *(
                    sp.Mul(
                        (-1) ** (k - i + 1),
                        self.lambda_vars[k - i],
                        self._lambda_2_sigma_pols[i],
                    )
                    for i in range(k)
                )
            )
            self._lambda_2_sigma_pols[k] = sp.expand_mul(self._lambda_2_sigma_pols[k])

    def _compute_adams_2_sigma_pols_recurrent(self, n: int):
        """
        Computes the Adams to sigma polynomials up to degree n using the recurrence relation.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        previous_len = len(self._adams_2_sigma_pols)

        self._adams_2_sigma_pols.update(
            (i, sp.Mul((-1) ** (i - 1), self.adams_vars[i]))
            for i in range(previous_len, n + 1)
        )

        for k in range(max(2, previous_len), n + 1):
            self._adams_2_sigma_pols[k] += sp.Add(
                *(
                    sp.Mul(
                        (-1) ** (i - 1),
                        self.adams_vars[i],
                        self._adams_2_sigma_pols[k - i],
                    )
                    for i in range(1, k)
                )
            )
            self._adams_2_sigma_pols[k] = sp.expand_mul(self._adams_2_sigma_pols[k] / k)

    def _compute_adams_2_sigma_pols_partitions(self, n: int):
        """
        Computes the Adams to sigma polynomials up to degree n using partitions.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        self._adams_2_sigma_pols[n] = sp.Add(
            *(
                sp.Mul(
                    *(self.adams_vars[a] for a in partition),
                    sp.Pow(
                        reduce(
                            lambda x, y: x * y,
                            (sp.factorial(partition.count(i)) for i in set(partition)),
                            1,
                        )
                        * reduce(lambda x, y: x * y, partition, 1),
                        -1,
                    ),
                    (-1) ** (len(partition) + n),
                )
                for partition in all_partitions(n)
            )
        )

    def _compute_sigma_2_adams_pols_partitions(self, n: int):
        """
        Computes the sigma to Adams polynomials up to degree n using partitions.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        self._sigma_2_adams_pols[n] = sp.Add(
            *(
                sp.Mul(
                    *(self.sigma_vars[a] for a in partition),
                    self.sigma_vars[n - l],
                    (
                        multinomial_coeff([partition.count(i) for i in set(partition)])
                        if partition
                        else 1
                    )
                    * (-1) ** (len(partition) + n + 1)
                    * (n - l),
                )
                for l in range(n)
                for partition in all_partitions(l)
            )
        )

    def _compute_sigma_2_adams_pols_recurrent(self, n: int):
        """
        Computes the sigma to Adams polynomials up to degree n using the recurrence relation.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        previous_len = len(self._sigma_2_adams_pols)

        self._sigma_2_adams_pols.update(
            (i, sp.Mul((-1) ** (i - 1), i, self.sigma_vars[i]))
            for i in range(previous_len, n + 1)
        )

        for k in range(max(2, previous_len), n + 1):
            self._sigma_2_adams_pols[k] -= sp.Add(
                *(
                    sp.Mul(
                        (-1) ** (k - i),
                        self.sigma_vars[k - i],
                        self._sigma_2_adams_pols[i],
                    )
                    for i in range(1, k)
                )
            )
            self._sigma_2_adams_pols[k] = sp.expand_mul(self._sigma_2_adams_pols[k])

    def _compute_adams_2_lambda_pols_old(self, n: int):
        """
        Computes the Adams to lambda polynomials up to degree n by substituting sigma variables with lambda variables.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        self._adams_2_lambda_pols[n] = sp.expand(
            self.get_lambda_2_sigma_pol(n).xreplace(
                {
                    self.lambda_vars[i]: self.get_adams_2_sigma_pol(i)
                    for i in range(n + 1)
                }
            )
        )

    def _compute_adams_2_lambda_pols_recurrent(self, n: int):
        """
        Computes the Adams to lambda polynomials up to degree n using the recurrence relation.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        previous_len = len(self._adams_2_lambda_pols)

        for k in range(max(1, previous_len), n + 1):
            self._adams_2_lambda_pols[k] = (
                self.adams_vars[k]
                - sp.Add(
                    *(
                        sp.Mul(
                            (-1) ** (k - i),
                            i,
                            self._adams_2_lambda_pols[i],
                            self.get_adams_2_sigma_pol(k - i),
                        )
                        for i in range(1, k)
                    )
                )
            ) / k

            self._adams_2_lambda_pols[k] = sp.expand_mul(self._adams_2_lambda_pols[k])

    def _compute_adams_2_lambda_pols_partitions(self, n: int):
        """
        Computes the Adams to lambda polynomials up to degree n using partitions.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        self._adams_2_lambda_pols[n] = sp.Add(
            *(
                sp.Mul(
                    *(self.adams_vars[a] for a in partition),
                    sp.Pow(
                        reduce(
                            lambda x, y: x * y,
                            (sp.factorial(partition.count(i)) for i in set(partition)),
                            1,
                        )
                        * reduce(lambda x, y: x * y, partition, 1),
                        -1,
                    ),
                )
                for partition in all_partitions(n)
            )
        )

    def _compute_lambda_2_adams_pols_recurrent(self, n: int):
        """
        Computes the lambda to Adams polynomials up to degree n using the recurrence relation.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        previous_len = len(self._lambda_2_adams_pols)

        for k in range(max(1, previous_len), n + 1):
            self._lambda_2_adams_pols[k] = -sp.Add(
                *(
                    sp.Mul(
                        (-1) ** i,
                        i,
                        self.lambda_vars[k - i],
                        self.get_lambda_2_sigma_pol(i),
                    )
                    for i in range(1, k + 1)
                )
            )
            self._lambda_2_adams_pols[k] = sp.expand_mul(self._lambda_2_adams_pols[k])

    def _compute_lambda_2_adams_pols_partitions(self, n: int):
        """
        Computes the lambda to Adams polynomials up to degree n using partitions.

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
        --------
        None
        """
        self._lambda_2_adams_pols[n] = sp.Add(
            *(
                sp.Mul(
                    *(self.lambda_vars[a] for a in partition),
                    self.lambda_vars[n - l],
                    (
                        multinomial_coeff([partition.count(i) for i in set(partition)])
                        if partition
                        else 1
                    )
                    * (-1) ** len(partition)
                    * (n - l),
                )
                for l in range(n)
                for partition in all_partitions(l)
            )
        )


if __name__ == "__main__":
    from time import perf_counter

    groth = LambdaRingContext()

    n = 15

    print("sigma to lambda polynomials: ")

    s_rec = perf_counter()
    res_rec = groth.get_sigma_2_adams_pol(n)
    e_rec = perf_counter()

    groth.mode = "recurrent"
    groth._sigma_2_adams_pols = {}

    s_old = perf_counter()
    res_old = groth.get_sigma_2_adams_pol(n)
    e_old = perf_counter()

    # print(f"-----result rec-----\n{res_rec}")
    print(f"-----result-----\n{res_rec - res_old}")

    print(f"-----time rec----- {e_rec - s_rec}")
