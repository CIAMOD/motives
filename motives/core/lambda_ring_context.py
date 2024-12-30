import sympy as sp
from ..utils import all_partitions, multinomial_coeff, SingletonMeta
from functools import reduce


class LambdaRingContext(metaclass=SingletonMeta):
    """
    A class representing a lambda-ring context. This singleton class contains methods
    for computing the universal polynomials relating the lambda, sigma and Adams operations
    of a lambda-ring, as well as keeping a cache of the already computed polynomials.

    Attributes:
    -----------
    lambda_vars : list
        A list of SymPy symbols representing the lambda variables.
    sigma_vars : list
        A list of SymPy symbols representing the sigma variables.
    adams_vars : list
        A list of SymPy symbols representing the Adams variables.
    _lambda_2_sigma_pols : list
        A list of SymPy polynomials mapping lambda variables to sigma variables.
    _adams_2_lambda_pols : list
        A list of SymPy polynomials mapping Adams variables to lambda variables.
    _adams_2_sigma_pols : list
        A list of SymPy polynomials mapping Adams variables to sigma variables.
    _sigma_2_adams_pols : list
        A list of SymPy polynomials mapping sigma variables to Adams variables.
    _lambda_2_adams_pols : list
        A list of SymPy polynomials mapping lambda variables to Adams variables.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the lambda-ring context.

        Parameters:
        -----------
        mode : str
            The mode to use to compute the Adams to lambda polynomials.
            It can be "recurrent", "partitions", or "old".
        """
        self.lambda_vars: list[sp.Expr] = [sp.Integer(1)]
        self.sigma_vars: list[sp.Expr] = [sp.Integer(1)]
        self.adams_vars: list[sp.Expr] = [sp.Integer(1)]
        self._lambda_2_sigma_pols: list[sp.Expr] = []
        self._adams_2_lambda_pols: list[sp.Expr] = []
        self._adams_2_sigma_pols: list[sp.Expr] = []
        self._sigma_2_adams_pols: list[sp.Expr] = []
        self._lambda_2_adams_pols: list[sp.Expr] = []

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
        λ_n(x)=P(σ_1(x),...,σ_n(x)) and σ_n(x)=P(λ_1(x),...,λ_n(x))

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
        σ_n(x)=P(ψ_1(x),...,ψ_n(x))

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
        ψ_n(x)=P(σ_1(x),...,σ_n(x))

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
        λ_n(x)=P(ψ_1(x),...,ψ_n(x))

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
        ψ_n(x)=P(λ_1(x),...,λ_n(x))

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
        Computes the array of lambda to sigma polynomials up to degree n.
        λ_n(x)=P(σ_1(x),...,σ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._lambda_2_sigma_pols)

        self._lambda_2_sigma_pols += [1] * (n + 1 - previous_len)

        for k in range(max(1, previous_len), n + 1):
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
        Computes the array of Adams to sigma polynomials up to degree n using the recurrence relation for the
        inverse Newton polynomials.
        σ_n(x)=P(ψ_1(x),...,ψ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._adams_2_sigma_pols)

        self._adams_2_sigma_pols += [
            sp.Mul((-1) ** (i - 1), self.adams_vars[i])
            for i in range(previous_len, n + 1)
        ]

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
        Computes the array of Adams to sigma polynomials up to degree n using partitions.
        σ_n(x)=P(ψ_1(x),...,ψ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._adams_2_sigma_pols)

        self._adams_2_sigma_pols += [0] * (n + 1 - previous_len)

        for k in range(max(1, previous_len), n + 1):
            self._adams_2_sigma_pols[k] = sp.Add(
                *(
                    sp.Mul(
                        *(self.adams_vars[a] for a in partition),
                        sp.Pow(
                            reduce(
                                lambda x, y: x * y,
                                (
                                    sp.factorial(partition.count(i))
                                    for i in set(partition)
                                ),
                                1,
                            )
                            * reduce(lambda x, y: x * y, partition, 1),
                            -1,
                        ),
                        (-1) ** (len(partition) + k),
                    )
                    for partition in all_partitions(k)
                )
            )

    def _compute_sigma_2_adams_pols_partitions(self, n: int):
        """
        Computes the array of Adams to sigma polynomials up to degree n using partitions.
        σ_n(x)=P(ψ_1(x),...,ψ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._sigma_2_adams_pols)

        self._sigma_2_adams_pols += [0] * (n + 1 - previous_len)

        for k in range(max(1, previous_len), n + 1):
            self._sigma_2_adams_pols[k] = sp.Add(
                *(
                    sp.Mul(
                        *(self.sigma_vars[a] for a in partition),
                        self.sigma_vars[k - l],
                        (
                            multinomial_coeff(
                                [partition.count(i) for i in set(partition)]
                            )
                            if partition
                            else 1
                        )
                        * (-1) ** (len(partition) + k + 1)
                        * (k - l),
                    )
                    for l in range(k)
                    for partition in all_partitions(l)
                )
            )

    def _compute_sigma_2_adams_pols_recurrent(self, n: int):
        """
        Computes the array of sigma to Adams polynomials up to degree n using the recurrence relation for the
        Hirzebruch-Newton polynomials.
        ψ_n(x  (-1) ** (i - 1) * i * self.sigma_vars[i]
            for i in raeters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._sigma_2_adams_pols)

        self._sigma_2_adams_pols += [
            sp.Mul((-1) ** (i - 1), i, self.sigma_vars[i])
            for i in range(previous_len, n + 1)
        ]

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
        Computes the array of Adams to lambda polynomials up to degree n swapping the sigma variables for
        lambda variables.
        λ_n(x)=P(ψ_1(x),...,ψ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._adams_2_lambda_pols)

        self._adams_2_lambda_pols += [0] * (n + 1 - previous_len)

        for k in range(max(1, previous_len), n + 1):
            self._adams_2_lambda_pols[k] = sp.expand(
                self.get_lambda_2_sigma_pol(k).xreplace(
                    {
                        self.lambda_vars[i]: self.get_adams_2_sigma_pol(i)
                        for i in range(n + 1)
                    }
                )
            )

    def _compute_adams_2_lambda_pols_recurrent(self, n: int):
        """
        Computes the array of Adams to lambda polynomials up to degree n using the recurrence relation for the
        Adams to lambda polynomials.
        λ_n(x)=P(ψ_1(x),...,ψ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._adams_2_lambda_pols)

        self._adams_2_lambda_pols += [0] * (n + 1 - previous_len)

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
        Computes the array of Adams to lambda polynomials up to degree n from the formula:
        Σ[a1,...,ai ∈ p(n)] (Π_j (ψ_aj(x)/aj) * 1 / (#1s * #2s * ... * #ks))
        λ_n(x)=P(ψ_1(x),...,ψ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._adams_2_lambda_pols)

        self._adams_2_lambda_pols += [0] * (n + 1 - previous_len)

        for k in range(max(1, previous_len), n + 1):
            self._adams_2_lambda_pols[k] = sp.Add(
                *(
                    sp.Mul(
                        *(self.adams_vars[a] for a in partition),
                        sp.Pow(
                            reduce(
                                lambda x, y: x * y,
                                (
                                    sp.factorial(partition.count(i))
                                    for i in set(partition)
                                ),
                                1,
                            )
                            * reduce(lambda x, y: x * y, partition, 1),
                            -1,
                        ),
                    )
                    for partition in all_partitions(k)
                )
            )

    def _compute_lambda_2_adams_pols_recurrent(self, n: int):
        """
        Computes the array of lambda to Adams polynomials up to degree n swapping the sigma variables for
        lambda variables in the convolution.
        ψ_n(x)=P(λ_1(x),...,λ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._lambda_2_adams_pols)

        self._lambda_2_adams_pols += [0] * (n + 1 - previous_len)

        # Create the polynomials using the convolution ψ_n(x)=-Σ(-1)**i*i*σ_i(x)*λ_(n-i)(x))
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
        Computes the array of lambda to Adams polynomials up to degree n from partitions
        ψ_n(x)=P(λ_1(x),...,λ_n(x))

        Parameters:
        -----------
        n : int
            The maximum degree to compute the polynomials for.

        Returns:
            None
        """
        previous_len = len(self._lambda_2_adams_pols)

        self._lambda_2_adams_pols += [0] * (n + 1 - previous_len)

        for k in range(max(1, previous_len), n + 1):
            self._lambda_2_adams_pols[k] = sp.Add(
                *(
                    sp.Mul(
                        *(self.lambda_vars[a] for a in partition),
                        self.lambda_vars[k - l],
                        (
                            multinomial_coeff(
                                [partition.count(i) for i in set(partition)]
                            )
                            if partition
                            else 1
                        )
                        * (-1) ** len(partition)
                        * (k - l),
                    )
                    for l in range(k)
                    for partition in all_partitions(l)
                )
            )


if __name__ == "__main__":
    from time import perf_counter

    groth = LambdaRingContext()

    n = 15

    print("sigma to lambda polynomials: ")

    s_rec = perf_counter()
    res_rec = groth.get_lambda_2_sigma_pol(n)
    e_rec = perf_counter()

    # s_old = perf_counter()
    # res_old = groth_old.get_adams_2_sigma_pol(n)
    # e_old = perf_counter()

    # print(f"-----result rec-----\n{res_rec}")
    print(f"-----result-----\n{res_rec}")

    print(f"-----time rec----- {e_rec-s_rec}")
