from .models import Problem

PROBLEMS = [
    Problem(
        id="even-sum",
        title="The sum of two even integers",
        topic="Direct proof",
        statement="Prove that the sum of any two even integers is even.",
        hint="Write each even integer as twice an integer.",
    ),
    Problem(
        id="odd-square",
        title="The square of an odd integer",
        topic="Direct proof",
        statement="Prove that the square of any odd integer is odd.",
        hint="Start with n = 2k + 1 and expand n².",
    ),
    Problem(
        id="consecutive-product",
        title="Consecutive integers",
        topic="Proof by cases",
        statement="Prove that n(n + 1) is even for every integer n.",
        hint="Consider separately whether n is even or odd.",
    ),
    Problem(
        id="even-square",
        title="If a square is even",
        topic="Contrapositive",
        statement="For any integer n, prove that if n² is even, then n is even.",
        hint="Prove the contrapositive: if n is odd, then n² is odd.",
    ),
    Problem(
        id="sum-formula",
        title="The first n positive integers",
        topic="Induction",
        statement="Prove that 1 + 2 + ⋯ + n = n(n + 1)/2 for every integer n ≥ 1.",
        hint="Check n = 1. Assume the formula for k, then add k + 1.",
    ),
    Problem(
        id="set-inclusion",
        title="An intersection is a subset",
        topic="Sets",
        statement="For any sets A and B, prove that A ∩ B ⊆ A.",
        hint="Take an arbitrary element of A ∩ B and use the definition of intersection.",
    ),
]
BY_ID = {problem.id: problem for problem in PROBLEMS}

# Trusted examples are context, not a required wording or proof method.
REFERENCES = {
    "even-sum": "Let a=2r and b=2s, r,s integers. a+b=2(r+s), hence even.",
    "odd-square": "n=2k+1 implies n²=4k²+4k+1=2(2k²+2k)+1, hence odd.",
    "consecutive-product": (
        "If n=2k, n(n+1)=2k(n+1). If n=2k+1, n+1=2(k+1). "
        "In either case the product is twice an integer."
    ),
    "even-square": (
        "Contrapositive: if n=2k+1 is odd, n²=2(2k²+2k)+1 is odd. Thus n² even implies n even."
    ),
    "sum-formula": (
        "Base n=1: 1=1·2/2. Assuming S_k=k(k+1)/2, "
        "S_(k+1)=S_k+k+1=(k+1)(k+2)/2. Induction proves the formula."
    ),
    "set-inclusion": (
        "For arbitrary x in A∩B, the intersection definition gives x in A and x in B, "
        "hence x in A. Therefore A∩B⊆A, including when the intersection is empty."
    ),
}
