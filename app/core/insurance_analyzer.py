from typing import Dict, List
from dataclasses import dataclass

@dataclass
class InsurancePolicy:
    company: str
    product_name: str
    coverage: Dict[str, float]
    premium: float
    terms: Dict[str, str]

@dataclass
class CoverageComparison:
    policy_a: InsurancePolicy
    policy_b: InsurancePolicy
    similarities: List[str]
    differences: Dict[str, str]
    premium_difference: float

class InsuranceAnalyzer:
    def __init__(self):
        self.coverage_categories = [
            "accident", "hospitalization", "critical_illness", "death", "disability"
        ]

    def parse_policy(self, raw_text: str) -> InsurancePolicy:
        """Parse raw policy text into structured InsurancePolicy object"""
        # TODO: Implement text parsing logic
        return InsurancePolicy(
            company="",
            product_name="",
            coverage={},
            premium=0.0,
            terms={}
        )

    def compare_policies(self, policy_a: InsurancePolicy, policy_b: InsurancePolicy) -> CoverageComparison:
        """Compare two insurance policies and return comparison results"""
        similarities = []
        differences = {}
        premium_difference = abs(policy_a.premium - policy_b.premium)

        for category in self.coverage_categories:
            if category in policy_a.coverage and category in policy_b.coverage:
                a_cov = policy_a.coverage[category]
                b_cov = policy_b.coverage[category]
                
                if a_cov == b_cov:
                    similarities.append(category)
                else:
                    differences[category] = {
                        "policy_a": a_cov,
                        "policy_b": b_cov
                    }

        return CoverageComparison(
            policy_a=policy_a,
            policy_b=policy_b,
            similarities=similarities,
            differences=differences,
            premium_difference=premium_difference
        )
