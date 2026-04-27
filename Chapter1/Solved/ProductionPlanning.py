# PYTHON VERSION -- 3.8.20
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from docplex.mp.model import Model


# ---------------------------------------------------------------------------
# Data layer — pure value objects, no optimization logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Product:
    id: str
    name: str
    selling_price: float
    other_variable_cost: float
    market_demand_min: Optional[float]
    market_demand_max: Optional[float]
    raw_material_g: float
    labor_a_hours: float
    labor_b_hours: float
    machine_hours: float


@dataclass(frozen=True)
class Resource:
    id: str
    cost_per_unit: float
    availability_max: Optional[float]


@dataclass(frozen=True)
class Scenario:
    products: List[Product]
    resources: List[Resource]

    @classmethod
    def from_json(cls, path: Path) -> "Scenario":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        products = [
            Product(
                id=p["id"],
                name=p.get("name", p["id"]),
                selling_price=p["selling_price"],
                other_variable_cost=p.get("other_variable_cost", 0.0),
                market_demand_min=p.get("market_demand_min"),
                market_demand_max=p.get("market_demand_max"),
                raw_material_g=p["raw_material_g"],
                labor_a_hours=p["labor_a_hours"],
                labor_b_hours=p["labor_b_hours"],
                machine_hours=p.get("machine_hours", 0.0),
            )
            for p in raw["products"]
        ]

        resources = [
            Resource(
                id=key,
                cost_per_unit=val["cost_per_unit"],
                availability_max=val.get("availability_max"),
            )
            for key, val in raw["resources"].items()
        ]

        return cls(products=products, resources=resources)

    def resource_usage(self, product: Product, resource: Resource) -> float:
        """Units of resource consumed per unit of product produced."""
        usage_map: Dict[str, float] = {
            "raw_material": product.raw_material_g,
            "labor_a":      product.labor_a_hours,
            "labor_b":      product.labor_b_hours,
            "machine":      product.machine_hours,
        }
        return usage_map.get(resource.id, 0.0)


# ---------------------------------------------------------------------------
# Model layer — owns the docplex model, exposes clean accessors
# ---------------------------------------------------------------------------

class ProductionModel:
    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._model = Model(name="production_planning")
        self._vars: Dict[str, Any] = {}
        self._solution = None
        self._resource_costs = {r.id: r.cost_per_unit for r in scenario.resources}

    # --- build ---------------------------------------------------------------

    def build(self) -> None:
        self._create_variables()
        self._add_objective()
        self._add_resource_constraints()

    def _create_variables(self) -> None:
        for p in self._scenario.products:
            lb = p.market_demand_min if p.market_demand_min is not None else 0.0
            ub = p.market_demand_max if p.market_demand_max is not None else self._model.infinity
            self._vars[p.id] = self._model.continuous_var(lb=lb, ub=ub, name="make_%s" % p.id)

    def _unit_profit(self, product: Product) -> float:
        rc = self._resource_costs
        return (
            product.selling_price
            - product.raw_material_g * rc.get("raw_material", 0.0)
            - product.labor_a_hours  * rc.get("labor_a", 0.0)
            - product.labor_b_hours  * rc.get("labor_b", 0.0)
        )

    def _add_objective(self) -> None:
        self._model.maximize(
            self._model.sum(
                self._unit_profit(p) * self._vars[p.id]
                for p in self._scenario.products
            )
        )

    def _add_resource_constraints(self) -> None:
        for resource in self._scenario.resources:
            if resource.availability_max is None:
                continue
            self._model.add_constraint(
                self._model.sum(
                    self._scenario.resource_usage(p, resource) * self._vars[p.id]
                    for p in self._scenario.products
                ) <= resource.availability_max,
                ctname="%s_capacity" % resource.id,
            )

    # --- solve ---------------------------------------------------------------

    def solve(self) -> bool:
        with open("cplex.log", "w") as log_file:
            cplex = self._model.get_cplex()
            cplex.set_log_stream(log_file)
            cplex.set_results_stream(log_file)
            cplex.set_warning_stream(log_file)
            cplex.set_error_stream(log_file)
            self._solution = self._model.solve()

        if self._solution:
            self._solution.export("model.sol", format="sol")
        return self._solution is not None

    # --- accessors (valid only after a successful solve) ---------------------

    def production_qty(self, product_id: str) -> float:
        return self._vars[product_id].solution_value

    def unit_profit(self, product: Product) -> float:
        return self._unit_profit(product)

    @property
    def objective_value(self) -> float:
        return self._model.objective_value


# ---------------------------------------------------------------------------
# Output layer — formats and exports results, does not touch the docplex model
# ---------------------------------------------------------------------------

class SolutionReport:
    def __init__(self, scenario: Scenario, model: ProductionModel) -> None:
        self._scenario = scenario
        self._model = model

    def print_summary(self) -> None:
        print("\nOptimal production plan")
        print("-" * 90)
        print(
            "%-3s  %-30s %7s  %11s  %6s  %5s  %5s  %6s"
            % ("ID", "Name", "qty", "unit_profit", "rm_g", "A_h", "B_h", "mach_h")
        )
        print("-" * 90)

        resource_totals = {r.id: 0.0 for r in self._scenario.resources}
        total_profit = 0.0

        for p in self._scenario.products:
            qty = self._model.production_qty(p.id)
            profit_pu = self._model.unit_profit(p)
            total_profit += profit_pu * qty
            for r in self._scenario.resources:
                resource_totals[r.id] += self._scenario.resource_usage(p, r) * qty

            if qty <= 1e-6:
                continue

            min_tag = " [min=%g]" % p.market_demand_min if p.market_demand_min is not None else ""
            print(
                "%-3s  %-30s %7.2f  %11.2f  %6.2f  %5.2f  %5.2f  %6.2f"
                % (p.id, p.name + min_tag, qty, profit_pu,
                   p.raw_material_g, p.labor_a_hours, p.labor_b_hours, p.machine_hours)
            )

        print("-" * 90)
        for r in self._scenario.resources:
            cap = "%6.0f" % r.availability_max if r.availability_max is not None else "   n/a"
            print("%-18s: %7.2f / %s" % (r.id, resource_totals[r.id], cap))
        print("%-18s: $%d,.2f" % ("Objective value", self._model.objective_value))
        print("%-18s: $%d,.2f" % ("Recomputed profit", total_profit))

    def to_json(self, path: Path) -> None:
        production = [
            {
                "product_id": p.id,
                "name": p.name,
                "quantity": self._model.production_qty(p.id),
            }
            for p in self._scenario.products
        ]
        resource_usage = [
            {
                "resource_id": r.id,
                "usage": sum(
                    self._scenario.resource_usage(p, r) * self._model.production_qty(p.id)
                    for p in self._scenario.products
                ),
                "capacity": r.availability_max,
            }
            for r in self._scenario.resources
        ]
        output = {
            "objective_value": self._model.objective_value,
            "production": production,
            "resource_usage": resource_usage,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print("\nResults saved to '%s'" % path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    scenario = Scenario.from_json(Path(__file__).resolve().parents[2] / "Chapter1/Solved/input_data.json")

    model = ProductionModel(scenario)
    model.build()

    if not model.solve():
        print("No feasible solution found.")
        return

    report = SolutionReport(scenario, model)
    report.print_summary()
    report.to_json(Path("Chapter1/Solved/output_result.json"))


if __name__ == "__main__":
    main()
