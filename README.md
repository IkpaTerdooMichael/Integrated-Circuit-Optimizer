# IC Optimizer

ML-driven analog circuit sizing using Bayesian optimization.

This project demonstrates how machine learning can be applied to integrated
circuit design-space exploration. It optimizes a 2-stage CMOS operational
amplifier for Power, Performance, and Area (PPA) trade-offs using Gaussian
process Bayesian optimization.

Built as a portfolio project for the Erasmus Mundus **Semiconductor Chips
Integration and Innovation (SSI2)** master's program.

## Features

- Interactive design targets (minimum DC gain, minimum gain-bandwidth product)
- Adjustable objective weights for power, area, and spec penalties
- Gaussian process Bayesian optimization (`gp_minimize` from scikit-optimize)
- Live convergence plot showing best cost per iteration
- PPA trade-off scatter plot colored by DC gain
- One-click CSV export of the full optimization history
- Runs anywhere with no EDA tool installation required

## Design Variables

| Variable | Description              | Range      | Unit |
|----------|--------------------------|------------|------|
| W1       | Input pair width         | 1 - 50     | um   |
| W3       | PMOS mirror width        | 1 - 50     | um   |
| W5       | Second stage PMOS width  | 1 - 100    | um   |
| W6       | Second stage NMOS width  | 1 - 100    | um   |
| Ibias    | Bias current             | 1 - 50     | uA   |

## Objective Function

The optimizer minimizes a weighted cost:

    cost = w_power * power_mw
         + w_area  * (area_um2 / 100)
         + w_pen   * spec_penalty

Where `spec_penalty` penalizes designs that miss the DC gain or GBW targets.
All three weights are adjustable in the sidebar.

## Performance Metrics

| Metric   | Description                          | Unit |
|----------|--------------------------------------|------|
| DC Gain  | Open-loop DC voltage gain            | dB   |
| GBW      | Gain-bandwidth product               | Hz   |
| Power    | Total static power dissipation       | mW   |
| Area     | Total transistor gate area           | um2  |

## How It Works

1. **Design variables** - transistor widths and bias current are the search space.
2. **Surrogate model** - a closed-form analytical model estimates gain, GBW,
   power, and area for any candidate sizing. This keeps the app dependency-free.
3. **Objective** - a scalar cost combines power, area, and spec-miss penalties.
4. **Optimizer** - Gaussian process Bayesian optimization efficiently samples
   the design space, balancing exploration and exploitation.

## Dependencies

- streamlit - web app framework
- numpy - numerical computation
- pandas - data handling
- scikit-optimize - Bayesian optimization
- matplotlib - plotting

## Extending to Production

The analytical model in `simulate_opamp()` is a stand-in for a real circuit
simulator. To make this production-grade, replace it with a SPICE call:

    from PySpice.Spice.Netlist import Circuit

    def simulate_opamp(W1, W3, W5, W6, Ibias):
        # build netlist, run ngspice, return parsed metrics
        ...

Add `PySpice>=1.5` to `requirements.txt` and install `ngspice` on the host.
For Streamlit Cloud, use `packages.txt` to install system dependencies:

    ngspice

## Possible Extensions

- Replace analytical model with ngspice or Xyce SPICE simulation
- Add a Graph Neural Network surrogate trained on real layout data
- Extend to multi-objective optimization (NSGA-II, Pareto front visualization)
- Support additional topologies (folded cascode, telescopic, two-stage Miller)
- Add Monte Carlo / corner analysis for yield estimation
- Integrate with OpenROAD for digital placement and routing optimization

 This project touches on:

- Analog circuit design fundamentals (op-amp topologies, PPA trade-offs)
- Design automation and optimization methodology
- Machine learning applied to engineering problems
- Reproducible, deployable software engineering practice

