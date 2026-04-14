"""Inverse design optimizer: pixel grid → target S-parameters.

Uses a trained CNN surrogate for fast fitness evaluation and
genetic algorithm for combinatorial optimization of binary grids.
Top candidates are validated with full OpenEMS simulation.
"""
