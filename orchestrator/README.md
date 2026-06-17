# VOF Pressure-Projection Orchestrator

This directory contains a host-side orchestration class
(`Navatala::Cfd::VofPressureOrchestrator`) that composes the CFD kernel
sequence — a Volume-of-Fluid pressure-projection workflow — into a single
`execute(runTime, mesh)` entry point.

The class is offered as source for review and integration experiments; it
is not a turnkey public solver in this release. Plugging it into a CFD
host build requires the host's own mesh / time / field types and the
integration glue that wires them through the runtime; the constructor
accepts those types by reference but the code intentionally does not
ship the host-side adapters.

